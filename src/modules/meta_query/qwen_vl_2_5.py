import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

import os, sys
sys.path.append(os.getcwd())

from tqdm import tqdm
from typing import Optional
from einops import rearrange
from safetensors.torch import load_file, safe_open
from transformers import AutoProcessor, Qwen2_5_VLConfig

# from src.modules.vlm.models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
# from src.modules.meta_query.modeling_qwen2_5_vl_with_save import Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from src.modules.meta_query.qwen_vl_utils import process_vision_info

def _from_pretrained_kwargs(torch_dtype, device):
    """
    DeepSpeed ZeRO-3 is not compatible with passing `device_map` into HF `from_pretrained`.
    For ZeRO-3 we let DeepSpeed handle parameter placement/sharding.
    """
    kwargs = {"torch_dtype": torch_dtype}
    if not is_deepspeed_zero3_enabled():
        kwargs["device_map"] = device
    return kwargs

def _infer_num_embeddings(model, config):
    # Under ZeRO-3 the embedding parameters can be sharded/unavailable; prefer config when needed.
    try:
        emb = model.get_input_embeddings()
        if emb is not None and getattr(emb, "num_embeddings", None):
            return int(emb.num_embeddings)
    except Exception:
        pass

    for attr in ("vocab_size",):
        if getattr(getattr(config, "text_config", None), attr, None) is not None:
            return int(getattr(config.text_config, attr))
        if getattr(config, attr, None) is not None:
            return int(getattr(config, attr))

    raise ValueError("Unable to infer num_embeddings from model/config")

def _infer_embedding_dim(model, config):
    # 1. 优先从配置对象中查找 (Static & Reliable)
    # 按照优先级顺序定义查找路径
    lookup_paths = [
        (config, "text_config", "hidden_size"),
        (config, "hidden_size"),
        (model.config if hasattr(model, "config") else None, "text_config", "hidden_size"),
        (model.config if hasattr(model, "config") else None, "hidden_size"),
    ]

    for root, *attrs in lookup_paths:
        val = root
        for attr in attrs:
            val = getattr(val, attr, None)
            if val is None:
                break
        if val is not None:
            return int(val)

    # 2. 只有配置找不到时，才尝试读取模型权重 (Dynamic & Risky in ZeRO-3)
    try:
        emb = model.get_input_embeddings()
        # 注意：在 ZeRO-3 下，为了安全获取真实形状，最好使用 ds_shape
        # 但如果是初始化阶段，直接读取 config 才是王道
        w = getattr(emb, "weight", None)
        if w is not None and hasattr(w, "ds_shape"): # DeepSpeed 提供的原始形状属性
            return int(w.ds_shape[1])
        if w is not None and hasattr(w, "shape") and len(w.shape) >= 2:
            return int(w.shape[1])
    except Exception:
        pass

    raise ValueError("Unable to infer embedding_dim from model/config")

def load_state_dict(model_dir, postfix=".safetensors"):
    chunk_path_list = [os.path.join(model_dir, name) for name in os.listdir(model_dir) if name.endswith(postfix)]
    chunk_length = len(chunk_path_list)

    state_dict = {}
    for chunk_path in tqdm(chunk_path_list, total=chunk_length):
        if postfix == ".safetensors":
            chunk_state_dict = load_file(chunk_path, device="cpu")
        else:
            chunk_state_dict = torch.load(chunk_path, map_location="cpu")
        if "module" in chunk_state_dict.keys():
            chunk_state_dict = chunk_state_dict["module"]
        state_dict.update(chunk_state_dict)

    return state_dict

def parse_mtss_interleave(mtss_caption):
    """Parse the MTSS interleave caption to extract entity initial character positions.

    Extracts PERSON_xxx, OBJECT_xxx, SCENE_xxx entries from the
    [Cast & Setting Introduction] section and returns a dict mapping
    each entity name to its first occurrence character index in mtss_caption.

    Args:
        mtss_caption: The video_caption string in MTSS format.

    Returns:
        dict: {entity_name: char_index}, e.g. {"PERSON_1": 120, "OBJECT_1": 450, ...}
    """
    import re

    # Extract text between [Cast & Setting Introduction] and [Shot Narrative Script]
    section_pattern = r'\[Cast & Setting Introduction\](.*?)\[Shot Narrative Script\]'
    section_match = re.search(section_pattern, mtss_caption, re.DOTALL)
    if not section_match:
        return {}

    section = section_match.group(1)
    section_start = section_match.start(1)

    # Find all PERSON_xxx, OBJECT_xxx, SCENE_xxx entity names with positions.
    # Only accept an entity if it is preceded by "\n\n" (paragraph boundary),
    # which filters out spurious mentions embedded inside descriptions.
    entity_pattern = r'(?:PERSON|OBJECT|SCENE)_\d+'
    seen = {}
    for m in re.finditer(entity_pattern, section):
        entity = m.group()
        if entity in seen:
            continue
        # Check that the two characters immediately before the match are "\n\n"
        local_start = m.start()
        if local_start >= 2 and section[local_start - 2:local_start] == "\n\n":
            seen[entity] = section_start + local_start
        # Also accept if the entity is at the very beginning of the section
        # (possibly after only whitespace)
        elif section[:local_start].strip() == "":
            seen[entity] = section_start + local_start

    return seen

def split_mtss_interleave(mtss_caption, interleave_result):
    """Split mtss_caption into N+2 segments based on entity positions.

    Given N entities (PERSON_xxx, OBJECT_xxx, SCENE_xxx) in the
    [Cast & Setting Introduction] section, the caption is split into N+2
    ordered segments:
      - "head":        everything up to and including "[Cast & Setting Introduction]"
      - "PERSON_xxx" / "OBJECT_xxx" / "SCENE_xxx":  each segment starts with the
                        entity name and extends to the next entity (text_after holds
                        the description)
      - "tail":        from "[Shot Narrative Script]" to the end of the caption

    Args:
        mtss_caption: The full video_caption string in MTSS format.
        interleave_result: Dict from parse_mtss_interleave,
            e.g. {"PERSON_1": 512, "OBJECT_1": 914, "SCENE_1": 1100, ...}

    Returns:
        dict (insertion-ordered): Keys are "head", entity names (OBJECT_xxx /
              PERSON_xxx / SCENE_xxx), and "tail".  Each value is a dict:
              {
                  "head":     {"text_after": "...up to [Cast & Setting Introduction]..."},
                  "PERSON_1": {"text_after": "...: A young woman wearing ..."},
                  "OBJECT_1": {"text_after": "...: A red sports car ..."},
                  "SCENE_1":  {"text_after": "...: An indoor room ..."},
                  "tail":     {"text_after": "...[Shot Narrative Script]...rest..."},
              }
              Returns {"head": {"text_after": mtss_caption}} when
              interleave_result is empty or section markers are missing.
    """

    # --- locate the two section markers --------------------------------
    head_marker = "[Cast & Setting Introduction]"
    tail_marker = "[Shot Narrative Script]"

    head_end = mtss_caption.find(head_marker)
    tail_start = mtss_caption.find(tail_marker)

    # Fallback: markers not found → return the whole caption as head
    if head_end == -1 or tail_start == -1 or not interleave_result:
        return {"head": {"text_after": mtss_caption}}

    head_end += len(head_marker)  # inclusive of the marker itself

    # --- sort all entities by position (PERSON / OBJECT / SCENE) -------
    sorted_entities = sorted(interleave_result.items(), key=lambda x: x[1])

    if not sorted_entities:
        return {"head": {"text_after": mtss_caption}}

    # --- build result dict (N+2 segments) ------------------------------
    result = {}

    # 1) head: from start to end of "[Cast & Setting Introduction]"
    result["head"] = {"text_after": mtss_caption[:head_end]}

    # 2) middle N segments: each starts at entity, text_after until next
    #    split point (next entity or tail_marker)
    for i, (entity_name, entity_pos) in enumerate(sorted_entities):
        entity_end = entity_pos + len(entity_name)
        if i + 1 < len(sorted_entities):
            next_pos = sorted_entities[i + 1][1]
            text_after = mtss_caption[entity_end:next_pos]
        else:
            # last entity → text_after goes up to tail_marker
            text_after = mtss_caption[entity_end:tail_start]
        result[entity_name] = {"text_after": text_after}

    # 3) tail: from "[Shot Narrative Script]" to the end
    result["tail"] = {"text_after": mtss_caption[tail_start:]}

    return result

class Tokenizer_v4(nn.Module):
    def __init__(self, 
                 processor, 
                 max_edge=384, 
                 max_aspect_ratio=1.75, ):
        super().__init__()
        self.max_edge = max_edge
        self.max_aspect_ratio = max_aspect_ratio
        self.processor = processor

    def tensor_to_pil_and_resize(self, input_tensor, padding_info=None):
        """Convert the torch.Tensor to PIL.Image and apply resize
        """
        if padding_info is not None:
            unpad_height = input_tensor.size(1) - padding_info[1].item()
            unpad_width = input_tensor.size(2) - padding_info[2].item()
            input_tensor = input_tensor[:, :int(unpad_height), :int(unpad_width)]
            
        _, h, w = input_tensor.shape
        if h < w and w > self.max_edge:
            scale = w / self.max_edge
            w = self.max_edge
            h = int(h / scale)
        elif w < h and h > self.max_edge:
            scale = h / self.max_edge
            h = self.max_edge
            w = int(w / scale)
        pil_image = to_pil_image((input_tensor.float() + 1.0) / 2.0)
        pil_image = pil_image.resize((w, h))
        return pil_image

    def tokenize(self, video_prompts, condition_images, ref_ids, padding_info=None):
        """Tokenize the text prompts and condition images into tokens
        Args:
            video_prompts(Str): video prompt
            condition_images(Dict):
        """

        messages = [
            {
                "role": "system", 
                "content": [
                    {
                        "type": "text", 
                        "text": "Analyze the input images to create a rich, detailed visual description (age, gender, appearance, attire, accessories, environment, objects). Use this analysis to fully enrich and refine the corresponding parts of the existing textual description. Finally, generate a new, high-fidelity video that seamlessly presents this unified textual and visual information. The video must feature high, complex motion while maintaining absolute semantic and appearance consistency across all elements described.",
                    }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text", 
                        "text": video_prompts
                    }
                ]
            }
        ]
                
        for key, val in condition_images.items():
            if "ff" in key:
                messages.append(
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text", 
                                "text": "The first frame looks like: "
                            }, 
                            {
                                "type": "image", 
                                "image": self.tensor_to_pil_and_resize(val, padding_info=padding_info),
                            }
                        ]
                    }
                )
            if "lf" in key:
                messages.append(
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text", 
                                "text": "The last frame looks like: "
                            }, 
                            {
                                "type": "image", 
                                "image": self.tensor_to_pil_and_resize(val, padding_info=padding_info),
                            }
                        ]
                    }
                )
            if "ref" in key:
                if isinstance(val, list):
                    valid_num = len(val) - padding_info[0].item() if padding_info is not None else len(val)
                    ref_ids = ref_ids.view(-1)
                    img_id = 0
                    for refid, ref_img in zip(ref_ids, val[:valid_num]):
                        obj_type = ""
                        if refid < 100:
                            obj_type = "PERSON"
                            pass
                        elif refid >= 100 and refid < 200:
                            obj_type = "OBJECT"
                            refid = refid % 100
                        elif refid >= 200:
                            obj_type = "SCENE"
                            refid = refid % 200
                        
                        messages.append(
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "text", 
                                        "text": f"The [{obj_type}_{refid.item()}] looks like: "
                                    }, 
                                    {
                                        "type": "image", 
                                        "image": self.tensor_to_pil_and_resize(val[img_id], padding_info=padding_info),
                                    }
                                ]
                            }
                        )
                        img_id += 1
                else:
                    messages.append(
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "text", 
                                    "text": "The [PERSON_1] looks like: "
                                }, 
                                {
                                    "type": "image", 
                                    "image": self.tensor_to_pil_and_resize(val, padding_info=padding_info),
                                }
                            ]
                        }
                    )
            if "vid" in key:
                # messages += f"<|im_start|>The source video looks like: <|vision_start|><|image_pad|><|vision_end|><|im_end|>"
                pass
        
        # # DEBUG: 
        # messages.append(
        #     {
        #         "role": "user", 
        #         "content": [
        #             {
        #                 "type": "text", 
        #                 "text": video_prompts
        #             }, 
        #         ]
        #     }
        # )

        chat_template = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
        )

        # chat_template = chat_template + suffix
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_template], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt", 
        )   # token of image: 151655
        # inputs = inputs.to(self.model.device)
        return inputs

    def forward(self, 
                video_caption, 
                condition_images, 
                ref_ids, 
                return_mask=True,           # not used, just to keep args compatibility
                add_special_tokens=True,    # not used, just to keep args compatibility
                padding_info=None
    ):
        """Tokenize the input video caption and condition images. 
        The function's input args are designed to compatible with T5EncoderModel's tokenizer as much as possible.
        """
        inputs = self.tokenize(video_prompts=video_caption, 
                               condition_images=condition_images, 
                               ref_ids=ref_ids, 
                               padding_info=padding_info)
        return inputs

class Tokenizer_v4_mtss(nn.Module):
    def __init__(self, 
                 processor, 
                 max_edge=384, 
                 max_aspect_ratio=1.75, ):
        super().__init__()
        self.max_edge = max_edge
        self.max_aspect_ratio = max_aspect_ratio
        self.processor = processor

    def tensor_to_pil_and_resize(self, input_tensor, padding_info=None):
        """Convert the torch.Tensor to PIL.Image and apply resize
        """
        if padding_info is not None:
            unpad_height = input_tensor.size(1) - padding_info[1].item()
            unpad_width = input_tensor.size(2) - padding_info[2].item()
            input_tensor = input_tensor[:, :int(unpad_height), :int(unpad_width)]
            
        _, h, w = input_tensor.shape
        if h < w and w > self.max_edge:
            scale = w / self.max_edge
            w = self.max_edge
            h = int(h / scale)
        elif w < h and h > self.max_edge:
            scale = h / self.max_edge
            h = self.max_edge
            w = int(w / scale)
        pil_image = to_pil_image((input_tensor.float() + 1.0) / 2.0)
        pil_image = pil_image.resize((w, h))
        return pil_image

    def tokenize(self, video_prompts, condition_images, ref_ids, padding_info=None):
        """Tokenize the text prompts and condition images into tokens
        Args:
            video_prompts(Str): video prompt
            condition_images(Dict):
        """
        # Get Global Rank (unique across all nodes)
        global_rank = int(os.environ.get("RANK", 0))
        # Get Local Rank (unique within the current node)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        def parse_refid(refkey):
            import re
            # Extract the first number found anywhere in refkey
            num_match = re.search(r'\d+', refkey)
            refid = int(num_match.group()) if num_match else 1

            reftype = refkey.lower()
            if "person" in reftype:
                return refid
            elif "object" in reftype:
                return refid + 100
            elif "scene" in reftype:
                return refid + 200

        def check_reference_validity(ref_ids, video_prompts):
            """Check if the reference is valid in the video prompt
            """
            if isinstance(ref_ids, torch.Tensor):
                ref_ids = ref_ids.view(-1).tolist()
            valid_cnt = 0
            invalid_cnt = 0
            for ref_id in ref_ids:
                if ref_id < 100:
                    refkey = f"PERSON_{ref_id}"
                elif ref_id >= 100 and ref_id < 200:
                    refkey = f"OBJECT_{ref_id % 100}"
                elif ref_id >= 200:
                    refkey = f"SCENE_{ref_id % 200}"
                if refkey in video_prompts:
                    valid_cnt += 1
                else:
                    invalid_cnt += 1
            return valid_cnt > 0

        messages = [
            {
                "role": "system", 
                "content": [
                    {
                        "type": "text", 
                        "text": "Analyze the input images to create a rich, detailed visual description (age, gender, appearance, attire, accessories, environment, objects). Use this analysis to fully enrich and refine the corresponding parts of the existing textual description. Finally, generate a new, high-fidelity video that seamlessly presents this unified textual and visual information. The video must feature high, complex motion while maintaining absolute semantic and appearance consistency across all elements described.",
                    }
                ]
            },
            # {
            #     "role": "user", 
            #     "content": [
            #         {
            #             "type": "text", 
            #             "text": video_prompts
            #         }
            #     ]
            # }
        ]

        if isinstance(video_prompts, list):
            video_prompts = video_prompts[0]
        interleave_results = parse_mtss_interleave(video_prompts)
        reference_validity = check_reference_validity(ref_ids, video_prompts)
        
        append_valid_references = False
        if interleave_results == {} or not reference_validity:
            if not reference_validity:
                print(f"⚠️⚠️⚠️ [information] global_rank: {global_rank}, local_rank: {local_rank}, failed to assign any reference images with caption, we append them all at once.")
            for key, val in condition_images.items():
                if "ref" in key:
                    if isinstance(val, list):
                        valid_num = len(val) - padding_info[0].item() if padding_info is not None else len(val)
                        if isinstance(ref_ids, torch.Tensor):
                            ref_ids = ref_ids.view(-1)
                        img_id = 0
                        for refid, ref_img in zip(ref_ids, val[:valid_num]):
                            obj_type = ""
                            if refid < 100:
                                obj_type = "PERSON"
                                pass
                            elif refid >= 100 and refid < 200:
                                obj_type = "OBJECT"
                                refid = refid % 100
                            elif refid >= 200:
                                obj_type = "SCENE"
                                refid = refid % 200
                            
                            messages.append(
                                {
                                    "role": "user", 
                                    "content": [
                                        {
                                            "type": "text", 
                                            "text": f"The {obj_type}_{refid} looks like: "
                                        }, 
                                        {
                                            "type": "image", 
                                            "image": self.tensor_to_pil_and_resize(val[img_id], padding_info=padding_info),
                                        }
                                    ]
                                }
                            )
                            img_id += 1
                            append_valid_references = True
                    else:
                        messages.append(
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "text", 
                                        "text": "The PERSON_1 looks like: "
                                    }, 
                                    {
                                        "type": "image", 
                                        "image": self.tensor_to_pil_and_resize(val, padding_info=padding_info),
                                    }
                                ]
                            }
                        )
                        append_valid_references = True
        else:
            split_interleaves = split_mtss_interleave(video_prompts, interleave_results)
            
            # global style
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": split_interleaves["head"]["text_after"]
                        }
                    ]
                }
            )

            del split_interleaves["head"]

            # Build messages from the dict: for each entity, add text_before + entity image
            if isinstance(ref_ids, torch.Tensor):
                ref_ids = ref_ids.view(-1).tolist()

            # Interleave the text and image
            for entity_name, segments in split_interleaves.items():
                if "tail" in entity_name:
                    continue
                entity_id = parse_refid(entity_name)
                # print('---->',entity_id)
                if entity_id in ref_ids:
                    index = ref_ids.index(entity_id)
                    entity_image = condition_images["ref"][index]
                    # image
                    messages.append(
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "text", 
                                    "text": f"The {entity_name} looks like: "
                                }, 
                                {
                                    "type": "image", 
                                    "image": self.tensor_to_pil_and_resize(entity_image, padding_info=padding_info),
                                }
                            ]
                        }
                    )
                    append_valid_references = True
                    # textual description
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"{entity_name}\n{segments['text_after']}"
                                }
                            ]
                        }
                    )
                    # print(f"💡💡💡 [INFO] add vision and text description to VLM")
                else:
                    # textual description
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"{entity_name}\n{segments['text_after']}"
                                }
                            ]
                        }
                    )
                    # print(f"💡💡💡 [INFO] add text description to VLM")

            if not append_valid_references:
                print(f"⚠️⚠️⚠️ [information] global_rank: {global_rank}, local_rank: {local_rank}, unable to interleave any valid reference images, we append them all at once.")
                for key, val in condition_images.items():
                    if "ref" in key:
                        if isinstance(val, list):
                            valid_num = len(val) - padding_info[0].item() if padding_info is not None else len(val)
                            if isinstance(ref_ids, torch.Tensor):
                                ref_ids = ref_ids.view(-1).tolist()
                            img_id = 0
                            for refid, ref_img in zip(ref_ids, val[:valid_num]):
                                obj_type = ""
                                if refid < 100:
                                    obj_type = "PERSON"
                                    pass
                                elif refid >= 100 and refid < 200:
                                    obj_type = "OBJECT"
                                    refid = refid % 100
                                elif refid >= 200:
                                    obj_type = "SCENE"
                                    refid = refid % 200

                                messages.append(
                                    {
                                        "role": "user", 
                                        "content": [
                                            {
                                                "type": "text", 
                                                "text": f"The {obj_type}_{refid} looks like: "
                                            }, 
                                            {
                                                "type": "image", 
                                                "image": self.tensor_to_pil_and_resize(val[img_id], padding_info=padding_info),
                                            }
                                        ]
                                    }
                                )
                                img_id += 1
                        else:
                            messages.append(
                                {
                                    "role": "user", 
                                    "content": [
                                        {
                                            "type": "text", 
                                            "text": "The PERSON_1 looks like: "
                                        }, 
                                        {
                                            "type": "image", 
                                            "image": self.tensor_to_pil_and_resize(val, padding_info=padding_info),
                                        }
                                    ]
                                }
                            )

            # If there's trailing text after the last entity, add it
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": split_interleaves["tail"]["text_after"]
                        }
                    ]
                }
            )
        
        chat_template = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_template], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt", 
        )   # token of image: 151655
        # inputs = inputs.to(self.model.device)
        return inputs

    def forward(self, 
                video_caption, 
                condition_images, 
                ref_ids, 
                return_mask=True,           # not used, just to keep args compatibility
                add_special_tokens=True,    # not used, just to keep args compatibility
                padding_info=None
    ):
        """Tokenize the input video caption and condition images. 
        The function's input args are designed to compatible with T5EncoderModel's tokenizer as much as possible.
        """
        inputs = self.tokenize(video_prompts=video_caption, 
                               condition_images=condition_images, 
                               ref_ids=ref_ids, 
                               padding_info=padding_info)
        return inputs

# ===================================
# MetaQuery，使用指代关系修复的caption，适配MTSS图文混排
# ===================================
class QwenVL2_5_Encoder_v2_mtss(nn.Module):
    def __init__(self, 
                 pretrained_model_name_or_path, 
                 max_edge=384, 
                 max_aspect_ratio=1.75, 
                 text_len=512, 
                 drop_tokens=93, 
                 num_metaqueries=256, 
                 device=None, 
                 dtype=None):
        super().__init__()
        assert device is not None and dtype is not None

        self.max_edge = max_edge
        self.max_aspect_ratio = max_aspect_ratio
        self.text_len = text_len
        self.num_metaqueries = num_metaqueries
        self.device = device
        self.dtype = dtype
        self.drop_tokens = drop_tokens
        
        config = Qwen2_5_VLConfig.from_pretrained(
            pretrained_model_name_or_path
        )
        config.torch_dtype = "float32"
        config.text_config.torch_dtype = "float32"
        config.vision_config.torch_dtype = "float32"
        
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=self.dtype, 
            device_map=self.device
        )

        self.model.language_model.config.use_sliding_window = False
        self.model.language_model.config.sliding_window = None
        self.embedding_dim = self.model.get_input_embeddings().weight.shape[1]

        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path
        )
        
        self.tokenizer = Tokenizer_v4_mtss(
            processor=self.processor, 
            max_edge=max_edge, 
            max_aspect_ratio=max_aspect_ratio, 
        )

        self.query_tokens = nn.Parameter(torch.randn(num_metaqueries, self.embedding_dim), requires_grad=True)

    def reorg_condition_images(self, context):

        condition_images = {}
        if "lf_pixel_value" in context.keys():
            condition_images["lf"] = context["lf_pixel_value"][0,:,0]   # (c h w)
        if "ff_pixel_value" in context.keys():
            condition_images["ff"] = context["ff_pixel_value"][0,:,0]   # (c h w)
        if "ref_pixel_values" in context.keys():
            condition_images["ref"] = [context["ref_pixel_values"][0,:,idx] 
                                       for idx in range(context["ref_pixel_values"].size(2))]
        if "src_pixel_values" in context.keys():
            pass

        return condition_images

    def prepare_embeddings(self, 
                           input_ids : Optional[torch.LongTensor] = None, 
                           pixel_values : Optional[torch.LongTensor] = None, 
                           attention_mask : Optional[torch.LongTensor] = None, 
                           image_grid_thw : Optional[torch.LongTensor] = None, 
                           num_metaqueries: Optional[int] = None, ):
        """
        Args:
            input_ids(Tensor): Shape(b l)
            pixel_values(Tensor): Shape(f c)
            attention_mask(Tensor): Shape(b l)
            image_grid_thw(Tensor): Shape(b 3)
        """
        
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.model.model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            n_image_tokens = (input_ids == self.model.model.config.image_token_id).sum()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens.item() != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens.item()}, features {n_image_features}"
                )

            mask = input_ids == self.model.model.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:
            # FSDP3 fix: even when this rank has no pixel_values, we must still
            # call get_image_features so that the FSDP all-gather on the vision
            # encoder parameters is executed collectively across all ranks.
            # Skipping this branch causes a deadlock because other ranks that DO
            # have pixel_values are blocked waiting for this rank to participate
            # in the all-gather.
            visual = self.model.model.visual
            patch_embed = visual.patch_embed
            # Build a minimal dummy input that matches the expected patch-embed shape
            temporal_patch = getattr(patch_embed, 'temporal_patch_size', 2)
            patch_size = getattr(patch_embed, 'patch_size', 14)
            in_channels = getattr(patch_embed, 'in_channels', 3)
            dummy_pixel = torch.zeros(
                1, in_channels, temporal_patch, patch_size, patch_size,
                device=inputs_embeds.device, dtype=inputs_embeds.dtype,
            )
            dummy_thw = torch.tensor(
                [[temporal_patch, patch_size, patch_size]],
                device=inputs_embeds.device, dtype=torch.long,
            )
            _ = self.model.model.get_image_features(dummy_pixel, dummy_thw)

        pixel_values_videos = None
        video_grid_thw = None
        second_per_grid_ts = None

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        # calculate RoPE index once per generation in the pre-fill stage only
        query_ids = input_ids[:, -1:].repeat(1, num_metaqueries)
        query_mask = attention_mask[:, -1:].repeat(1, num_metaqueries)
        cat_input_ids = torch.cat([input_ids, query_ids], dim=1)
        cat_attention_mask = torch.cat([attention_mask, query_mask], dim=1)

        # input_len = cat_input_ids.size(1)
        # if input_len > 2048:
        #     cat_input_ids = cat_input_ids[:,-2048:]
        #     cat_attention_mask = cat_attention_mask[:,-2048:]
        #     crop_len = input_len - 2048
        #     inputs_embeds = inputs_embeds[:, crop_len:]
        #     attention_mask = attention_mask[:, crop_len:]

        position_ids, rope_deltas = self.model.model.get_rope_index(
            cat_input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            cat_attention_mask,
        )

        return inputs_embeds, attention_mask, position_ids, rope_deltas
        
    def forward(self, 
                video_prompts, 
                context, 
                return_mask=True,           # not used, just to keep args compatibility
                add_special_tokens=True,    # not used, just to keep args compatibility
                padding_info=None,
                ):
        """
        Args:
            video_prompts(String):
            context(Dict): dict of image tensors, the pixel value of each image ranges in [-1, 1]
        """
        
        # Organize context
        condition_images = self.reorg_condition_images(context)

        # Tokenize
        ref_ids = context.get("ref_id", torch.tensor([[1,2]]))   # If not passed, we set ref_id to [1, 2] by default
        inputs = self.tokenizer(video_prompts, condition_images, ref_ids=ref_ids, padding_info=padding_info)
        input_ids = inputs.input_ids.to(self.device)
        
        if hasattr(inputs, "pixel_values") and hasattr(inputs, "image_grid_thw"):
            pixel_values = self.check_shape(
                inputs["pixel_values"].to(self.device), 
                inputs["image_grid_thw"].to(self.device))
            image_grid_thw = inputs.image_grid_thw.to(self.device)
        else:
            pixel_values = None
            image_grid_thw = None

        attention_mask = inputs.attention_mask.to(self.device)

        (
            inputs_embeds, attention_mask, position_ids, rope_deltas
        ) = self.prepare_embeddings(
            input_ids=input_ids, 
            pixel_values=pixel_values, 
            attention_mask=attention_mask, 
            image_grid_thw=image_grid_thw, 
            num_metaqueries=self.num_metaqueries)

        

        # Concat meta_queries
        bsz = inputs_embeds.size(0)
        meta_queries = self.query_tokens.view(-1, self.num_metaqueries, self.embedding_dim).repeat(bsz, 1, 1)
        cat_attn = torch.ones(bsz, self.num_metaqueries).to(attention_mask)
        cat_inputs_embeds = torch.cat([inputs_embeds, meta_queries], dim=1)
        cat_attention_mask = torch.cat([attention_mask, cat_attn], dim=1)

        # Forward pass
        outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=cat_attention_mask,
            past_key_values=None,
            inputs_embeds=cat_inputs_embeds,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            cache_position=None,
        )
        
        # generated_ids = self.model.generate(**inputs.to(device=self.device), max_new_tokens=512)
        # generated_ids_trimmed = [
        #     # out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        #     out_ids[self.drop_tokens:] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        # ]
        # output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print(output_text)
        
        last_hidden_state = outputs.last_hidden_state
        prompt_embeds = last_hidden_state[:, -self.num_metaqueries:]
        attention_mask = cat_attention_mask[:, -self.num_metaqueries:]

        return prompt_embeds, attention_mask

    @torch.no_grad()
    def encode_vision_features(self, context, padding_info=None,):
        # Organize context
        condition_images = self.reorg_condition_images(context)

        # Tokenize
        inputs = self.tokenizer("", condition_images, padding_info=padding_info)
        input_ids = inputs.input_ids.to(self.device)
        pixel_values = self.check_shape(
            inputs["pixel_values"].to(self.device), 
            inputs["image_grid_thw"].to(self.device))
        
        vision_embeds = self.model.model.get_image_features(
            pixel_values, 
            inputs["image_grid_thw"].to(self.device))   # (f 2048)

        return vision_embeds

    def check_shape(self, pixel_values, image_grid_thw):
        total_seq_lens = 0
        for grid_thw in image_grid_thw:
            total_seq_lens += (grid_thw[1].item() * grid_thw[2].item())
        if total_seq_lens != pixel_values.size(0):
            if total_seq_lens > pixel_values.size(0):
                print(f" ----- {total_seq_lens} | {pixel_values.shape} | {image_grid_thw}")
                pad_len = total_seq_lens - pixel_values.size(0)
                pad_pixel_values = torch.zeros_like(pixel_values[:1]).repeat(pad_len, 1)
                pixel_values = torch.cat([pixel_values, pad_pixel_values], dim=0)
            elif total_seq_lens < pixel_values.size(0):
                print(f" ===== {total_seq_lens} | {pixel_values.shape} | {image_grid_thw}")
                pixel_values = pixel_values[:total_seq_lens]
        return pixel_values
