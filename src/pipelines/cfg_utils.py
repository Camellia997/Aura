from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CFGCtrlParams:
    smc_cfg_enable: bool = False
    smc_cfg_lambda: float = 0.05
    smc_cfg_K: float = 0.3
    no_cfg_warmup_steps: int = 0

    @classmethod
    def build(
        cls,
        *,
        smc_cfg_enable: bool = False,
        smc_cfg_lambda: float = 0.05,
        smc_cfg_k: Optional[float] = None,
        smc_cfg_K: Optional[float] = None,
        no_cfg_warmup_steps: int = 0,
    ) -> "CFGCtrlParams":
        if smc_cfg_K is None:
            smc_cfg_K = 0.3 if smc_cfg_k is None else smc_cfg_k
        return cls(
            smc_cfg_enable=bool(smc_cfg_enable),
            smc_cfg_lambda=float(smc_cfg_lambda),
            smc_cfg_K=float(smc_cfg_K),
            no_cfg_warmup_steps=int(no_cfg_warmup_steps),
        )

    @property
    def enabled(self) -> bool:
        return (
            self.smc_cfg_enable
            or self.no_cfg_warmup_steps > 0
        )


@dataclass
class CFGCtrlState:
    prev_guidance_eps: Optional[torch.Tensor] = None


class CFGCtrlMixin:
    def cfg_ctrl_apply(
        self,
        *,
        noise_pred_posi: torch.Tensor,
        noise_pred_nega: torch.Tensor,
        cfg_scale: float,
        progress_id: int,
        params: CFGCtrlParams,
        state: CFGCtrlState,
    ) -> torch.Tensor:
        warmup_no_cfg = params.no_cfg_warmup_steps > 0 and progress_id < params.no_cfg_warmup_steps
        guidance_eps = noise_pred_posi - noise_pred_nega

        if params.smc_cfg_enable and not warmup_no_cfg:
            if state.prev_guidance_eps is None:
                state.prev_guidance_eps = guidance_eps.detach()
            s = (guidance_eps - state.prev_guidance_eps) + params.smc_cfg_lambda * state.prev_guidance_eps
            u_sw = -params.smc_cfg_K * torch.sign(s)
            guidance_eps = guidance_eps + u_sw
            state.prev_guidance_eps = guidance_eps.detach()
            return noise_pred_nega + cfg_scale * guidance_eps

        if warmup_no_cfg:
            # 与 CFG-Ctrl-ToComplete 一致：warmup 阶段直接用 conditional 预测。
            return noise_pred_posi

        return noise_pred_nega + cfg_scale * guidance_eps

    def cfg_ctrl_streams_apply(
        self, 
        *, 
        noise_pred_it, 
        noise_pred_i, 
        noise_pred_neg, 
        cfg_scale_img, 
        cfg_scale_txt, 
        progress_id, 
        params_i: CFGCtrlParams, 
        params_t: CFGCtrlParams, 
        state_i: CFGCtrlState,
        state_t: CFGCtrlState,
    ) -> torch.Tensor:
        warmup_no_cfg = params_i.no_cfg_warmup_steps > 0 and progress_id < params_i.no_cfg_warmup_steps
        guidance_eps_i = noise_pred_i - noise_pred_neg
        guidance_eps_t = noise_pred_it - noise_pred_i

        if params_i.smc_cfg_enable and not warmup_no_cfg:
            if state_i.prev_guidance_eps is None:
                state_i.prev_guidance_eps = guidance_eps_i.detach()
            if state_t.prev_guidance_eps is None:
                state_t.prev_guidance_eps = guidance_eps_t.detach()

            s_i = (guidance_eps_i - state_i.prev_guidance_eps) + params_i.smc_cfg_lambda * state_i.prev_guidance_eps
            s_t = (guidance_eps_t - state_t.prev_guidance_eps) + params_t.smc_cfg_lambda * state_t.prev_guidance_eps

            u_sw_i = -params_i.smc_cfg_K * torch.sign(s_i)
            u_sw_t = -params_t.smc_cfg_K * torch.sign(s_t)

            guidance_eps_i = guidance_eps_i + u_sw_i
            guidance_eps_t = guidance_eps_t + u_sw_t

            state_i.prev_guidance_eps = guidance_eps_i.detach()
            state_t.prev_guidance_eps = guidance_eps_t.detach()
            return noise_pred_neg + cfg_scale_img * guidance_eps_i + cfg_scale_txt * guidance_eps_t

        if warmup_no_cfg:
            # 与 CFG-Ctrl-ToComplete 一致：warmup 阶段直接用 conditional 预测。
            return noise_pred_it

        return noise_pred_neg + cfg_scale_i * guidance_eps_i

