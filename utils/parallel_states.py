import os

import torch.distributed as dist


class COMM_INFO:

    def __init__(self):
        self.group = None
        self.sp_size = 1
        self.global_rank = 0
        self.rank_within_group = 0
        self.first_rank_within_group = 0 # first_rank in sp_group
        self.group_id = 0

        self.dp_group = None
        self.rank_in_dp_group = 0
        self.dp_group_id = 0
        self.dp_size = 0


nccl_info = COMM_INFO()
_SEQUENCE_PARALLEL_STATE = False


def initialize_sequence_parallel_state(sequence_parallel_size):
    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        initialize_sequence_parallel_group(sequence_parallel_size)
    else:
        nccl_info.sp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_group = 0
        nccl_info.group_id = int(os.getenv("RANK", "0"))
        nccl_info.rank_in_dp_group = nccl_info.global_rank
        nccl_info.dp_size = int(os.getenv("WORLD_SIZE", "1")) # dp_size=world_size


def initialize_sequence_parallel_state_zero3(sequence_parallel_size):
    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        initialize_sequence_parallel_group(sequence_parallel_size)
    else:
        # nccl_info.global_rank = int(os.getenv("RANK", "0"))
        # nccl_info.rank_within_group = 0
        # nccl_info.group_id = int(os.getenv("RANK", "0"))
        # nccl_info.rank_in_dp_group = nccl_info.global_rank
        nccl_info.sp_size = 1
        nccl_info.dp_size = dist.get_world_size()
        nccl_info.sp_group = dist.group.WORLD 
        nccl_info.dp_group = dist.group.WORLD

def set_sequence_parallel_state(state):
    global _SEQUENCE_PARALLEL_STATE
    _SEQUENCE_PARALLEL_STATE = state


def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE


def initialize_sequence_parallel_group(sequence_parallel_size):
    """Initialize the sequence parallel group."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    assert (
        world_size % sequence_parallel_size == 0
    ), "world_size must be divisible by sequence_parallel_size, but got world_size: {}, sequence_parallel_size: {}".format(
        world_size, sequence_parallel_size
    )
    nccl_info.sp_size = sequence_parallel_size
    nccl_info.global_rank = rank
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    nccl_info.dp_size = num_sequence_parallel_groups
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            nccl_info.group = group
            nccl_info.rank_within_group = rank - i * sequence_parallel_size
            nccl_info.group_id = i
            nccl_info.first_rank_within_group = ranks[0]
    
    dp_size = num_sequence_parallel_groups
    dp_groups = sequence_parallel_size
    # print('dp_size', dp_size, 'dp_groups', dp_groups)
    for i in range(dp_groups):
        dp_ranks = []
        for j in range(dp_size):
            dp_ranks.append(i+j*sequence_parallel_size)
        dp_group = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            #print(rank, i , dp_group)
            nccl_info.dp_group = dp_group
            #nccl_info.rank_in_dp_group = (rank-i)//2
            nccl_info.rank_in_dp_group = (rank - i) // dp_groups
            nccl_info.dp_group_id = i



def destroy_sequence_parallel_group():
    """Destroy the sequence parallel group."""
    dist.destroy_process_group()

def get_sequence_parallel_world_size():
    return nccl_info.sp_size

def get_sequence_parallel_rank():
    return nccl_info.rank_within_group

def get_sp_group():
    return nccl_info.group

def get_dp_group():
    return nccl_info.dp_group
