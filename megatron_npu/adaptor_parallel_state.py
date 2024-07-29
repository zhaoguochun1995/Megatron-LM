import torch


_EXPERT_DATA_PARALLEL_GROUP = None
_EXPERT_PARALLEL_GROUP = None


def initialize_model_parallel_moe(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        expert_parallel_size: int = 8,
        enable_expert_tensor_parallel=False
) -> None:
    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    data_parallel_size: int = world_size // (tensor_model_parallel_size * 
                                             pipeline_model_parallel_size)

    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    num_data_parallel_groups: int = world_size // data_parallel_size
    num_expert_parallel_groups: int = world_size // expert_parallel_size

    global _EXPERT_DATA_PARALLEL_GROUP
    global _EXPERT_PARALLEL_GROUP
    if enable_expert_tensor_parallel:
        data_parallel_groups = []
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(tensor_model_parallel_size):
                ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
                data_parallel_groups.append(list(ranks))

        expert_parallel_group_ranks = []
        expert_data_parallel_group_ranks = []
        for dp_ranks in data_parallel_groups:
            part_ep_groups = []
            for i in range(0, data_parallel_size, expert_parallel_size):
                ranks = dp_ranks[i:i + expert_parallel_size]
                part_ep_groups.append(ranks)
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    _EXPERT_PARALLEL_GROUP = group
            expert_parallel_group_ranks.extend(part_ep_groups)

            for expert_dp_ranks in zip(*part_ep_groups):
                expert_data_parallel_group_ranks.append(list(expert_dp_ranks))
                group = torch.distributed.new_group(expert_dp_ranks)
                if rank in expert_dp_ranks:
                    _EXPERT_DATA_PARALLEL_GROUP = group
    else:
        all_expert_data_parallel_group_ranks = []
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(expert_parallel_size):
                ranks = range(start_rank + j, end_rank, expert_parallel_size)
                all_expert_data_parallel_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    _EXPERT_DATA_PARALLEL_GROUP = group
    
        for i in range(num_expert_parallel_groups):
            ranks = range(i * expert_parallel_size, 
                        (i + 1) * expert_parallel_size)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _EXPERT_PARALLEL_GROUP = group
    


def get_expert_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _EXPERT_DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _EXPERT_DATA_PARALLEL_GROUP


def get_expert_parallel_group():
    """Get the expert parallel group the caller rank belongs to."""
    assert _EXPERT_PARALLEL_GROUP is not None, \
        'expert parallel group is not initialized'
    return _EXPERT_PARALLEL_GROUP


def get_expert_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_expert_data_parallel_group())


def get_expert_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_expert_data_parallel_group())