from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist

import megatron
from megatron import get_timers, get_args, core, get_num_microbatches
from megatron.core import mpu, tensor_parallel
from megatron.model.enums import AttnMaskType, ModelType, LayerType, AttnType
from megatron.model.transformer import _args_to_kwargs
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.transformer import ParallelAttention, CoreAttention, ParallelMLP
from .adaptor_core_tensor_parallel import ColumnParallelLinearMoE, RowParallelLinearMoE
from megatron.model.module import MegatronModule
import torch_npu

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def ParallelMLPInit(self, init_method, output_layer_init_method):
    super(ParallelMLP, self).__init__()
    args = get_args()
    self.is_x_model = args.is_x_model
    self.moe = args.moe

    if self.moe:
        self.dense_h_to_4h = ColumnParallelLinearMoE(
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
            moe=args.moe,
            enable_expert_tensor_parallel=args.enable_expert_tensor_parallel,
            **_args_to_kwargs())
        for name, param in self.dense_h_to_4h.named_parameters():
            param.moe = True
    else:
        # Project to 4h.
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
            **_args_to_kwargs())

    self.bias_gelu_fusion = args.bias_gelu_fusion
    self.activation_func = F.gelu
    if args.openai_gelu:
        self.activation_func = openai_gelu
    elif args.onnx_safe:
        self.activation_func = erf_gelu

    # Project back to h.
    if self.is_x_model:
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            args.ffn_hidden_size // 2,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            **_args_to_kwargs())
    elif self.moe:
        self.dense_4h_to_h = RowParallelLinearMoE(
            args.ffn_hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            moe=args.moe,
            enable_expert_tensor_parallel=args.enable_expert_tensor_parallel,
            **_args_to_kwargs())
        for name, param in self.dense_4h_to_h.named_parameters():
            param.moe = True
    else:
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            **_args_to_kwargs())


def ParallelMLPForward(self, hidden_states):
    # [s, b, 4hp]
    intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

    if self.is_x_model:
        x = intermediate_parallel + bias_parallel
        x, gates = x.chunk(2, dim=-1)
        intermediate_parallel = x * F.gelu(gates)
    else:
        intermediate_parallel = F.gelu(bias_parallel + intermediate_parallel)

    # [s, b, h]
    output, output_bias = self.dense_4h_to_h(intermediate_parallel)
    return output, output_bias


def CoreAttentionForward(self, query_layer, key_layer,
                         value_layer, attention_mask):
    # ===================================
    # Raw attention scores. [b, np, s, s]
    # ===================================

    # [b, np, sq, sk]
    output_size = (query_layer.size(1),
                   query_layer.size(2),
                   query_layer.size(0),
                   key_layer.size(0))

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query_layer = query_layer.view(output_size[2],
                                   output_size[0] * output_size[1], -1)
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key_layer = key_layer.view(output_size[3],
                               output_size[0] * output_size[1], -1)

    # preallocting input tensor: [b * np, sq, sk]
    matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
        (output_size[0] * output_size[1], output_size[2], output_size[3]),
        query_layer.dtype, "mpu")

    # Raw attention scores. [b * np, sq, sk]
    # matmul_result = torch.baddbmm(
    #    matmul_input_buffer,
    #    query_layer.transpose(0, 1),   # [b * np, sq, hn]
    #    key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
    #    beta=0.0, alpha=(1.0/self.norm_factor))
    matmul_result = torch.bmm(query_layer.transpose(0, 1), key_layer.permute(1, 2, 0))
    matmul_result *= 1.0 / self.norm_factor

    # change view to [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    # ===========================
    # Attention probs and dropout
    # ===========================

    # attention scores and attention mask [b, np, sq, sk]
    attention_probs = self.scale_mask_softmax(attention_scores,
                                              attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.

    if not self.sequence_parallel:
        with tensor_parallel.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)
    else:
        attention_probs = self.attention_dropout(attention_probs)

    # =========================
    # Context layer. [sq, b, hp]
    # =========================

    # value_layer -> context layer.
    # [sk, b, np, hn] --> [b, np, sq, hn]

    # context layer shape: [b, np, sq, hn]
    output_size = (value_layer.size(1),
                   value_layer.size(2),
                   query_layer.size(0),
                   value_layer.size(3))

    # change view [sk, b * np, hn]
    value_layer = value_layer.view(value_layer.size(0),
                                   output_size[0] * output_size[1], -1)

    # change view [b * np, sq, sk]
    attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                           output_size[2], -1)

    # matmul: [b * np, sq, hn]
    context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

    # change view [b, np, sq, hn]
    context_layer = context_layer.view(*output_size)

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

    # [sq, b, np, hn] --> [sq, b, hp]
    new_context_layer_shape = context_layer.size()[:-2] + \
                              (self.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)

    return context_layer


class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None, pre_tockens=None, next_tockens=None, shape_order='SBH'):
        super().__init__()
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        args = get_args()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.pre_tockens = pre_tockens
        self.next_tockens = next_tockens
        self.shape_order = shape_order
        self.group_query_attention = args.group_query_attention

    def forward(self, q, k, v, n, attention_mask):

        scale = 1.0 / math.sqrt(q.size(2) / n) if self.softmax_scale is None else self.softmax_scale
        seq_length = q.size(0)
        if self.shape_order == 'BSH':
            seq_length = q.size(1)

        if self.group_query_attention:
            if not hasattr(self, 'attention_mask'):
                self.attention_mask = torch.triu(torch.ones([2048, 2048], dtype=torch.bool), diagonal=1).npu()
            output = torch_npu.npu_fusion_attention( \
                q, k, v, n, self.shape_order, \
                pse=None, \
                padding_mask=None, \
                atten_mask=self.attention_mask, \
                scale=scale, \
                pre_tockens=self.pre_tockens, \
                next_tockens=self.next_tockens, \
                keep_prob=1 - self.dropout_p, \
                inner_precise=0,
                sparse_mode=4
                )[0]
        else:
            if not hasattr(self, 'attention_mask'):
                self.attention_mask = (torch.tril(torch.ones([seq_length, seq_length]), diagonal=-(self.pre_tockens + 1)) + torch.triu(torch.ones([seq_length, seq_length]), diagonal=self.next_tockens + 1)).bool().npu()
            output = torch_npu.npu_fusion_attention( \
                q, k, v, n, self.shape_order, \
                pse=None, \
                padding_mask=None, \
                atten_mask=self.attention_mask, \
                scale=scale, \
                pre_tockens=self.pre_tockens, \
                next_tockens=self.next_tockens, \
                keep_prob=1 - self.dropout_p, \
                inner_precise=0
                )[0]

        return output


def ParallelAttentionInit(self, init_method,
                          output_layer_init_method, layer_number,
                          attention_type=AttnType.self_attn,
                          attn_mask_type=AttnMaskType.padding):
    super(ParallelAttention, self).__init__()
    args = get_args()
    self.layer_number = max(1, layer_number)
    self.attention_type = attention_type
    self.attn_mask_type = attn_mask_type
    self.params_dtype = args.params_dtype
    self.sequence_parallel = args.sequence_parallel
    self.shape_order = args.shape_order

    self.group_query_attention = args.group_query_attention
    self.num_query_groups = args.num_query_groups

    query_projection_size = args.kv_channels * args.num_attention_heads
    if self.group_query_attention:
        kv_projection_size = args.kv_channels * args.num_query_groups
    else:
        kv_projection_size = args.kv_channels * args.num_attention_heads

    self.use_flash_attn = args.use_flash_attn
    if self.use_flash_attn:

        assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
                                                      'self-attention for now')
        assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
                                                            'supports causal mask for now')
        if rearrange is None:
            raise ImportError('einops is not installed, please install with pip install einops')

    # Per attention head and per partition values.
    world_size = mpu.get_tensor_model_parallel_world_size()
    self.hidden_size_per_attention_head = core.utils.divide(
        query_projection_size, args.num_attention_heads)
    self.num_attention_heads_per_partition = core.utils.divide(
        args.num_attention_heads, world_size)

    if self.group_query_attention:
        if args.num_query_groups % world_size != 0:
            raise NotImplementedError('Currently the num_query_groups should be '
                                      'a multiple of the tensor parallel size')
        self.num_query_groups_per_partition = core.utils.divide(
            args.num_query_groups, world_size)
    else:
        self.num_query_groups_per_partition = self.num_attention_heads_per_partition

    # Strided linear layer.
    if attention_type == AttnType.self_attn:
        self.query_key_value = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            query_projection_size + 2 * kv_projection_size,
            gather_output=False,
            init_method=init_method,
            async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
            **_args_to_kwargs())
    else:
        assert attention_type == AttnType.cross_attn

        if self.group_query_attention:
            raise NotImplementedError("Grouped query attention not implemented for cross-attention.")
        assert query_projection_size == kv_projection_size

        self.query = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            query_projection_size,
            gather_output=False,
            init_method=init_method,
            async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
            **_args_to_kwargs())

        self.key_value = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            2 * kv_projection_size,
            gather_output=False,
            init_method=init_method,
            async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
            **_args_to_kwargs())

    self.core_attention = CoreAttention(self.layer_number,
                                        self.attn_mask_type)
    self.checkpoint_core_attention = args.recompute_granularity == 'selective'

    if self.use_flash_attn:
        self.core_attention_flash = FlashSelfAttention(
            causal=True, attention_dropout=args.attention_dropout,
            pre_tockens=args.pre_tockens, next_tockens=args.next_tockens,
            shape_order=args.shape_order
        )

    # Output.
    self.dense = tensor_parallel.RowParallelLinear(
        query_projection_size,
        args.hidden_size,
        input_is_parallel=True,
        init_method=output_layer_init_method,
        skip_bias_add=True,
        **_args_to_kwargs())


def ParallelAttentionForward(self, hidden_states, attention_mask,
                             encoder_output=None, inference_params=None):
    # hidden_states: [sq, b, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    if inference_params:
        if self.layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_len = inference_params.max_sequence_len
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size)
            inference_value_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size)
            inference_params.key_value_memory_dict[self.layer_number] = (
                inference_key_memory, inference_value_memory)
        else:
            inference_key_memory, inference_value_memory = \
                inference_params.key_value_memory_dict[self.layer_number]

    # =====================
    # Query, Key, and Value
    # =====================

    if self.attention_type == AttnType.self_attn:
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                    * self.hidden_size_per_attention_head
            ),
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query_layer,
         key_layer,
         value_layer) = torch.split(
            mixed_x_layer,
            [
                (
                        self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                        * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head
            ],
            dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query_layer = query_layer.reshape(query_layer.size(0), query_layer.size(1), -1,
                                          self.hidden_size_per_attention_head)
    else:
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv_layer, _ = self.key_value(encoder_output)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            2 * self.hidden_size_per_attention_head)
        mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key_layer,
         value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query_layer, _ = self.query(hidden_states)
        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query_layer.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        query_layer = query_layer.view(*new_tensor_shape)

    # ==================================
    # Adjust key and value for inference
    # ==================================

    if inference_params:
        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key_layer.size(1)
        assert batch_end <= inference_key_memory.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key_layer.size(0)
        assert sequence_end <= inference_key_memory.size(0)
        # Copy key and values.
        inference_key_memory[sequence_start:sequence_end,
        batch_start:batch_end, ...] = key_layer
        inference_value_memory[sequence_start:sequence_end,
        batch_start:batch_end, ...] = value_layer
        key_layer = inference_key_memory[
                    :sequence_end, batch_start:batch_end, ...]
        value_layer = inference_value_memory[
                      :sequence_end, batch_start:batch_end, ...]

    # ==================================
    # core attention computation
    # ==================================

    if not self.use_flash_attn:
        if self.checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask)
    else:

        hidden_head_num = query_layer.size(2)
        if self.shape_order == 'BSH':
            q, k, v = [rearrange(x, 's b h d -> b s (h d)').contiguous()
                       for x in (query_layer, key_layer, value_layer)]
        elif self.shape_order == 'SBH':
            q, k, v = [rearrange(x, 's b h d -> s b (h d)').contiguous()
                       for x in (query_layer, key_layer, value_layer)]
        else:
            raise ImportError('flash attention shape order must be SBH or BSH, please add args shape-order')

        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context_layer = self.core_attention_flash(q, k, v, hidden_head_num, attention_mask)
        else:
            context_layer = self.core_attention_flash(q, k, v, hidden_head_num, attention_mask)

        if self.shape_order == 'BSH':
            context_layer = torch.tensor(1.0).to(context_layer.dtype).npu() * context_layer
            context_layer = rearrange(context_layer, 'b s D -> s b D').contiguous()

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.dense(context_layer)

    return output, bias

exp_selection_uniform_map: Dict[torch.device, Callable] = {}
uniform_map: Dict[torch.device, Callable] = {}


class SwitchMLP(MegatronModule):
    """
    Top-1 Mixture of Experts Layer.
    Routes input to one of N MLP "experts"
    Currently supports Sinkhorn based expert routing.
    """
    _initialized = False

    def __init__(self,
                 init_method,
                 output_layer_init_method,
                 expert_parallel=True,
                 sequence_parallel=True,
                 add_bias_linear=True,
                 enable_expert_tensor_parallel=False):
        super().__init__()
        args = get_args()
        if args.bf16:
            dtype = torch.bfloat16
        elif args.fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        self.dtype = dtype
        self.router = torch.nn.Linear(args.hidden_size, args.num_experts, dtype=dtype)
        self.add_bias = add_bias_linear
        self.expert_parallel = expert_parallel
        self.sequence_parallel = sequence_parallel
        self.router_activation = torch.sigmoid
        self.num_experts = args.num_experts
        self.tp_size = args.tensor_model_parallel_size

        self.ep_size = args.expert_parallel_size
        self.capacity_factor = args.capacity_factor
        self.min_capacity = args.min_capacity

        self.noisy_gate_policy = args.noisy_gate_policy
        self.use_residule_for_exceed = args.use_residule_for_exceed
        self.enable_expert_tensor_parallel = enable_expert_tensor_parallel

        self.moe = args.moe
        if self.moe:
            if not SwitchMLP._initialized:
                from .adaptor_parallel_state import initialize_model_parallel_moe
                args = get_args()
                initialize_model_parallel_moe(
                    args.tensor_model_parallel_size,
                    args.pipeline_model_parallel_size,
                    args.expert_parallel_size,
                    args.enable_expert_tensor_parallel
                )
                SwitchMLP._initialized = True
            from .adaptor_parallel_state import get_expert_parallel_group
            self.ep_group = get_expert_parallel_group()

            self.num_local_experts = (
                self.num_experts // args.expert_parallel_size
            )
        else:
            self.num_local_experts = self.num_experts

        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.num_local_experts):
            expert = ParallelMLP(init_method, output_layer_init_method)
            self.local_experts.append(expert)

    def forward(self, hidden_states):
        if not self.moe:
            # hidden_states: [s, b, h]
            s = hidden_states.size(0)
            b = hidden_states.size(1)
            h = hidden_states.size(2)
            route = self.router(hidden_states)
            route = torch.nn.functional.softmax(route, dim=2)
            max_prob, max_ind = torch.max(route, dim=2)
            max_prob = torch.unsqueeze(max_prob, 2)  # [s b 1]

            # TODO (rprenger) TODO this could be made easier to read
            # Converting [s, b, h] to [s*b, h].
            # Each vector could be routed differently
            hidden_states = hidden_states.view(-1, hidden_states.size(2))  # [s*b h]
            max_prob = max_prob.view(-1, max_prob.size(2))  # [s*b 1]
            max_ind = max_ind.view(-1)  # [s*b]

            output_total = torch.empty_like(hidden_states)
            output_bias_total = torch.empty_like(hidden_states)
            # TODO (rprenger) This does each expert in serial, but it could be parallelized

            for expert_num, expert in enumerate(self.experts):
                local_indices = (max_ind == expert_num).nonzero()
                hidden = hidden_states[local_indices, :]
                output, output_bias = expert(hidden)
                output_bias = output_bias.expand_as(output)
                output_total[local_indices, :] = output
                output_bias_total[local_indices, :] = output_bias

            output_total = output_total * max_prob
            output_bias_total = output_bias_total * max_prob
            output_total = output_total.view(s, b, h)
            output_bias_total = output_bias_total.view(s, b, h)

            return output_total, output_bias_total

        hidden_shape = hidden_states.shape
        reshaped_hidden_states = hidden_states.reshape(-1, hidden_shape[-1])
        if self.noisy_gate_policy == 'Jitter':
            reshaped_hidden_states = multiplicative_jitter(
                reshaped_hidden_states, device=reshaped_hidden_states.device)


        route = self.router(reshaped_hidden_states.to(dtype=self.dtype))  # (num_tokens, num_experts)
        route = route.view(-1, self.num_experts)  # (num_tokens, num_experts)

        l_aux, combine_weights, dispatch_mask, exp_counts, exceed_mask = top1gating(
            route, self.enable_expert_tensor_parallel, self.min_capacity, self.tp_size, self.capacity_factor
        )
        dispatch_input = torch.einsum('sec,sm->ecm', dispatch_mask.type_as(route[1]), reshaped_hidden_states)
        capacity = dispatch_mask.shape[-1]
        self.exp_counts_exceed = torch.clamp(exp_counts - capacity, min=0).float().sum() \
                                 / reshaped_hidden_states.shape[0]
        self.exp_counts_padding = torch.clamp(capacity - exp_counts, min=0).float().sum() \
                                  / reshaped_hidden_states.shape[0]

        dispatch_input = dispatch_input.reshape(self.ep_size, -1, dispatch_input.shape[-1])
        tmp_dispatched = dispatch_input

        dispatch_input = _AlltoAll.apply(self.ep_group, tmp_dispatched)

        dispatch_input = dispatch_input.reshape(self.ep_size, self.num_local_experts, -1, hidden_shape[-1])

        chunks = dispatch_input.chunk(self.num_local_experts, dim=1)
        expert_output = []
        for chunk, expert in zip(chunks, self.local_experts):
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]
            expert_output += [out]

        expert_output = torch.cat(expert_output, dim=1)
        tmp_expert_output = expert_output.view(self.ep_size, -1, expert_output.shape[-1]).contiguous()

        expert_output = _AlltoAll.apply(self.ep_group, tmp_expert_output)

        combine_output = torch.einsum("sec,ecm->sm", combine_weights.type_as(hidden_states), expert_output)

        if self.use_residule_for_exceed:
            exceed_mask = exceed_mask.unsqueeze(1)
            combine_output = exceed_mask * reshaped_hidden_states + (1 - exceed_mask) * combine_output

        a = combine_output.reshape(hidden_shape)
        output_bias_total = torch.zeros_like(a)

        return a, output_bias_total


@torch.jit.script
def _capacity(
        num_tokens,
        num_experts,
        capacity_factor,
        min_capacity,
        align_tp_num,):
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if capacity < min_capacity:
        capacity = min_capacity
    capacity = math.ceil(capacity / align_tp_num) * align_tp_num
    return capacity


capacity_dict: Dict[Tuple, Tuple] = {}


def get_capacity(
        gates,
        capacity_factor,
        min_capacity,
        align_tp_num,):
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    key = (num_tokens, num_experts, min_capacity, align_tp_num)
    if key not in capacity_dict:
        capacity = _capacity(num_tokens, num_experts, capacity_factor, min_capacity, align_tp_num)
        capacity_dict[key] = (torch.tensor(capacity, device=gates.device, dtype=torch.int32), capacity)
    return capacity_dict[key]


def top1gating(
        logits,
        ep_size,
        min_capacity,
        tp_size,
        capacity_factor,
        tp_with_ep=False,
        use_rts=False):
    gates = F.softmax(logits, dim=1)

    if not tp_with_ep and tp_size > 1:
        capacity, capacity_host = get_capacity(gates, capacity_factor, min_capacity, align_tp_num=tp_size)
    else:
        capacity, capacity_host = get_capacity(gates, capacity_factor, min_capacity, align_tp_num=1)

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts).float()

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach()

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=device),
                                                          high=torch.tensor(1.0, device=device)).rsample
            exp_selection_uniform_map[logits.device] = uniform
        mask_rand = mask1 * uniform(mask1.shape).to(mask1.dtype)
    else:
        mask_rand = mask1.float()

    assert logits.shape[0] >= min_capacity, \
        ("No. of tokens (batch-size) should be greater than min_capacity. "
         "Either set min_capacity to 0 or increase your batch size.")

    top_idx = _top_idx(mask_rand, capacity_host)
    ones = torch.ones_like(top_idx, dtype=mask1.dtype)
    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, ones)

    exceed_mask = torch.abs(mask1 - new_mask1).sum(dim=1)

    locations1 = torch.cumsum(new_mask1, dim=0) - 1

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * new_mask1, dim=1)

    # Normalize gate probabilities
    new_mask1_float = new_mask1.float()
    gates_drop = gates * new_mask1_float

    locations1_sc = _one_hot_to_float(locations1_s.int(), capacity_host)
    combine_weights = torch.einsum('se,sc->sec', gates_drop, locations1_sc)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts.int(), exceed_mask.int()


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


class _AlltoAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, group, input):
        ctx.group = group
        input = input.contiguous().npu_format_cast(2)
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, _AlltoAll.apply(ctx.group, *grad_outputs)


def multiplicative_jitter(x, device, epsilon=1e-2):
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(low=torch.tensor(1.0 - epsilon, device=device),
                                                      high=torch.tensor(1.0 + epsilon, device=device)).rsample
        uniform_map[device] = uniform
    output = x * uniform(x.shape).to(x.dtype)
    return output


megatron.model.transformer.ParallelMLP.__init__ = ParallelMLPInit
megatron.model.transformer.ParallelMLP.forward = ParallelMLPForward
megatron.model.transformer.CoreAttention.forward = CoreAttentionForward
megatron.model.transformer.FlashSelfAttention = FlashSelfAttention
megatron.model.transformer.ParallelAttention.__init__ = ParallelAttentionInit
megatron.model.transformer.ParallelAttention.forward = ParallelAttentionForward
megatron.model.transformer.SwitchMLP = SwitchMLP

