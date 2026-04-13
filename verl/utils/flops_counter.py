
from typing import TYPE_CHECKING, List, Tuple

import torch


if TYPE_CHECKING:
    from transformers.models.llama.configuration_llama import LlamaConfig


def get_device_flops(unit: str = "T") -> float:
    def unit_convert(number: float, level: str):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number

        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1

        return number

    device_name = torch.cuda.get_device_name()
    flops = float("inf")
    if "H100" in device_name or "H800" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12

    flops_unit = unit_convert(flops, unit)
    return flops_unit


class FlopsCounter:
    

    def __init__(self, config: "LlamaConfig"):
        _ESTIMATE_FUNC = {
            "llama": self._estimate_llama_flops,
            "qwen2": self._estimate_llama_flops,
            "qwen2_moe": self._estimate_qwen2_moe_flops,
            "qwen2_vl": self._estimate_llama_flops,
            "qwen2_5_vl": self._estimate_llama_flops,
            "qwen3": self._estimate_llama_flops,
            "qwen3_vl": self._estimate_llama_flops,
            "qwen3_moe": self._estimate_qwen2_moe_flops,
            "qwen3_vl_moe": self._estimate_qwen2_moe_flops,
        }

        if config.model_type not in _ESTIMATE_FUNC:
            print(f"Only support {_ESTIMATE_FUNC.keys()}, but got {config.model_type}. MFU will always be zero.")

        self.config = getattr(config, "text_config", config)
        self._estimate_flops = _ESTIMATE_FUNC.get(config.model_type, self._estimate_unknown_flops)

    def _estimate_unknown_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        return 0

    def _estimate_llama_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        config = self.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        num_attention_heads = config.num_attention_heads
        intermediate_size = config.intermediate_size

        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        dense_N_flops = 6 * dense_N * tokens_sum

        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen

        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen2_moe_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        config = self.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        num_attention_heads = config.num_attention_heads
        moe_intermediate_size = config.moe_intermediate_size
        moe_topk = config.num_experts_per_tok
        num_experts = config.num_experts

        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        moe_mlp_N = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        dense_N_flops = 6 * dense_N * tokens_sum

        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen

        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_flops(self, batch_seqlens: List[int], delta_time: float) -> Tuple[float, float]:
        
        tokens_sum = sum(batch_seqlens)
        estimated_flops = self._estimate_flops(tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
