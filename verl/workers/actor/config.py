

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    freeze_vision_tower: bool = False

    def post_init(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.model_path is not None and os.path.exists(self.model_path):
            self.model_path = os.path.abspath(self.model_path)

        if self.tokenizer_path is not None and os.path.exists(self.tokenizer_path):
            self.tokenizer_path = os.path.abspath(self.tokenizer_path)


@dataclass
class OptimConfig:
    lr: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    strategy: str = "adamw"
    lr_warmup_ratio: float = 0.0
    lr_warmup_steps: Optional[int] = None
    min_lr_ratio: Optional[float] = None
    warmup_style: str = "constant"
    training_steps: int = field(default=-1, init=False)


@dataclass
class FSDPConfig:
    enable_full_shard: bool = True
    enable_cpu_offload: bool = False
    enable_rank0_init: bool = True
    use_orig_params: bool = False
    torch_dtype: Optional[str] = None
    fsdp_size: int = -1
    mp_param_dtype: str = "bf16"
    mp_reduce_dtype: str = "fp32"
    mp_buffer_dtype: str = "fp32"


@dataclass
class OffloadConfig:
    offload_params: bool = False
    offload_optimizer: bool = False


@dataclass
class ActorConfig:
    strategy: str = "fsdp"
    global_batch_size: int = 256
    
    micro_batch_size_per_device_for_update: int = 4
    
    micro_batch_size_per_device_for_experience: int = 16
    
    max_grad_norm: float = 1.0
    
    clip_ratio_low: float = 0.2
    
    clip_ratio_high: float = 0.3
    
    clip_ratio_dual: float = 3.0
    
    loss_avg_mode: str = "token"
    
    ppo_epochs: int = 1
    
    padding_free: bool = True
    
    dynamic_batching: bool = True
    
    ulysses_size: int = 1
    
    use_torch_compile: bool = True
    
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    global_batch_size_per_device: int = field(default=-1, init=False)
    disable_kl: bool = field(default=False, init=False)
    use_kl_loss: bool = field(default=False, init=False)
    kl_penalty: str = field(default="kl", init=False)
    kl_coef: float = field(default=0.0, init=False)
    
    compute_vision_kl: bool = field(default=False, init=False)

    use_on_perception: bool = field(default=False, init=False)
    top_p_perception_tokens: float = field(default=0.2, init=False)
    use_on_entropy: bool = field(default=False, init=False)
    top_p_entropy_tokens: float = field(default=0.2, init=False)
    use_advantage_shaping: bool = field(default=False, init=False)
    advantage_scaling_min: float = field(default=0.8, init=False)
    use_entropy_penalty: bool = field(default=False, init=False)
    entropy_penalty_coef: float = field(default=0.06, init=False)

    use_token_shaping: bool = field(default=False, init=False)
    token_shaping_weight_min: float = field(default=0.9, init=False)
    token_shaping_weight_max: float = field(default=1.1, init=False)


@dataclass
class RefConfig:
    strategy: str = "fsdp"
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    micro_batch_size_per_device_for_experience: int = field(default=-1, init=False)
    padding_free: bool = field(default=False, init=False)
    dynamic_batching: bool = field(default=False, init=False)
    ulysses_size: int = field(default=1, init=False)
    use_torch_compile: bool = field(default=True, init=False)
