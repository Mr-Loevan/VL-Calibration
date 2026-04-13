

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple, List

from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class ValidationDatasetConfig:
    
    files: str = ""
    reward_function: Optional[str] = None
    name: Optional[str] = None

@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    val_files_multi: Optional[str] = None
    val_reward_functions: Optional[str] = None
    val_datasets: Optional[List[ValidationDatasetConfig]] = None
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        if self.image_dir is not None:
            if os.path.exists(self.image_dir):
                self.image_dir = os.path.abspath(self.image_dir)
            else:
                print(f"Image directory {self.image_dir} not found.")
                self.image_dir = None

        if self.format_prompt is not None:
            if os.path.exists(self.format_prompt):
                self.format_prompt = os.path.abspath(self.format_prompt)
            else:
                print(f"Format prompt file {self.format_prompt} not found.")
                self.format_prompt = None


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    
    lam: float = 1.0
    
    adv_estimator: str = "grpo"
    
    disable_kl: bool = False
    
    use_kl_loss: bool = False
    
    kl_penalty: str = "kl"
    
    kl_coef: float = 1e-3
    
    kl_type: str = "fixed"
    
    kl_horizon: float = 10000.0
    
    kl_target: float = 0.1
    
    online_filtering: bool = False
    
    filter_key: str = "overall"
    
    filter_low: float = 0.01
    
    filter_high: float = 0.99
    
    
    compute_vision_kl: bool = False
    

    use_on_perception: bool = False
    
    top_p_perception_tokens: float = 0.2
    

    use_on_entropy: bool = False
    
    top_p_entropy_tokens: float = 0.2
    

    use_advantage_shaping: bool = False
    
    advantage_scaling_min: float = 0.8
    

    use_entropy_penalty: bool = False
    
    entropy_penalty_coef: float = 0.06
    

    use_token_shaping: bool = False
    
    token_shaping_weight_min: float = 0.9
    
    token_shaping_weight_max: float = 1.1
    


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    
    max_steps: Optional[int] = None
    
    project_name: str = "easy_r1"
    
    experiment_name: str = "demo"
    
    logger: Tuple[str] = ("console", "wandb")
    
    nnodes: int = 1
    
    n_gpus_per_node: int = 8
    
    max_try_make_batch: int = 20
    
    critic_warmup: int = 0
    
    val_freq: int = -1
    
    val_before_train: bool = True
    
    val_only: bool = False
    
    val_generations_to_log: int = 0
    
    save_freq: int = -1
    
    save_limit: int = -1
    
    save_model_only: bool = False
    
    save_checkpoint_path: Optional[str] = None
    
    load_checkpoint_path: Optional[str] = None
    
    ray_timeline: Optional[str] = None
    
    find_last_checkpoint: bool = True
    

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)
        if self.load_checkpoint_path is not None:
            if os.path.exists(self.load_checkpoint_path):
                self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)
            else:
                print(f"Model checkpoint {self.load_checkpoint_path} not found.")
                self.load_checkpoint_path = None


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        
        self.worker.actor.compute_vision_kl = self.algorithm.compute_vision_kl

        self.worker.actor.use_on_perception = self.algorithm.use_on_perception
        self.worker.actor.top_p_perception_tokens = self.algorithm.top_p_perception_tokens
        self.worker.actor.use_on_entropy = self.algorithm.use_on_entropy
        self.worker.actor.top_p_entropy_tokens = self.algorithm.top_p_entropy_tokens
        self.worker.actor.use_advantage_shaping = self.algorithm.use_advantage_shaping
        self.worker.actor.advantage_scaling_min = self.algorithm.advantage_scaling_min
        self.worker.actor.use_entropy_penalty = self.algorithm.use_entropy_penalty
        self.worker.actor.entropy_penalty_coef = self.algorithm.entropy_penalty_coef

        self.worker.actor.use_token_shaping = self.algorithm.use_token_shaping
        self.worker.actor.token_shaping_weight_min = self.algorithm.token_shaping_weight_min
        self.worker.actor.token_shaping_weight_max = self.algorithm.token_shaping_weight_max

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
