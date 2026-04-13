
import inspect
import re
from typing import Iterable, Union

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel
from vllm import LLM
from vllm.distributed import parallel_state as vllm_ps

from ...protocol import DataProto, all_gather_data_proto
from ...utils.fsdp_utils import load_fsdp_model, offload_fsdp_model
from ...utils.model_utils import print_gpu_memory_usage
from .base import BaseShardingManager


class FSDPVLLMShardingManager(BaseShardingManager):
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        device_mesh: DeviceMesh,
        use_param_offload: bool,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        self.use_param_offload = use_param_offload
        self.loaded = False

        self.world_size = dist.get_world_size()
        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group

        self.freed_bytes = 0

        self.torch_random_states = torch.cuda.get_rng_state()
        gen_dp_rank = self.device_mesh["dp"].get_local_rank()
        torch.cuda.manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.torch_random_states)

    def _rename_weight_keys(self, actor_weights: dict[str, Union[torch.Tensor, DTensor]], model: PreTrainedModel):
        if not hasattr(model, "_checkpoint_conversion_mapping"):
            return actor_weights

        reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
        original_weights = {}
        for key, value in actor_weights.items():
            for pattern, replacement in reverse_key_mapping.items():
                replacement = replacement.lstrip("^")
                replacement = re.sub(r"\(.*\)", "", replacement)
                key, n_replace = re.subn(pattern, replacement, key)
                if n_replace > 0:
                    break

            original_weights[key] = value

        return original_weights

    def _make_weight_iterator(
        self, actor_weights: dict[str, Union[torch.Tensor, DTensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in actor_weights.items():
            yield name, tensor.full_tensor() if self.world_size != 1 else tensor

    def _sync_weight_to_vllm(self):
        if self.use_param_offload:
            load_fsdp_model(self.module)

        actor_weights = get_model_state_dict(self.module)
        actor_weights = self._rename_weight_keys(actor_weights, self.module._fsdp_wrapped_module)
        print_gpu_memory_usage("After gather model weights in sharding manager")

        model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        model.load_weights(self._make_weight_iterator(actor_weights))

        del actor_weights
        if self.use_param_offload:
            offload_fsdp_model(self.module)

        torch.cuda.empty_cache()
        print_gpu_memory_usage("After sync model weights in sharding manager")

    def load_vllm_and_sync_weights(self):
        
        torch.cuda.empty_cache()
        assert self.loaded is False, "vllm engine has already been loaded"
        self.loaded = True

        print_gpu_memory_usage("Before vllm wake up in sharding manager")
        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["weights"])
        else:
            self.inference_engine.wake_up()

        self._sync_weight_to_vllm()

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["kv_cache"])

        print_gpu_memory_usage("After vllm wake up in sharding manager")
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def offload_vllm(self):
        
        assert self.loaded is True, "vllm engine has not been loaded"
        self.loaded = False

        print_gpu_memory_usage("Before vllm offload in sharding manager")
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]
        self.inference_engine.sleep(level=1)
        free_bytes_after_sleep = torch.cuda.mem_get_info()[0]
        self.freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        print_gpu_memory_usage("After vllm offload in sharding manager")

        self.module.train()
        torch.cuda.empty_cache()

        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        
        all_gather_data_proto(data, size=self.tp_size, group=self.tp_group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        
        if self.tp_size > 1:
            data = data.chunk(chunks=self.tp_size)[self.tp_rank]

        return data
