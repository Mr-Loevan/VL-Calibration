

from torch.distributed.device_mesh import DeviceMesh

from ...protocol import DataProto, all_gather_data_proto
from ...utils.ulysses import get_ulysses_sequence_parallel_group, set_ulysses_sequence_parallel_group
from .base import BaseShardingManager


class FSDPUlyssesShardingManager(BaseShardingManager):
    

    def __init__(self, device_mesh: DeviceMesh):
        super().__init__()
        self.device_mesh = device_mesh

    def __enter__(self):
        if self.device_mesh is not None:
            self.prev_sp_group = get_ulysses_sequence_parallel_group()
            set_ulysses_sequence_parallel_group(self.device_mesh["sp"].get_group())

    def __exit__(self, exc_type, exc_value, traceback):
        if self.device_mesh is not None:
            set_ulysses_sequence_parallel_group(self.prev_sp_group)

    def preprocess_data(self, data: DataProto) -> DataProto:
        
        if self.device_mesh is not None:
            sp_size = self.device_mesh["sp"].size()
            sp_group = self.device_mesh["sp"].get_group()
            all_gather_data_proto(data, size=sp_size, group=sp_group)

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        
        if self.device_mesh is not None:
            sp_size = self.device_mesh["sp"].size()
            sp_rank = self.device_mesh["sp"].get_local_rank()
            data = data.chunk(chunks=sp_size)[sp_rank]

        return data
