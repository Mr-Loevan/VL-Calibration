

from abc import ABC, abstractmethod
from typing import Any

import torch

from ...protocol import DataProto
from .config import ActorConfig


__all__ = ["BasePPOActor"]


class BasePPOActor(ABC):
    def __init__(self, config: ActorConfig):
        
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> dict[str, Any]:
        
        pass
