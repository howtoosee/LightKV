from torch import nn
from abc import ABC, abstractmethod
from typing import Tuple


class LightKVBase(nn.Module, ABC):
    @abstractmethod
    def set_img_indices(self, img_indices: Tuple[Tuple[Tuple[int, int]]]) -> None:
        raise NotImplementedError("This method is not implemented yet.")


    @abstractmethod
    def forward(self):
        raise NotImplementedError("This method is not implemented yet.")
