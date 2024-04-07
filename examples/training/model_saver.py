from typing import Any
import torch

from xlab.core.resources import get_model_path


class ModelSaver:
    @staticmethod
    def save(model, name):
        torch.save(model, get_model_path(name))

    @staticmethod
    def load(name) -> Any:
        return torch.load(get_model_path(name))
