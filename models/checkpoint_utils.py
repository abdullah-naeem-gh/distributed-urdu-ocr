"""
Checkpoint loading utilities for deployment.
"""

from collections import OrderedDict
from typing import Any, Dict, Mapping

import torch


_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "net", "weights")


def _looks_like_state_dict(obj: Mapping[str, Any]) -> bool:
    if not obj:
        return False
    first_value = next(iter(obj.values()))
    return isinstance(first_value, torch.Tensor)


def extract_state_dict(
    checkpoint_obj: Any,
    checkpoint_path: str,
) -> Dict[str, torch.Tensor]:
    """
    Normalize different checkpoint formats into a plain PyTorch state_dict.
    """
    if isinstance(checkpoint_obj, OrderedDict) and _looks_like_state_dict(checkpoint_obj):
        return dict(checkpoint_obj)

    if isinstance(checkpoint_obj, Mapping):
        if _looks_like_state_dict(checkpoint_obj):
            return dict(checkpoint_obj)

        for key in _STATE_DICT_KEYS:
            candidate = checkpoint_obj.get(key)
            if isinstance(candidate, Mapping) and _looks_like_state_dict(candidate):
                return dict(candidate)

        raise ValueError(
            f"Unsupported checkpoint format at {checkpoint_path}. "
            f"Expected a state_dict or one of {_STATE_DICT_KEYS} containing a state_dict."
        )

    raise TypeError(
        f"Unsupported checkpoint object type at {checkpoint_path}: "
        f"{type(checkpoint_obj).__name__}"
    )


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove DataParallel/DDP 'module.' prefix when present.
    """
    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict
