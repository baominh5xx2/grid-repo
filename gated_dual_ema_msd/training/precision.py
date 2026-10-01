"""BF16 training policy; CPU verification and evaluation remain FP32."""
from __future__ import annotations

import torch


def bf16_enabled(device: torch.device, requested: bool = True) -> bool:
    if not requested or device.type != "cuda":
        return False
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "BF16 training requires a CUDA GPU with BF16 support. "
                "Select a supported GPU before starting this run."
            )
    return True
