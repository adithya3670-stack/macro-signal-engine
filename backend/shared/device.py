from __future__ import annotations

import os
import warnings

import torch


def _mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def resolve_torch_device(preferred: str | None = None) -> torch.device:
    """
    Resolve training/inference device from preference or environment.
    Supported values: auto, cpu, cuda, cuda:<index>, mps.
    """
    value = str(preferred or os.getenv("MACRO_TORCH_DEVICE", "auto")).strip().lower()

    if value in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if value == "cpu":
        return torch.device("cpu")

    if value == "mps":
        if _mps_available():
            return torch.device("mps")
        warnings.warn("MACRO_TORCH_DEVICE='mps' requested but MPS is unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            warnings.warn(f"MACRO_TORCH_DEVICE='{value}' requested but CUDA is unavailable. Falling back to CPU.")
            return torch.device("cpu")
        try:
            candidate = torch.device(value)
        except Exception:
            warnings.warn(f"Invalid CUDA device '{value}'. Falling back to default CUDA device.")
            return torch.device("cuda")

        if candidate.index is not None:
            count = torch.cuda.device_count()
            if candidate.index < 0 or candidate.index >= count:
                warnings.warn(
                    f"CUDA device index {candidate.index} is out of range (available: 0..{max(count - 1, 0)}). "
                    "Falling back to default CUDA device."
                )
                return torch.device("cuda")
        return candidate

    warnings.warn(f"Unknown MACRO_TORCH_DEVICE='{value}'. Falling back to auto selection.")
    return resolve_torch_device("auto")


def configure_torch_runtime(device: torch.device) -> None:
    """
    Apply runtime optimizations for selected device.
    """
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def use_amp(device: torch.device) -> bool:
    return device.type == "cuda"


def use_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"
