"""Device management utilities for CUDA/CPU selection."""

import logging

import torch

from src.config import DeviceType

logger = logging.getLogger(__name__)


def get_device(device_preference: DeviceType | None = None) -> torch.device:
    """
    Determine the best available device for model inference.

    Args:
        device_preference: Preferred device ("cuda" or "cpu"). If None, auto-selects.

    Returns:
        torch.device: The selected device.
    """
    if device_preference is None:
        device_preference = "cuda" if torch.cuda.is_available() else "cpu"

    if device_preference == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        free_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using CUDA device. Total VRAM: {free_vram_gb:.2f} GB")
        return device
    else:
        logger.info("Using CPU device (CUDA not available or not requested)")
        return torch.device("cpu")


def get_free_vram_gb() -> float | None:
    """
    Get free VRAM in GB if CUDA is available.

    Returns:
        Free VRAM in GB, or None if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return None

    return torch.cuda.get_device_properties(0).total_memory / (1024**3)
