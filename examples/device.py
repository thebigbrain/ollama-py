import logging
import torch


logger = logging.getLogger(__file__)


def get_device():
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.debug(f"Using {device} device")

    return device
