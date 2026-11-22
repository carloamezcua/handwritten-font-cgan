import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Fija semillas para reproducibilidad en random, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False