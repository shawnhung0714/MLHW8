import numpy as np
import random
import torch
from pathlib import Path
from config import config
import sys
import logging
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",  # "DEBUG" "WARNING" "ERROR"
    stream=sys.stdout,
)

logger = logging.getLogger("MLHW8")

def fix_seed(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def save_chekcpoint(model):
    os.makedirs(config.save_path, exist_ok=True)
    torch.save(model, Path(config.save_path) / f"model_{config.model_type}.pt")


def load_checkpoint(model):
    checkpath = Path(config.save_path) / f"model_{config.model_type}.pt"
    if checkpath.exists():
        model = torch.load(checkpath)
        model.eval()
        logger.info(
            f"loaded checkpoint {checkpath}"
        )
    else:
        logger.warn(f"no checkpoints found at {checkpath}!")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
