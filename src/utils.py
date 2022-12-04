import torch
from dataclasses import dataclass
import os

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, size=1):
        self.count += size
        self.sum += val * size
        self.avg = self.sum / self.count

@dataclass
class Config:
    lr: float
    transformers_checkpoint: str
    checkpoint_dir: str
    marks: str
    dropout_rate: float
    batch_size: int
    num_train_epochs: int
    chunk_size: int
    tb_summary_path: str
    log_freq: int
    

def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str):
    """Saves model and training parameters
    Args:
        state: (dict) model's state_dict
        is_best: (bool) True if model is the best seen so far.
        checkpoint_dir: (str) Directory name to save checkpoint files in.
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Creating a directory {checkpoint_dir}")
        os.mkdir(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir, 'checkpoint.pt'))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'best_checkpoint.pt'))