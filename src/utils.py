from collections import Counter
from omegaconf import DictConfig
import torch
from dataclasses import dataclass, fields
import os
import typing
import json

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
    wand_project: str
    debug: bool
    lr: float
    transformers_checkpoint: str
    embed_size: int
    model_class: str
    checkpoint_dir: str
    marks: str
    mark2name: dict
    batch_size: int
    num_train_epochs: int
    chunk_size: int
    tb_summary_path: str
    log_freq: int
    text_chunk_size: int
    processed_dataset_path: str
    load_checkpoint: bool
    load_processed_dataset: bool


    @classmethod
    def from_dict(cls: typing.Type["Config"], arg_dict: dict):
        field_set = {f.name for f in fields(cls) if f.init}
        filtered_arg_dict = {k : v for k, v in arg_dict.items() if k in field_set}
        return cls(**filtered_arg_dict, debug=False)
 

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

def count_labels(labels:list[list[int]]):
    # Flatten the list of lists into a single list
    flattened_list = [item for sublist in labels for item in sublist]
    # Use Counter to count the occurrences of each unique label
    counts = Counter(flattened_list)
    return counts

def calculate_percentage(counter: Counter):
    total_counts = sum(counter.values())
    
    if total_counts == 0:
        # Avoid division by zero if the counter is empty
        return dict.fromkeys(counter, 0.0)

    percentage_dict = {key: count / total_counts * 100 for key, count in counter.items()}
    return percentage_dict


def get_id2label(config: DictConfig):
    id2label = {i: label for i, label in enumerate(config["marks"]+"O")}
    return id2label

def get_label2id(config: DictConfig):
    id2label = get_id2label(config)
    label2id = {v: k for k, v in id2label.items()}
    return label2id

def get_label2name(config: DictConfig):
    label2name = {k: config["mark2name"][k] for k in config["marks"]+"O"}
    return label2name

def get_token_stats(input_ids: list[list]) -> dict:
    """recieve list of lists containing input_ids (encoded token), where each inner list is a sample.
    Returns: min, max, avg token length.
    """
    len_list = [len(inner_list) for inner_list in input_ids]
    min_len = min(len_list)
    max_len = max(len_list)
    avg_len = sum(len_list)/len(len_list)
    return {"min_length": min_len, "max_length": max_len, "avg_length": avg_len}

# ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))
# arg_dict = json.load(open(os.path.join(ROOT_DIR, 'config', 'config.json'), encoding='utf-8'))
# # append project root to relative paths
# for k in ["checkpoint_dir", "tb_summary_path", "processed_dataset_path"]:
#     arg_dict[k] = os.path.join(ROOT_DIR, arg_dict[k])
# config = Config.from_dict(arg_dict)