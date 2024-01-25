"""Load preprocessed data and create a train, val, test sets and save them to disk.
"""

import os
import logging

from omegaconf import DictConfig
from datasets import Dataset, load_from_disk, concatenate_datasets
import wandb
from utils import count_labels, calculate_percentage, get_id2label, get_token_stats

logger = logging.getLogger(__name__)

def load_dataset(config: DictConfig) -> Dataset:
    """Load preprocessed dataset, book by book, and return one Dataset object concatenating all books
    """
    data_home_dir = config["split"]["input_path"]
    list_book_names = [dir_name for dir_name in os.listdir(data_home_dir) 
                       if os.path.isdir(os.path.join(data_home_dir, dir_name))]
    
    datasets_lst: list[Dataset] = []
    for book in list_book_names:
        datasets_lst.append(load_from_disk(os.path.join(data_home_dir, book)))
    combined_dataset = concatenate_datasets(datasets_lst)
    return combined_dataset

def split_data(datasets: Dataset):
    """Split the preprocessed dataset into 0.6 train, 0.2 val, and 0.2 test
    """
    data_splits_1 = datasets.train_test_split(test_size=0.4, seed=42)
    train = data_splits_1["train"]
    data_splits_2 = data_splits_1["test"].train_test_split(test_size=0.5, seed=42)
    val = data_splits_2["train"]
    test = data_splits_2["test"]
    return train, val, test

def go(config: DictConfig):
    run = wandb.init(job_type="splitting_data")
    run.config.update(config)

    id2label = get_id2label(config)
    
    logger.info("Splitting dataset")
    dataset = load_dataset(config)
    train, val, test = split_data(dataset)
    stats = []
    for split, name in zip([train, val, test], ["train", "val", "test"]): 
        split.save_to_disk(os.path.join(config["split"]["output_path"], name))

        # generate label stats for each split
        split_label_count = count_labels(split["labels"])
        split_label_perc = calculate_percentage(split_label_count)

        # generate token stat for each split
        len_stats = get_token_stats(split["input_ids"])

        sample_count = len(split["input_ids"])
        # combine stats in a list
        split_stat = [
            name,
            len_stats["min_length"],
            len_stats["max_length"],
            len_stats["avg_length"],
            sample_count,
            *[f"{split_label_perc.get(key, 0):.4f}" for key in sorted(id2label.keys())],
            *[split_label_count.get(key, 0) for key in sorted(id2label.keys())]
        ]
        stats.append(split_stat)
    logger.info("Splitting is done")
    stats_table = wandb.Table(data=stats, columns=["Split",
                                                   "Min Token Count",
                                                   "Max Token Count",
                                                   "Avg Token Count",
                                                   "Sample Count",
                                                   *[f"perc({id2label[key]})" for key in sorted(id2label.keys())],
                                                   *[f"count({id2label[key]})" for key in sorted(id2label.keys())]
                                                    ])
    run.log({"Split Statistics": stats_table})
