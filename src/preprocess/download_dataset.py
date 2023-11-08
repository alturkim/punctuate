from datasets import load_dataset, concatenate_datasets, arrow_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
import numpy as np
import re
import string
import logging
import os

logger = logging.getLogger(__name__)


def get_raw_datasets() -> None:
    """
    Download raw dataset, save each book in seperate txt file
    """
    # columns: book, text where book is the title and text is the content as a list of one string 
    logger.info("Downloading dataset")
    raw_datasets = load_dataset("tashkeela", split="train")#, download_mode="force_redownload")
    counter = 1
    for example in raw_datasets:
        file_name = f"{counter:02d}_{example['book'].split('/')[-1]}"
        with open(os.path.join("../../data/raw", file_name), "w") as out_f:
            out_f.write(example["text"])
        counter += 1
    logger.info("Downloading is done")
    

if __name__ == "__main__":
    # pass
    get_raw_datasets()
