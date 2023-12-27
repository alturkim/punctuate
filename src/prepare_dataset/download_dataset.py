from datasets import load_dataset
import numpy as np
import logging
import os
import wandb

logger = logging.getLogger(__name__)

def get_raw_datasets(output_path: str) -> None:
    """
    Download raw dataset, save each book in seperate txt file
    """
    # columns: book, text where book is the title and text is the content as a list of one string 
    
    raw_datasets = load_dataset("tashkeela", split="train")#, download_mode="force_redownload")
    counter = 1
    for example in raw_datasets:
        file_name = f"{counter:02d}_{example['book'].split('/')[-1]}"
        with open(os.path.join(output_path, file_name), "w") as out_f:
            out_f.write(example["text"])
        counter += 1
    
def go(config):
    run = wandb.init(job_type="download_raw_data")
    run.config.update(config["download"])
    logger.info("Downloading dataset")
    get_raw_datasets(config["download"]["output_path"])
    logger.info("Downloading is done")

    logger.info(f"Uploading a sample file from {config['download']['output_path']} to Weights & Biases")

    artifact = wandb.Artifact(
        "sample_raw_data.txt",
        type="raw data",
        description="A sample raw data file which represent one book",
    )
    artifact.add_file(os.path.join(config['download']['output_path'], "01_أحكام القرآن لابن العربي.txt"))
    run.log_artifact(artifact)
    artifact.wait()

if __name__ == "__main__":
    # pass
    get_raw_datasets()
