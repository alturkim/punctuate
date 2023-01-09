from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from datasets import load_dataset, load_from_disk

from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForTokenClassification, get_scheduler
import evaluate
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from pprint import pprint
import os
from models.baseModel import BaseModel
from models.baseModel import criterion
import utils
from utils import save_checkpoint, RunningAverage, config

from prepare_dataset import id2label, label2id, preprocess, get_raw_datasets, data_collator

writer = SummaryWriter(config.tb_summary_path)

num_of_labels = len(config.marks) + 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

metrics = [evaluate.load(m) for m in ["precision", "recall", "f1"]]

def compute_metrics()-> dict:
    result = dict()
    for metric in metrics:
        r = metric.compute(average=None, labels=[l for l in range(len(id2label))])
        m = list(r.keys())[0]
        result[m] = {id2label[i]: v for i, v in enumerate(r[m])}
    return result

def postprocess(predictions: torch.Tensor, labels: torch.Tensor):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # Remove ignored index (special tokens) 
    labels = [[l for l in label if l != -100] for label in labels]
    labels = sum(labels, [])
    predictions = sum(predictions, [])
    return predictions, labels

class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 config: utils.Config,
                 device: torch.device) -> None:

        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

        self.model.to(self.device)

    def train(self) -> None:
        running_loss = 0.0
        num_training_steps = self.config.num_train_epochs * len(self.train_dataloader)
        progress_bar = tqdm(range(num_training_steps))
        best_val_loss = float('inf')
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        for epoch in range(self.config.num_train_epochs):
            # Training
            self.model.train()
            for i, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                running_loss += loss.item()
                if i % self.config.log_freq == self.config.log_freq - 1:    # every log_freq mini-batches...
                    # log the running loss
                    writer.add_scalar("training loss",
                                    running_loss / self.config.log_freq,
                                    epoch * len(self.train_dataloader) + i)
                    running_loss = 0.0

            # Evaluation
            # results after each epoch of training
            print("evaluation ... epoch:", epoch)
            val_loss = self.evaluation()
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            state = {'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': criterion,
                    }

            save_checkpoint(state, is_best, self.config.checkpoint_dir)
            results = compute_metrics()
            for mark in results["f1"].keys():
                writer.add_scalar("F1 for "+mark, results["f1"][mark], epoch)
        pprint(results)

    def evaluation(self):
        avg_loss = RunningAverage()
        self.model.eval()

        eval_progress_bar = tqdm(range(len(self.eval_dataloader)))
        for batch in self.eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1) # logits shape [16, 128, 9]
            labels = batch["labels"]
            avg_loss.update(outputs.loss.item())
            predictions, labels = postprocess(predictions, labels)
            for metric in metrics:
                metric.add_batch(predictions=predictions, references=labels)
            eval_progress_bar.update(1)
        return avg_loss.avg


if __name__ == "__main__":
    raw_datasets, punc_datasets = dict(), dict()

    # download and preprocess original dataset
    # raw_datasets["all"] = get_raw_datasets()  
    # punc_datasets["all"] = preprocess(raw_datasets["all"])
    
    # load previously processed dataset
    punc_datasets["all"] = load_from_disk(config.processed_dataset_path)
    punc_datasets["all"] = punc_datasets["all"].remove_columns("word_ids")

    data_splits_1 = punc_datasets["all"].train_test_split(test_size=0.3, seed=42)
    punc_datasets["train"] = data_splits_1["train"]
    data_splits_2 = data_splits_1["test"].train_test_split(test_size=0.3, seed=42)
    punc_datasets["val"] = data_splits_2["train"]
    punc_datasets["test"] = data_splits_2["test"]

    print("training dataset", punc_datasets["train"])
    print("validation dataset", punc_datasets["val"])
    print("testing dataset", punc_datasets["test"])

    train_dataloader = DataLoader(
        punc_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config.batch_size,
    )

    eval_dataloader = DataLoader(
        punc_datasets["val"], collate_fn=data_collator, batch_size=config.batch_size
    )

    model = BaseModel(config.transformers_checkpoint, num_of_labels)

    # load best checkpoint
    if config.load_checkpoint:
        model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, "best_checkpoint.pt"))['model_state_dict'])
    model.to(device)
 
    trainer = Trainer(model, train_dataloader, eval_dataloader, config, device)

    # results before training
    print("evaluating before training")
    trainer.evaluation()
    results = compute_metrics()
    print("results")
    pprint(results)

    # print(model)
    print("training ... ")
    trainer.train()
    writer.close()