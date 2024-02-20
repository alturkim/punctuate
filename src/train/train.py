"""Load split dataset, train and evaluate a model from the models dir
"""
import glob
import logging
import torch
from torch import nn
from datasets import load_dataset, load_from_disk

from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler, DataCollatorForTokenClassification, AutoTokenizer
import evaluate
from torch.optim import AdamW
from tqdm.auto import tqdm
import os

import wandb
from models.bert_finetune import BERTFinetune
from models.lstm_classifier import LSTMClassifier
from models.largeModel import LargeModel
from utils import save_checkpoint, RunningAverage, get_id2label, get_label2name
from omegaconf import DictConfig
from pprint import pprint
torch.manual_seed(0)
logger = logging.getLogger(__name__)

def postprocess(predictions: torch.Tensor, labels: torch.Tensor):
    """utility to flatten the list of lists of predictions and labels,
       and also ignore predictions corresponding to the label -100
       which is ignored in the loss calculation.
    """
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    predictions = [
        [p for (p, l) in zip(prediction_sublst, label_sublst) if l != -100]
        for prediction_sublst, label_sublst in zip(predictions, labels)
    ]
    # Remove ignored index
    labels = [[l for l in label if l != -100] for label in labels]
    # flatten the labels list of list
    labels = sum(labels, [])
    predictions = sum(predictions, [])
    return predictions, labels

def process_logits(binary_logits: torch.Tensor, multiclass_logits: torch.Tensor, config):
    """the model generates two logits, one from each classifier,this maps those to
    one prediction tensor corresponding to the original labels."""
    no_punc_label = config["train"]["hier_ignore_index"]
    multiclass_predictions = torch.argmax(multiclass_logits, dim=-1)
    binary_logits = torch.squeeze(torch.sigmoid(binary_logits), -1)
    preds = torch.where(binary_logits > 0.5, multiclass_predictions, no_punc_label)
    return preds

class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 config: DictConfig,
                 device: torch.device) -> None:

        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.device = device
        self.optimizer = AdamW([params for params in self.model.parameters() if params.requires_grad],
                                lr=self.config["train"]["lr"])
        self.metrics = [evaluate.load(m) for m in ["precision", "recall", "f1"]]
        self.id2label = get_id2label(self.config)
        self.label2name = get_label2name(self.config)

        logger.info(f"Training:- Moving the model to {device}")
        self.model.to(self.device)
        logger.info(f"Training:- Model is now on {device}")

    def train(self) -> None:
        log_freq = self.config["train"]["log_freq"]
        # log gradients
        wandb.watch(self.model, log_freq=log_freq)
        # running_loss is avg loss of a number of minibatch that equals to log_freq
        running_loss = RunningAverage()
        # training_loss is avg loss per epoch
        training_loss = RunningAverage()
        # wandb table
        performance_stats_lst = []

        if not self.config["train"]["debug"]:
            num_epochs = self.config["train"]["num_train_epochs"]
        else:
            num_epochs = self.config["train"]["debug_num_train_epochs"] 

        # len(self.train_dataloader) is the number of minibatch in one epoch (data size / batch size)
        num_training_steps = num_epochs * len(self.train_dataloader)
        progress_bar = tqdm(range(num_training_steps))
        best_val_loss = float('inf')

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(0.20 * num_training_steps),
            num_training_steps=num_training_steps,
        )

        logger.info(f"Training:- Starting training loop for {num_epochs} epoch(s)")
        for epoch in range(num_epochs):
            # Training
            training_loss.reset()
            self.model.train()

            for i, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # forward passs
                loss, binary_logits, multiclass_logits = self.model(**batch)
                # backward pass
                loss.backward()

                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()

                progress_bar.update(1)
                running_loss.update(loss.item())
                training_loss.update(loss.item())

                # every log_freq mini-batches...
                if (i+1)%log_freq == 0:
                    wandb.log({"step":epoch * len(self.train_dataloader) + i,
                                "mini_batch_loss": running_loss.avg}) 
                    running_loss.reset()


            # Evaluation
            # results after each epoch of training
            logger.info(f"Training:- Evaluation for epoch: {epoch}")
            val_loss = self.evaluate()
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            state = {'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss
                    }

            save_checkpoint(state, is_best, self.config["train"]["checkpoint_dir"])
            if is_best:
                model_artifact = wandb.Artifact("best_checkpoint", "model_checkpoint")
                model_artifact.add_file(os.path.join(
                    self.config["train"]["checkpoint_dir"], 'best_checkpoint.pt'),
                    name="best_model")
                wandb.run.log_artifact(model_artifact)
            perf_stat = []
            perf_stat.append(epoch)
            perf_stat.append(training_loss.avg)
            perf_stat.append(val_loss)

            results:dict = self.compute_metrics()
            pprint({"epoch": epoch, 
                       "train_loss": training_loss.avg,
                       "val_loss": val_loss,
                        **results["f1"]})
            
            wandb.log({"epoch": epoch, 
                       "train_loss": training_loss.avg,
                       "val_loss": val_loss,
                        **results["f1"]})
            for mark in results["f1"].keys():
                perf_stat.append(results["f1"][mark])
            performance_stats_lst.append(perf_stat)
        # the table is not going to be logged until the end of training, above, there is frequent log
        performance_table = wandb.Table(data=performance_stats_lst, 
                                        columns=["Epoch", "Training Loss", "Validation Loss",
                                                *[f"F1_{mark}" for mark in results["f1"].keys()]])
        logger.info("Training:- Logging Performance Table to WandB ...")
        wandb.log({"Performance Statistics": performance_table})
    
    def evaluate(self):
        avg_loss = RunningAverage()
        self.model.eval()

        eval_progress_bar = tqdm(range(len(self.eval_dataloader)))
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                loss, binary_logits, multiclass_logits = self.model(**batch)
            predictions = process_logits(binary_logits, multiclass_logits, self.config)
            labels = batch["labels"]
            assert labels.shape[1] == predictions.shape[1], "check pred in evaluation loop"
            avg_loss.update(loss.item())
            # predictions and labels are lists of integers
            predictions, labels = postprocess(predictions, labels)
           
            for metric in self.metrics:
                metric.add_batch(predictions=predictions, references=labels)

            eval_progress_bar.update(1)
        return avg_loss.avg

    def compute_metrics(self) -> dict:
        result = dict()
        all_labels = [l for l in range(len(self.id2label))]
        for metric in self.metrics:
            # metric.compute takes care of reseting the computation
            if metric.name == "precision":
                metric_result:list[float] = metric.compute(average=None, labels=all_labels, zero_division=0)[metric.name]
            else:
                metric_result:list[float] = metric.compute(average=None, labels=all_labels)[metric.name]


            result[metric.name] = {self.label2name[self.id2label[i]]: v 
                                   for i, v in zip(all_labels, metric_result)}
        return result

def get_dataloader(config, dataset, split):
    # data_collator will take care of padding sequences and labels
    tokenizer = AutoTokenizer.from_pretrained(config["transformers_checkpoint"])
    hf_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)
    batch_size = config["train"]["batch_size"]
    def lstm_collator(features):
        input_ids_lengths = [len(features[i]["input_ids"]) for i in range(len(features))]
        labels_lengths = [len(features[i]["labels"]) for i in range(len(features))]

        batch = hf_collator.torch_call(features)
        input_ids:torch.Tensor = batch["input_ids"]
        labels:torch.Tensor = batch["labels"]
        # packed_input_ids = torch.nn.utils.rnn.pack_padded_sequence(input_ids,
        #     lengths= input_ids_lengths,
        #     batch_first=True,
        #     enforce_sorted=False) 
        # packed_labels = torch.nn.utils.rnn.pack_padded_sequence(labels,
        #     lengths=labels_lengths,
        #     batch_first=True,
        #     enforce_sorted=False)
        
        return {"input_ids": input_ids, "labels": labels}

    if config["train"]["model_class"].find("LSTM") >= 0:
        collate_fn = lstm_collator   
    else:
        collate_fn = hf_collator

    if config["train"]["debug"]:
        dataset[split] = dataset[split].select(list(range(config["train"]["debug_samples"])))
        logger.info("Training:- Debugging mode, using few samples...")
        batch_size = config["train"]["debug_batch_size"]
    
    if split == "val_on_train":
        print("using part of train for val,, debugging")
        dataset[split] = dataset["train"].select(list(range(5000)))
        logger.info("Training:- Debugging mode, using few samples...")

    dataloader = DataLoader(
                dataset[split],
                shuffle=True,
                collate_fn=collate_fn,
                batch_size=batch_size
            )
    return dataloader
    


def go(config: DictConfig):
    """each split is a dict-like object containing these keys, the value of each is a list
    "attention_mask", "input_ids", "labels", "token_type_ids", "word_ids"
    data are truncated to be of size config tokenz_max_len, and special tokens were added.
    """
    run = wandb.init(job_type="training")
    config_log = config.copy()
    del config_log.marks
    del config_log.mark2name
    run.config.update(config_log)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Training:- Running on: " + str(device))
    
    # TODO verify that transformers checkpoint used to generate the processed data is the same used for training
    # load previously processed dataset
    logger.info("Training:- Loading dataset...")
    punc_datasets = dict()
    for split in ("train", "val", "test"):
        punc_datasets[split] = load_from_disk(os.path.join(config["train"]["input_path"], split))
        punc_datasets[split] = punc_datasets[split].remove_columns("word_ids")
        # TODO remove this special case
        if config["transformers_checkpoint"].startswith("CAMeL"):
            punc_datasets[split] = punc_datasets[split].remove_columns("token_type_ids")
    logger.info("Training:- Loading dataset is done.")

    logger.info("Training:- Creating Data Loaders...")

    train_dataloader = get_dataloader(config, punc_datasets, "train")
    eval_dataloader = get_dataloader(config, punc_datasets, "val")

    logger.info("Training:- Creating Data Loaders, done.")

    logger.info(f"Training:- Creating a {config['train']['model_class']} model...")
    if config["train"]["model_class"] == "LargeModel":
        model = LargeModel(config)
    elif config["train"]["model_class"] == "LSTMCLS":
        model = LSTMClassifier(config)
    elif config["train"]["model_class"] == "BERT":
        model = BERTFinetune(config)
    logger.info("Training:- Model creation is done.")
    
    # load best checkpoint
    if config["train"]["load_checkpoint"]:
        logger.info("Training:- Loading best checkpoint...")
        model.load_state_dict(
            torch.load(os.path.join(
                config["train"]["checkpoint_dir"], "best_checkpoint.pt"))['model_state_dict'])
        logger.info("Training:- Checkpoint loading is done.")

    logger.info("Training:- Creating a Trainer object")
    trainer = Trainer(model, train_dataloader, eval_dataloader, config, device)
    logger.info("Training:- Trainer object is created")

    # results before training
    logger.info("Training:- Evaluating the model BEFORE training ... ")
    trainer.evaluate()
    results = trainer.compute_metrics()
    pprint(results)
    logger.info("Training:- Calling trainer.train() ... ")
    trainer.train()
    logger.info("Training:- Done ... ")


if __name__ == "__main__":
    pass