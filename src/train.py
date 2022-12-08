from torch.utils.tensorboard import SummaryWriter
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForTokenClassification, get_scheduler
import evaluate
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import pprint

from models.baseModel import BaseModel
from models.baseModel import criterion
from utils import save_checkpoint, RunningAverage, config

from prepare_dataset import id2label, label2id, preprocess, get_raw_datasets, data_collator

tb_summary_path = 'runs/punct_pred_experiment_1'
writer = SummaryWriter(tb_summary_path)


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

def train():
    running_loss = 0.0
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')

    for epoch in range(config.num_train_epochs):
        # Training
        model.train()
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            running_loss += loss.item()
            if i % config.log_freq == config.log_freq-1:    # every log_freq mini-batches...
            # if True:
                # log the running loss
                writer.add_scalar("training loss",
                                running_loss / config.log_freq,
                                epoch * len(train_dataloader) + i)
                running_loss = 0.0

        # Evaluation
        # results each epoch of training
        val_loss = evaluation()
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        state = {'epoch': epoch+1,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': criterion,
                 }

        save_checkpoint(state, is_best, config.checkpoint_dir)
        results = compute_metrics()
        for mark in results["f1"].keys():
            writer.add_scalar("F1 for "+mark, results["f1"][mark], epoch)
    pprint(results)


def evaluation():
    avg_loss = RunningAverage()
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) # logits shape [16, 128, 9]
        labels = batch["labels"]
        avg_loss.update(outputs.loss.item())
        predictions, labels = postprocess(predictions, labels)
        for metric in metrics:
            metric.add_batch(predictions=predictions, references=labels)
    return avg_loss.avg


if __name__ == "__main__":
    raw_datasets, punc_datasets = dict(), dict()
    raw_datasets["train"] = get_raw_datasets().select([3])  
    raw_datasets["val"] = get_raw_datasets().select([3])  
    punc_datasets["train"] = preprocess(raw_datasets["train"])
    punc_datasets["val"] = preprocess(raw_datasets["val"])
    print("training dataset", punc_datasets["train"])
    print("validation dataset", punc_datasets["val"])
    
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
    # model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_checkpoint.pt"))['model_state_dict'])
    model.to(device)
 
    optimizer = AdamW(model.parameters(), lr=config.lr)

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = config.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # results before training
    evaluation()
    results = compute_metrics()
    print("before training:")
    print("results")
    pprint(results)

    # print(model)
    train()
    writer.close()