"""
BERT-based model
"""
import torch
from torch import nn
from transformers import AutoModel


class BERTFinetune(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Linear(768, self.num_labels) 
        
        self.config = config
        self.checkpoint = config["transformers_checkpoint"]
        self.model = AutoModel.from_pretrained(self.checkpoint)
        for param in self.model.parameters():
            param.requires_grad = False

        self.num_labels = len(config["marks"]) + 1
        self.linear_hidden_1 = config["train"]["classifier_net"]["mc_linear_hidden_1"]
        self.linear_hidden_2 = config["train"]["classifier_net"]["mc_linear_hidden_2"]


        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.7432966375816064, 0.02141450301537206, 0.16100701881706023, 0.04058612682252735, 0.03277620324212341, 0.0009195105213105526]),
            ignore_index=-100) 

        # Multiclass classification layers
        self.fc_multiclass_1 = nn.Linear(768, self.mc_linear_hidden_1)
        self.fc_multiclass_2 = nn.Linear(self.mc_linear_hidden_1, self.mc_linear_hidden_2)
        self.fc_multiclass_out = nn.Linear(self.mc_linear_hidden_2, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # last_hidden_state is of dim  [batch_size, seq_len, hidden_size]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        outputs = nn.tanh(self.fc_multiclass_1(outputs))
        outputs = nn.tanh(self.fc_multiclass_2(outputs))
        logits = self.fc_multiclass_out(outputs)

        loss = self.criterion(logits.permute(0, 2, 1), labels)
        return loss, logits
    

 