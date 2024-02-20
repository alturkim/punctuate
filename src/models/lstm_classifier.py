"""
A base model that use a BiLSTM with Hierarchical classifier 
"""
import torch
from torch import nn
from transformers import AutoModel
import torch.nn.functional as F
from .classifier_net import ClassifierNet
class LSTMClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.checkpoint = config["transformers_checkpoint"]
        self.pretrained = AutoModel.from_pretrained(self.checkpoint)
        for param in self.pretrained.parameters():
            param.requires_grad = False

        self.lstm_hidden = config["train"]["lstm_cls"]["lstm_hidden"] 
        self.embeddings_size = config["train"]["lstm_cls"]["embed_size"] 

        self.embeddings = self.pretrained.embeddings.word_embeddings
        # self.embeddings = nn.Embedding(30000, self.embeddings_size)

        self.bilstm = nn.LSTM(
            input_size=self.embeddings_size,
            hidden_size=self.lstm_hidden,
            num_layers=config["train"]["lstm_cls"]["lstm_layers"] , 
            batch_first=True, 
            bidirectional=True)
        self.hier_ignore_index = config["train"]["hier_ignore_index"] 
        self.num_labels = len(config["marks"]) + 1
        self.mc_linear_hidden_1 = config["train"]["classifier_net"]["mc_linear_hidden_1"]
        self.mc_linear_hidden_2 = config["train"]["classifier_net"]["mc_linear_hidden_2"]


        self.multiclass_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.7432966375816064, 0.02141450301537206, 0.16100701881706023, 0.04058612682252735, 0.03277620324212341, 0.0009195105213105526]),
            ignore_index=-100) 

        # Multiclass classification layer
        self.fc_multiclass_1 = nn.Linear(2*self.lstm_hidden, self.mc_linear_hidden_1)
        self.fc_multiclass_2 = nn.Linear(self.mc_linear_hidden_1, self.mc_linear_hidden_2)
        self.fc_multiclass_out = nn.Linear(self.mc_linear_hidden_2, self.num_labels)

    def forward(self, input_ids=None, labels=None):
        # shape [batch_size, seq_len, embeddings_size]
        output = self.embeddings(input_ids)
        # shape [batch_size, seq_len, 2*lstm_hidden]
        output, _ = self.bilstm(output)
        

        output_multiclass_1 = torch.tanh(self.fc_multiclass_1(output))
        output_multiclass_2 = self.fc_multiclass_2(output_multiclass_1)

        logits = self.fc_multiclass_out(output_multiclass_2)

        loss = self.multiclass_criterion(logits.permute(0, 2, 1), labels)
        return loss, logits
