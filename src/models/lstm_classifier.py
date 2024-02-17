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
        self.bilstm = nn.LSTM(
            input_size=self.embeddings_size,
            hidden_size=self.lstm_hidden,
            num_layers=config["train"]["lstm_cls"]["lstm_layers"] , 
            batch_first=True, 
            bidirectional=True)
        self.classifier_net = ClassifierNet(2*self.lstm_hidden, self.config)

    def forward(self, input_ids=None, labels=None):
        # labels = labels.data
        # shape [batch_size, seq_len, embeddings_size]
        output = self.embeddings(input_ids)
        # shape [batch_size, seq_len, 2*lstm_hidden]
        output, _ = self.bilstm(output) # mask your input here or do you
        loss, binary_logits, multiclass_logits = self.classifier_net(output, labels)
        return loss, binary_logits, multiclass_logits
