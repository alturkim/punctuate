"""
A base model that use a BiLSTM with Hierarchical classifier 
"""

from torch import nn
from transformers import AutoModel

class LSTMClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = len(config["marks"]) + 1
        self.checkpoint = config["train"]["transformers_checkpoint"]
        self.pretrained = AutoModel.from_pretrained(self.checkpoint)
        for param in self.pretrained.parameters():
            param.requires_grad = False

        self.lstm_hidden = config["train"]["lstm_cls"]["lstm_hidden"] 
        self.linear_hidden = config["train"]["lstm_cls"]["linear_hidden"] 
        self.embeddings_size = config["train"]["lstm_cls"]["embed_size"] 

        self.binary_criterion = nn.BCEWithLogitsLoss()
        self.multiclass_criterion = nn.CrossEntropyLoss(ignore_index=config["train"]["hier_ignore_index"])

        self.embeddings = self.pretrained.embeddings.word_embeddings
        self.bilstm = nn.LSTM(
            input_size=self.embeddings_size,
            hidden_size=self.lstm_hidden,
            num_layers=config["train"]["lstm_cls"]["lstm_layers"] , 
            batch_first=True, 
            bidirectional=True)

        # Binary classification layer
        self.fc_binary = nn.Linear(2*self.lstm_hidden, self.linear_hidden) # multiplying by 2 since its a bilstm
        self.fc_binary_out = nn.Linear(self.linear_hidden, 1)

        # Multiclass classification layer
        self.fc_multiclass = nn.Linear(2*self.lstm_hidden, self.linear_hidden)
        self.fc_multiclass_out = nn.Linear(self.linear_hidden, self.num_labels-1)


    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # shape [batch_size, seq_len, embeddings_size]
        output = self.embeddings(input_ids)
        # shape [batch_size, seq_len, 2*lstm_hidden]
        output = self.bilstm(output) # mask your input here or do you
        # Binary classification
        output_binary = self.fc_binary(output)# pos means there is a punc mark
        # shape [batch_size, seq_len, 1]
        binary_logits = self.fc_binary_out(output_binary)

        output_multiclass = self.fc_multiclass(output)
        multiclass_logits = self.fc_multiclass_out(output_multiclass)

        binary_loss = self.binary_criterion(binary_logits)
        multiclass_loss = self.multiclass_criterion(multiclass_logits)

        loss = binary_loss + multiclass_loss
        return loss, binary_logits, multiclass_logits
