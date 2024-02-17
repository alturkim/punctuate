import torch
from torch import nn
import torch.nn.functional as F

class ClassifierNet(nn.Module):
    def __init__(self, cls_input_size, config):
        super().__init__()
        self.hier_ignore_index = config["train"]["hier_ignore_index"] 
        self.num_labels = len(config["marks"]) + 1
        self.linear_hidden = config["train"]["lstm_cls"]["linear_hidden"] 
        self.binary_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.multiclass_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.74, 0.023, 0.17, 0.042, 0.034]),
            ignore_index=-100) 

        # Binary classification layer
        self.fc_binary = nn.Linear(cls_input_size, self.linear_hidden)
        self.fc_binary_out = nn.Linear(self.linear_hidden, 1)

        # Multiclass classification layer
        self.fc_multiclass = nn.Linear(cls_input_size, self.linear_hidden)
        self.fc_multiclass_out = nn.Linear(self.linear_hidden, self.num_labels-1)

    def forward(self, output, labels):
        """output is a tensor with shape batch, seq_len, hidden_size,
        where hidden_size equals cls_input_size in the constructor
        """
        # Binary classification
        # positive indicates there is a punc mark
        output_binary = F.relu(self.fc_binary(output))
        # shape [batch_size, seq_len, 1]
        binary_logits = self.fc_binary_out(output_binary)

        output_multiclass = torch.tanh(self.fc_multiclass(output))
        multiclass_logits = self.fc_multiclass_out(output_multiclass)

        # true for punc labels, false for no punc and -100 (pad token labels)        
        mask = (torch.logical_and(labels != self.hier_ignore_index, labels != -100))
        # true for pad token labels, false otherwise
        mask_neg100 = labels == -100
        # true for no punc label, false other wise
        mask_no_punc = labels == self.hier_ignore_index

        # multiclass_labels, same as labels, except that no punc label is replaced with -100
        # so it is ignored by the multiclass loss. Note: adding labels*mask_neg100 is to keep
        # the original pad token labels -100

        multiclass_labels = labels*mask + labels*mask_neg100 + -100*mask_no_punc
        # binary_labels are zero for no punc
        binary_labels = (mask.int() + labels*mask_neg100).float()
        
        binary_loss = self.binary_criterion(torch.squeeze(binary_logits, dim=-1), binary_labels)
        # masking -100 labels from loss calculation
        binary_loss_mask = labels !=100
        binary_loss = torch.mean(torch.sum(binary_loss*binary_loss_mask, axis=-1)/torch.sum(binary_loss_mask, axis=-1))
        multiclass_loss = self.multiclass_criterion(multiclass_logits.permute(0, 2, 1), multiclass_labels)
        loss = binary_loss + multiclass_loss
        return loss, binary_logits, multiclass_logits
