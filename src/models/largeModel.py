from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import TokenClassifierOutput

from .baseModel import criterion

class LargeModel(nn.Module):
    def __init__(self, config):
        super(LargeModel, self).__init__()
        self.num_labels = len(config.marks)+1
        self.embed_size = config.embed_size
        self.model = AutoModel.from_pretrained(config.transformers_checkpoint)
        for param in self.model.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(self.embed_size, self.embed_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.embed_size, self.num_labels) 

    def forward(self, input_ids=None, attention_mask=None,labels=None):
        # outputs is transformers.modeling_outputs.BaseModelOutput
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask) 
        outputs = self.linear(outputs.last_hidden_state)
        outputs = nn.ReLU()(outputs)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        loss = None
        if labels is not None:
            loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits)