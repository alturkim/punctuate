from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import TokenClassifierOutput

class BaseModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(BaseModel, self).__init__()
        self.num_labels = num_labels

        self.model = AutoModel.from_pretrained(checkpoint) # ,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768,num_labels) 

    def forward(self, input_ids=None, attention_mask=None,labels=None):
        # outputs is transformers.modeling_outputs.BaseModelOutput
        # [16, 128, 768]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask) 
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits)