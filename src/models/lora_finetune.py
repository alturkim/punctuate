from peft import LoraConfig, TaskType, get_peft_model
import torch
from torch import nn
from transformers import AutoModel


class LORAFinetune(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = len(config["marks"]) + 1        
        self.config = config
        self.checkpoint = config["transformers_checkpoint"]
        self.pretrained = AutoModel.from_pretrained(self.checkpoint)
        for param in self.pretrained.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
                r=8, # rank - see hyperparameter section for more details
                lora_alpha=32, # scaling factor - see hyperparameter section for more details
                target_modules=["query", "value"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
        self.peft_model = get_peft_model(self.pretrained, lora_config)
        self.peft_model.print_trainable_parameters()

        self.linear_hidden_1 = config["train"]["classifier_net"]["mc_linear_hidden_1"]
        self.linear_hidden_2 = config["train"]["classifier_net"]["mc_linear_hidden_2"]


        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.7432966375816064, 0.02141450301537206, 0.16100701881706023, 0.04058612682252735, 0.03277620324212341, 0.0009195105213105526]),
            ignore_index=-100) 

        # Multiclass classification layers
        self.fc_multiclass_1 = nn.Linear(768, self.linear_hidden_1)
        self.fc_multiclass_2 = nn.Linear(self.linear_hidden_1, self.linear_hidden_2)
        self.fc_multiclass_out = nn.Linear(self.linear_hidden_2, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # last_hidden_state is of dim  [batch_size, seq_len, hidden_size]
        outputs = self.peft_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        outputs = torch.tanh(self.fc_multiclass_1(outputs))
        outputs = torch.tanh(self.fc_multiclass_2(outputs))
        logits = self.fc_multiclass_out(outputs)

        loss = self.criterion(logits.permute(0, 2, 1), labels)
        return loss, logits
    

 
# def create_lora_model(config):
#     lora_config = LoraConfig(
#         r=8, # rank - see hyperparameter section for more details
#         lora_alpha=32, # scaling factor - see hyperparameter section for more details
#         target_modules=["query", "value"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type=TaskType.TOKEN_CLS
#     )
#     model = BERTFinetune(config)
#     peft_model = get_peft_model(model, lora_config)
#     peft_model.print_trainable_parameters()
#     return peft_model