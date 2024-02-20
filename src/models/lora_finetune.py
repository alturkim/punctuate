from peft import LoraConfig, TaskType, get_peft_model
from bert_finetune import BERTFinetune

def create_lora_model(config):
    lora_config = LoraConfig(
        r=8, # rank - see hyperparameter section for more details
        lora_alpha=32, # scaling factor - see hyperparameter section for more details
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    model = BERTFinetune(config)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model