from datasets import load_dataset, concatenate_datasets, arrow_dataset, Dataset
from transformers import AutoTokenizer
import numpy as np
from typing import List, Union
import logging
import os
import glob
import wandb
from itertools import chain
from collections import Counter

logger = logging.getLogger(__name__)

def _create_labels(examples: List[str], marks: str, label2id) -> List[List[str]]:
    """
    Create labels at the word level (words are whatever separated by space), 
    where each word will have a label indicating the punctuation mark that 
    follows it, if any, and O if none.
    """
    labels = []
    for ex in examples:
        temp = []
        words = ex.strip().split()
        for idx, w in enumerate(words):
            w = w.strip()
            if w in marks:
                continue
            last_char = w[-1]
            if last_char in marks:
                # punc mark is attached to the end of the current word
                temp.append(label2id[last_char])
            elif idx < len(words) -1:
                # not the last word in this ex
                if words[idx+1][0] in marks:
                    # punct is attached to the next word or is the next word (isolated punc)
                    temp.append(label2id[words[idx+1][0]])
                else:
                    temp.append(label2id["O"])
            else:
              temp.append(label2id["O"])
          
        labels.append(temp)
    return labels

def _remove_punc(examples: List[str], marks: str) -> List[str]:
    table = str.maketrans("", "", marks)
    new_examples = []
    for ex in examples:
        new_examples.append(ex.translate(table))
    return new_examples

def align_labels_with_tokens(labels:List[str], word_ids: List[Union[int, str]], input_ids: List[int]) -> List[int] :
    # TODO may the label should be assigned to the last token instead of first
    """
    Make label list aligned with the tokens list. 
    Special tokens get a label of -100, to be ignored by the loss function.
    A word is divided into multiple tokens, only the first token gets the label of the word, and -100 is assigned to the other subtokens. 
    This is to avoid long words that split into lots of subtokens contributing heavily to the loss. 
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)              
        else:
            # Same word as previous token or a special token
            new_labels.append(-100)
    return new_labels

def _tokenize_and_align_labels(examples: list[str], tokenizer, label2id, config):
    """ returns transformers.tokenization_utils_base.BatchEncoding (a dict subclass)  
    """
    all_labels: List[List[int]] = _create_labels(examples, config["marks"], label2id)

    # remove marks from text before tokenization
    examples = _remove_punc(examples, config.marks)

    # tokenized_inputs is transformers.tokenization_utils_base.BatchEncoding  (a dict subclass) 
    # with keys dict_keys(['input_ids', 'attention_mask'])
    # the value of each key in tokenized_inputs is a List[List[int]] where each sub list represent one point in the dataset 
    tokenized_inputs = tokenizer(examples, add_special_tokens=False)
    
    # align labels with tokens
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids, tokenized_inputs.input_ids[i]))

    tokenized_inputs["labels"] = new_labels
    tokenized_inputs["word_ids"] = [tokenized_inputs.word_ids(i) for i in range(len(tokenized_inputs["input_ids"]))]
    return tokenized_inputs

def _group_samples(tokenized_inputs, original, config) -> list[list[int]]:
    max_length = config["tokens_max_len"]-2
    combined_lists_dict = {"input_ids": [], "word_ids": [], "labels": [], "attention_mask": []}
    combined_original = []
    curr_combined_input_ids_list = []
    start_index = 0  # Initialize the start index

    # for i in enumerate(tokenized_inputs["input_ids"]):
    for i, input_ids in enumerate(tokenized_inputs["input_ids"]):
        if not curr_combined_input_ids_list:
            # If the current combined list is empty, add the inner list
            curr_combined_input_ids_list = input_ids
        elif len(curr_combined_input_ids_list) + len(input_ids) <= max_length:
            # If adding the inner list does not exceed the threshold, combine them
            curr_combined_input_ids_list += input_ids
        else:
            # If adding the inner list exceeds the threshold, start a new combined list
            end_index = i - 1  # Calculate the end index of the previous combined list

            combined_lists_dict["input_ids"].append(curr_combined_input_ids_list)         
            # combined_lists_dict["word_ids"].append(list(chain(*tokenized_inputs["word_ids"][start_index:end_index+1])))
            combined_lists_dict["labels"].append(list(chain(*tokenized_inputs["labels"][start_index:end_index+1])))
            combined_lists_dict["attention_mask"].append(list(chain(*tokenized_inputs["attention_mask"][start_index:end_index+1])))
            combined_original.append(" ".join(original[start_index:end_index+1]))

            curr_combined_input_ids_list = input_ids
            start_index = i  # Update the start index

    # Add the last combined list (if any)
    if curr_combined_input_ids_list:
        end_index = len(tokenized_inputs["input_ids"]) - 1  # Calculate the end index for the last combined list

        combined_lists_dict["input_ids"].append(curr_combined_input_ids_list)         
        # combined_lists_dict["word_ids"].append(list(chain(*tokenized_inputs["word_ids"][start_index:end_index+1])))
        combined_lists_dict["labels"].append(list(chain(*tokenized_inputs["labels"][start_index:end_index+1])))
        combined_lists_dict["attention_mask"].append(list(chain(*tokenized_inputs["attention_mask"][start_index:end_index+1])))
        combined_original.append(" ".join(original[start_index:end_index+1]))

    # assert len(combined_original) == len(tokenized_inputs["input_ids"]), print(len(combined_original), len(tokenized_inputs["input_ids"]))
    for key in tokenized_inputs.keys():
        assert len(tokenized_inputs[key]) == len(tokenized_inputs["input_ids"]), f"problem with {key}"
    return combined_lists_dict, combined_original

def _tokens_labels_to_sent(input_ids: list[int], attention_mask, word_ids, labels, tokenizer, id2label):
    labels = np.array(labels)
    input_ids = np.array(input_ids)
    word_ids = np.array(word_ids)
    assert np.all(attention_mask), "Not all elements are True"
    # attention_mask = np.array(attention_mask, dtype=bool)
    # input_ids = input_ids[attention_mask]
    # word_ids = word_ids[attention_mask]

    labels = labels[labels!=-100]
    # if len([id for id in word_ids if id is None]) != 0:
    #     print(word_ids)
    #     print(attention_mask)
    #     print(labels)
    #     print(tokenizer.decode(input_ids))
    #     exit()
    assert len(labels) == np.max([id for id in word_ids]) + 1

    decoded_text = ""
    for i in range(len(labels)):
        word_input_ids = input_ids[word_ids == i]
        decoded_text += tokenizer.decode(word_input_ids)
        label = id2label[labels[i]]
        if label != "O":
            decoded_text += " " + label
        decoded_text += " "
    return decoded_text

def _test_by_reconstruct(original, tokenized_inputs, tokenizer, id2label):
    for key in tokenized_inputs.keys():
        assert len(original) == len(tokenized_inputs[key])

    reconstructed = []
    for input_ids, attention_mask, word_ids, labels in zip(tokenized_inputs["input_ids"],
                                                                         tokenized_inputs["attention_mask"],
                                                                         tokenized_inputs["word_ids"],
                                                                         tokenized_inputs["labels"]):
        sent: str = _tokens_labels_to_sent(input_ids, attention_mask, word_ids, labels, tokenizer, id2label)
        reconstructed.append(sent)
    return reconstructed


def preprocess(config, list_file_names: List[str], label2id, tokenizer, id2label) -> None:
    """
    The main preprocessing function which will call all necessary functions to produce a data set ready
    to be consumed by the training script
    """
    global_counter = Counter()
    for file in list_file_names:
        src_file_path = os.path.join(config["preprocess"]["input_path"], file)
        dst_file_path = os.path.join(config["preprocess"]["output_path"], file)

        with open(src_file_path, "r") as in_file:
            lines = in_file.readlines()
            results = []
            # divide the files by sentences ending in full stop
            for line in lines:
                # results += [l.strip()+".\n" for l in line.strip().split(".") if len(l.strip())>0]
                results += [l.strip()+".\n" for l in line.strip().split(".") 
                            if len(_remove_punc([l.strip()], config["marks"])[0])>0]

            # tokenize
            tokenized_inputs = _tokenize_and_align_labels(results, tokenizer, label2id, config)
            reconstructed = _test_by_reconstruct(results, tokenized_inputs, tokenizer, id2label)

            tokenized_inputs, results = _group_samples(tokenized_inputs, results, config)

            print("second tokenization")
            tokenized_inputs = _tokenize_and_align_labels(results, tokenizer, label2id, config)
            reconstructed = _test_by_reconstruct(results, tokenized_inputs, tokenizer, id2label)


        
  
        
        # with open(dst_file_path, "w") as out_file:
        #     for inner_list in reconstructed:
        #         line = str(inner_list)
        #         out_file.write(f"{line}\n")

        # with open(dst_file_path, "w") as out_file:
        #     for inner_list in tokenized:
        #         line = str(inner_list)
        #         out_file.write(f"{line}\n")
   
def go(config):
    # run = wandb.init(job_type="preprocess_data")
    # run.config.update(config)


    tokenizer = AutoTokenizer.from_pretrained(config["transformers_checkpoint"])
    # data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    id2label = {i: label for i, label in enumerate(config["marks"]+"O")}
    label2id = {v: k for k, v in id2label.items()}
    label2name = {k: config["mark2name"][k] for k in config["marks"]+"O"}


    list_file_names = [file_name.split("/")[-1] for file_name in 
                glob.glob(os.path.join(config["preprocess"]["input_path"], "*.txt"))]




    logger.info("Preprocessing dataset")
    preprocess(config, list_file_names, label2id, tokenizer, id2label)
    logger.info("Preprocessing is done")




   
    

    logger.info(f"Uploading a sample file from {config['preprocess']['output_path']} to Weights & Biases")

    # artifact = wandb.Artifact(
    #     "sample_raw_data.txt",
    #     type="raw data",
    #     description="A sample raw data file which represent one book",
    # )
    # artifact.add_file(os.path.join(config['download']['output_path'], "01_أحكام القرآن لابن العربي.txt"))
    # artifact.add_file(os.path.join(config['preprocess']['output_path'], "book.txt"))

    # run.log_artifact(artifact)

if __name__ == "__main__":
    pass
    # go(config)