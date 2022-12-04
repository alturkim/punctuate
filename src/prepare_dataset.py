from datasets import load_dataset, concatenate_datasets, arrow_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from pprint import pprint
import numpy as np
from typing import List, Union
import collections
import re
import string

from . import config

tokenizer = AutoTokenizer.from_pretrained(config.transformers_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
id2label = {i: label for i, label in enumerate(config.marks+"O")}
label2id = {v: k for k, v in id2label.items()}
# extra marks: marks that are not considered for prediction
extra_marks = string.punctuation.translate(str.maketrans("", "", config.marks))

def get_raw_datasets() -> arrow_dataset.Dataset:
    # columns: book, text where book is the title and text is the content as one string
    raw_datasets = load_dataset("tashkeela", split="train") 
    return raw_datasets
    
def create_labels(examples: List[str], marks: str) -> List[List[str]]:
    """
    create labels at the word level (words are whatever separated by space), 
    where each word will have a label indicating the punctuation mark that 
    follows it, if any, and O if none.
    """
    labels = []
    for ex in examples:
        temp = []
        words = ex.split()
        for idx, w in enumerate(words):
            if w in marks:
                continue
            last_char = w[-1]
            if last_char in marks:
                temp.append(label2id[last_char])
            elif idx < len(words) -1:
                if words[idx+1] in marks:
                    temp.append(label2id[words[idx+1]])
                elif words[idx+1][0] in marks:
                    # punct is attached to the next word
                    temp.append(label2id[words[idx+1][0]])
                else:
                    temp.append(label2id["O"])
            else:
              temp.append(label2id["O"])
          
        labels.append(temp)
    return labels

def remove_diacritics(examples: List[str]) -> List[str]:
    diact = re.compile(" ّ |  َ |  ً |  ُ |   ٌ |   ِ |   ٍ |   ْ  |  ـ   ", re.VERBOSE)
    new_examples = []
    for ex in examples:
        new_examples.append(re.sub(diact, '', ex))
    return new_examples

def remove_punc(examples: List[str], marks: str) -> List[str]:
    table = str.maketrans("", "", marks)
    new_examples = []
    for ex in examples:
        new_examples.append(ex.translate(table))
    return new_examples

def align_labels_with_tokens(labels:List[str], word_ids: List[Union[int, str]]) -> List[int] :
    """
    make label list match the tokens. special tokens get a label of -100. 
    so it will be ignored in the loss function.
    each word gets one label, so -100 is assigned to the other subtokens. 
    This is to avoid long words that split into lots of subtokens contributing 
    heavily to the loss. 
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
    

def tokenize_and_align_labels(examples: arrow_dataset.Dataset): # returns transformers.tokenization_utils_base.BatchEncoding (a dict subclass)
    examples["text"] = remove_punc(remove_diacritics(examples["text"]), extra_marks)
    all_labels = create_labels(examples["text"], config.marks)
    tokenized_inputs = tokenizer(remove_punc(examples["text"], config.marks))
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    tokenized_inputs["word_ids"] = [tokenized_inputs.word_ids(i) for i in range(len(tokenized_inputs["input_ids"]))]
    print("tokenize_and_align_labels type of tokenized_input", type(tokenized_inputs))
    return tokenized_inputs

def group_and_split_texts(examples: arrow_dataset.Batch) -> dict:
    """
    split book text (one string) into chunks
    """
    # Concatenate all texts 
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // config.chunk_size) * config.chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + config.chunk_size] for i in range(0, total_length, config.chunk_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def add_punct(decoded_str: str, word_ids: list, labels:list) -> str:
    output = ""
    words = decoded_str.split()
    unique_word_ids = np.unique(word_ids)
    labels = np.array(labels)
    word_ids = np.array(word_ids)

    assert len(words) == unique_word_ids.size
    for word_id, word in zip(unique_word_ids, words):
        punc_marks: np.ndarray = labels[word_ids==word_id]
        # TODO need better aggregation later
        if punc_marks[0] != -100 and id2label[punc_marks[0]] != "O":
            output_punc_mark = id2label[punc_marks[0]]
        else:
            output_punc_mark = ""
        output += word + output_punc_mark + " "
    return output

def preprocess(dataset: arrow_dataset.Dataset) -> arrow_dataset.Dataset:
    result = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns= dataset.column_names
    )
    result = result.map(group_and_split_texts, batched=True)
    result = result.remove_columns("word_ids")
    return result
