from datasets import load_dataset, concatenate_datasets, arrow_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from pprint import pprint
import numpy as np
from typing import List, Union
import collections
import re
import string
import logging

from utils import config

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained(config.transformers_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
id2label = {i: label for i, label in enumerate(config.marks+"O")}
label2id = {v: k for k, v in id2label.items()}
label2name = {k: config.mark2name[k] for k in config.marks+"O"}
# extra marks: marks that are not considered for prediction
extra_marks = string.punctuation.translate(str.maketrans("", "", config.marks))
# replace each mark in the inspection list by the corresponding mark in the replacement_list
inspection_list = ["'", ',', ';', '«', '»', '“', '﴾', '﴿']
replacement_list = ["", '،', '؛', '"', '"', '"', '"', '"']
mapping_marks = {k:v for k, v in zip(inspection_list, replacement_list)}

def get_raw_datasets() -> arrow_dataset.Dataset:
    # columns: book, text where book is the title and text is the content as a list of one string 
    logger.info("Downloading dataset")
    raw_datasets = load_dataset("tashkeela", split="train", download_mode="force_redownload")
    logger.info("done downloading")
    
    sub_datasets = []
    logger.info("Spliting book content into multiple strings")
    for d in raw_datasets:
        sub_datasets.append(_split_text(d))
    raw_datasets = concatenate_datasets(sub_datasets)
    # after spliting: columns: book, text where book is the title and text is the part of the content as a list of one string 
    # each book was one element in the dataset, now each book is represented by multiple elements, each one with same book value
    # but text value will be a list of one string representing part of the content
    return raw_datasets

def _split_text(examples: arrow_dataset.Dataset) -> arrow_dataset.Dataset:
    """
    Receives a Dataset object where columns are book and text, book is the book title and text is the content as a list of one string.
    It splits the content into a list of multiple strings each of size config.text_chunk_size.
    """
    result = {"book": [], "text": []}
    words =  examples["text"].split()
    sentences = [" ".join(words[i : i + config.text_chunk_size]) for i in range(0, len(words), config.text_chunk_size)]
    for sent in sentences:
        result["book"].append(examples["book"])
        result["text"].append(sent)
    examples = Dataset.from_dict(result)
    return examples

def _create_labels(examples: List[str], marks: str) -> List[List[str]]:
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

def _remove_diacritics(examples: List[str]) -> List[str]:
    diact = re.compile(" ّ |  َ |  ً |  ُ |   ٌ |   ِ |   ٍ |   ْ  |  ـ   ", re.VERBOSE)
    new_examples = []
    for ex in examples:
        new_examples.append(re.sub(diact, '', ex))
    return new_examples

def _remove_punc(examples: List[str], marks: str) -> List[str]:
    table = str.maketrans("", "", marks)
    new_examples = []
    for ex in examples:
        new_examples.append(ex.translate(table))
    return new_examples

def _remove_non_letters(examples: List[str], marks: str) -> List[str]:
    """
    Removes any non-arabic letters (keep punctuations)
    """
    p = re.compile(f"[^\u0600-\u06FF|{marks}|\s]")
    new_examples = []
    for ex in examples:
        new_examples.append(re.sub(p, '', ex))
    return new_examples

def _unify_punc_marks(examples: List[str], mapping_marks: dict) -> List[str]:
    """
    Reduce mutliple versions of the same mark to a unified mark
    """
    new_examples = []
    for ex in examples:
        new_examples.append(ex.translate(str.maketrans(mapping_marks)))
    return new_examples

def align_labels_with_tokens(labels:List[str], word_ids: List[Union[int, str]], input_ids: List[int]) -> List[int] :
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

def _tokenize_and_align_labels(examples: arrow_dataset.Dataset): # returns transformers.tokenization_utils_base.BatchEncoding (a dict subclass)
    # examples["text"] is a list of strings 
    
    examples["text"] = _unify_punc_marks(examples["text"], mapping_marks)
    # remove non letter characters except marks to be predicted
    examples["text"] = _remove_diacritics(examples["text"])
    examples["text"] = _remove_non_letters(examples["text"], config.marks)
    
    all_labels: List[List[int]] = _create_labels(examples["text"], config.marks)

    # remove marks from text before tokenization
    examples["text"] = _remove_punc(examples["text"], config.marks)
    tokenized_inputs = tokenizer(examples["text"]) # tokenized_inputs is transformers.tokenization_utils_base.BatchEncoding  (a dict subclass) with keys dict_keys(['input_ids', 'attention_mask'])
    # the value of each key in tokenized_inputs is a List[List[int]] where each sub list represent one point in the dataset 
    
    # align labels with tokens
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids, tokenized_inputs.input_ids[i]))

    tokenized_inputs["labels"] = new_labels
    tokenized_inputs["word_ids"] = [tokenized_inputs.word_ids(i) for i in range(len(tokenized_inputs["input_ids"]))]
    return tokenized_inputs


def _group_and_split_samples(examples) -> dict:
    """
    Create a dictionary where values are list of tokens of same size.
    """
    # Concatenate all texts 
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[-1]])
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
    """
    The main preprocessing function which will call all necessary functions to produce a data set ready to be consumed by the training script
    """
    result = dataset.map(
        _tokenize_and_align_labels,
        batched=True,
        remove_columns= ["book", "text"],
        
    )
    result = result.map(_group_and_split_samples, batched=True)
    return result

if __name__ == "__main__":
    logger.info("Preparing dataset ...")
    save_dataset_path = config.transformers_checkpoint.replace('/','_').replace('-','_')
    raw_datasets = get_raw_datasets()
    punc_datasets = preprocess(raw_datasets)
    punc_datasets.save_to_disk(f"data/{save_dataset_path}")