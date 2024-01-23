import random
from datasets import load_dataset, concatenate_datasets, arrow_dataset, Dataset, DatasetInfo
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

class Preprocessor:
    def __init__(self, config, tokenizer, label2id, id2label):
        self.config = config
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label
    

    def _create_labels(self, examples: List[str]) -> List[List[str]]:
        """
        Create labels at the word level (words are whatever separated by space), 
        where each word will have a label indicating the punctuation mark that 
        follows it, if any, and O if none.
        """
        marks = self.config["marks"]
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
                    temp.append(self.label2id[last_char])
                elif idx < len(words) -1:
                    # not the last word in this ex
                    if words[idx+1][0] in marks:
                        # punct is attached to the next word or is the next word (isolated punc)
                        temp.append(self.label2id[words[idx+1][0]])
                    else:
                        temp.append(self.label2id["O"])
                else:
                    temp.append(self.label2id["O"])
            
            labels.append(temp)
        return labels

    def _remove_punc(self, examples: List[str]) -> List[str]:
        table = str.maketrans("", "", self.config["marks"])
        new_examples = []
        for ex in examples:
            new_examples.append(ex.translate(table))
        return new_examples

    def align_labels_with_tokens(self, labels:List[str], word_ids: List[Union[int, str]]) -> List[int] :
        # TODO maybe the label should be assigned to the last token instead of first
        """
        Make label list aligned with the tokens list. 
        Special tokens get a label of -100, to be ignored by the loss function.
        A word is divided into multiple tokens, only the first token gets the label of the word, 
        and -100 is assigned to the other subtokens. 
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

    def _tokenize_and_align_labels(self, examples: list[str], add_special_tokens=False, truncation=False):
        """ returns transformers.tokenization_utils_base.BatchEncoding (a dict subclass)  
        """
        all_labels: List[List[int]] = self._create_labels(examples)

        # remove marks from text before tokenization
        examples = self._remove_punc(examples)

        # tokenized_inputs is transformers.tokenization_utils_base.BatchEncoding  (a dict subclass) 
        # with keys dict_keys(['input_ids', 'attention_mask'])
        # the value of each key in tokenized_inputs is a List[List[int]] where each sub list represent one point in the dataset 
        tokenized_inputs = self.tokenizer(examples, add_special_tokens=add_special_tokens, truncation=truncation, max_length=self.config["tokens_max_len"])
        
        # align labels with tokens
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            # align_labels_with_tokens will work whether special tokens are added or not
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        tokenized_inputs["word_ids"] = [tokenized_inputs.word_ids(i) for i in range(len(tokenized_inputs["input_ids"]))]
        return tokenized_inputs

    def _group_samples(self, tokenized_inputs, org_text: list[str]):
        """Combine multiple consective sample as long as their combined length is below max_length
        """
        max_length = self.config["tokens_max_len"] - 2 # -2 to account for added special tokens later
        combined_input_ids = []
        combined_original = []
        curr_combined_input_ids = []
        start_index = 0

        for i, input_ids in enumerate(tokenized_inputs["input_ids"]):
            if not curr_combined_input_ids:
                # If the current combined list is empty, add the inner list
                curr_combined_input_ids = input_ids
            elif len(curr_combined_input_ids) + len(input_ids) <= max_length:
                # If adding the inner list does not exceed the threshold, combine them
                curr_combined_input_ids += input_ids
            else:
                # If adding the inner list exceeds the threshold, start a new combined list
                end_index = i - 1  # Calculate the end index of the previous combined list

                combined_input_ids.append(curr_combined_input_ids)         
                combined_original.append(" ".join(org_text[start_index:end_index+1]))

                curr_combined_input_ids = input_ids
                start_index = i  # Update the start index

        # Add the last combined list (if any)
        if curr_combined_input_ids:
            end_index = len(tokenized_inputs) - 1  # Calculate the end index for the last combined list

            combined_input_ids.append(curr_combined_input_ids)         
            combined_original.append(" ".join(org_text[start_index:end_index+1]))

        return combined_input_ids, combined_original

    def _tokens_labels_to_sent(self, original, input_ids: list[int], attention_mask: list[int], word_ids: list[int], labels: list[int]) -> str:
        """Convert tokens and labels to a full sentence (single sample conversion)       
        """
        labels = np.array(labels)
        input_ids = np.array(input_ids)
        word_ids = np.array(word_ids)
        attention_mask = np.array(attention_mask, dtype=bool)

        input_ids = input_ids[attention_mask]
        word_ids = word_ids[attention_mask]
        labels = labels[labels!=-100]
        # check that the number of labels equals to number of words
        assert len(labels) == np.max([id for id in word_ids[word_ids != None]]) + 1, "Number of labels does NOT equal number of words"

        decoded_text = ""
        for i in range(len(labels)):
            word_input_ids = input_ids[word_ids == i]
            decoded_text += self.tokenizer.decode(word_input_ids)
            label = self.id2label[labels[i]]
            if label != "O":
                decoded_text += " " + label
            decoded_text += " "
        return decoded_text

    def _test_by_reconstruct(self, original, tokenized_inputs):
        for key in tokenized_inputs.keys():
            assert len(original) == len(tokenized_inputs[key]), "number of samples are NOT equal"

        reconstructed = []
        for idx, (input_ids, attention_mask, word_ids, labels) in enumerate(zip(tokenized_inputs["input_ids"],
                                                                tokenized_inputs["attention_mask"],
                                                                tokenized_inputs["word_ids"],
                                                                tokenized_inputs["labels"])):
 
            sent: str = self._tokens_labels_to_sent(original[idx],input_ids, attention_mask, word_ids, labels)
            reconstructed.append(sent)
        return reconstructed

    def preprocess(self, list_file_names: List[str]) -> (list, list):
        """
        The main preprocessing method which will call all necessary methods to produce a data set ready
        to be consumed by the training script.
        """
        random_samples = []
        stats = []
        random.seed(1)

        for file in list_file_names:
            src_file_path = os.path.join(self.config["preprocess"]["input_path"], file)
            dst_file_path = os.path.join(self.config["preprocess"]["output_path"], file)

            with open(src_file_path, "r") as in_file:
                lines = in_file.readlines()
                org_text = []
                # divide the files by sentences ending in full stop
                for line in lines:
                    # results += [l.strip()+".\n" for l in line.strip().split(".") if len(l.strip())>0]
                    org_text += [l.strip()+"۔\n" for l in line.strip().split(".") 
                                if len(self._remove_punc([l.strip()])[0])>0]

                # tokenize
                tokenized_inputs = self._tokenize_and_align_labels(org_text, add_special_tokens=False, truncation=True)
                self._test_by_reconstruct(org_text, tokenized_inputs)
                # _group_samples will not work properly when add_special_tokens is true above
                _, org_text = self._group_samples(tokenized_inputs, org_text)
                org_text = [s for s in org_text if len(s) > 0]
                # the tokenized_inputs needs to be with special tokens and all to avoid having to modify the labels later
                tokenized_inputs = self._tokenize_and_align_labels(org_text, add_special_tokens=True, truncation=True)
                reconstructed = self._test_by_reconstruct(org_text, tokenized_inputs)

            # save preprocessed book to disk
            Dataset.from_dict(tokenized_inputs).save_to_disk(dst_file_path[:-4]) #:-4 remove .txt 

            # prepare stats for logging
            idx = random.randint(0, len(org_text)-1)
            sample = [
                file, org_text[idx], reconstructed[idx]
            ]
            book_count = count_labels(tokenized_inputs["labels"])
            book_perc = calculate_percentage(book_count)
            book_stat = [
                file,
                *[book_count.get(key, 0) for key in sorted(self.id2label.keys())], 
                *[f"{book_perc.get(key, 0):.1f}" for key in sorted(self.id2label.keys())]
            ]
            random_samples.append(sample)
            stats.append(book_stat)
            
        return random_samples, stats
    

def go(config):
    run = wandb.init(job_type="preprocess_data")
    run.config.update(config)

    tokenizer = AutoTokenizer.from_pretrained(config["transformers_checkpoint"])
    id2label = {i: label for i, label in enumerate(config["marks"]+"O")}
    label2id = {v: k for k, v in id2label.items()}
    
    preprocessor = Preprocessor(config, tokenizer, label2id, id2label)

    list_file_names = [file_name.split("/")[-1] for file_name in 
                glob.glob(os.path.join(config["preprocess"]["input_path"], "*.txt"))]

    logger.info("Preprocessing dataset")
    random_samples, stats = preprocessor.preprocess(list_file_names)
    logger.info("Preprocessing is done")

    sample_table = wandb.Table(data=random_samples, columns=["Book", "Original", "Reconstructed"])
    run.log({"Sample Data": sample_table})

    # create another table for label statistics
    stats_table = wandb.Table(data=stats, columns=["Book",
                                                   *[f"count({id2label[key]})" for key in sorted(id2label.keys())],
                                                   *[id2label[key] for key in sorted(id2label.keys())]
                                                    ])
    run.log({"Statistics": stats_table})


def count_labels(labels:list[list[int]]):
    # Flatten the list of lists into a single list
    flattened_list = [item for sublist in labels for item in sublist]
    # Use Counter to count the occurrences of each unique label
    counts = Counter(flattened_list)
    return counts

def calculate_percentage(counter):
    total_counts = sum(counter.values())
    
    if total_counts == 0:
        # Avoid division by zero if the counter is empty
        return dict.fromkeys(counter, 0.0)

    percentage_dict = {key: count / total_counts * 100 for key, count in counter.items()}
    return percentage_dict

if __name__ == "__main__":
    pass
    # go(config)