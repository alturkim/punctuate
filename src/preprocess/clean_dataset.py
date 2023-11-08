from typing import List
import re
import string
import logging
import os
import glob

logger = logging.getLogger(__name__)

# extra marks: marks that are not considered for prediction
extra_marks = string.punctuation.translate(str.maketrans("", "", config.marks))
# replace each mark in the inspection list by the corresponding mark in the replacement_list
inspection_list = ["'", ',', ';', '«', '»', '“', '﴾', '﴿']
replacement_list = ["", '،', '؛', '"', '"', '"', '"', '"']
mapping_marks = {k:v for k, v in zip(inspection_list, replacement_list)}

def _remove_diacritics(examples: List[str]) -> List[str]:
    diact = re.compile(" ّ |  َ |  ً |  ُ |   ٌ |   ِ |   ٍ |   ْ  |  ـ   ", re.VERBOSE)
    new_examples = []
    for ex in examples:
        new_examples.append(re.sub(diact, '', ex))
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

def clean(list_file_names: List[str], src_data_home="../../data/raw", dst_data_home="../data/clean") -> None:
    for file in list_file_names:
        src_file_path = os.path.join(src_data_home, file)
        dst_file_path = os.path.join(dst_data_home, file)
        with open(src_file_path, "r") as in_file:
            lines = in_file.readlines()

            lines = _unify_punc_marks(lines, mapping_marks)
            lines = _remove_diacritics(lines)
            lines = _remove_non_letters(lines, config.marks)
        with open(dst_file_path, "w") as out_file:
            out_file.writelines(lines)


if __name__ == "__main__":
    logger.info("Cleaning dataset ...")
    list_file_names = [file_name.split("/")[-1] for file_name in glob.glob("../../data/raw/*.txt")[10:20]]
    # clean(list_file_names)