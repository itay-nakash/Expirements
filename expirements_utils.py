import json
import itertools
from calendar import c
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import bert_score
from torch import Tensor, logit
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForMaskedLM.from_pretrained("roberta-base")


def write_dict_to_json(dict,fname):
    file_path='data/'+fname
    with open(file_path) as outfile:
        json.dump(dict, outfile)

def read_dict_from_json(fname):
    file_path='data/'+fname
    with open(file_path) as json_file:
        return json.load(json_file)

def mask_word(sentence,location):
    sentence[location]=tokenizer.mask_token
    return sentence

#TODO: add desc of what it returns:
def convert_sentences_list_to_model_input(senteneces: List[List[str]]):
    # convert the sentences list to input_ids:
    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in senteneces]
    #add padding:
    input_ids_padded = tokenizer(tokenizer.batch_decode(input_ids), add_special_tokens=False, padding=True, return_tensors="pt")
    return input_ids_padded

