import json

from typing import Any, Dict, List, Optional, Set, Tuple, Union

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


# for batch tensors:
def cosine_similarity(tensor1: Tensor, tensor2: Tensor, eps: float = 1e-8):
    tensor1_norm = tensor1.norm(dim=1).unsqueeze(1)
    tensor2_norm = tensor2.norm(dim=1).unsqueeze(1)
    tensor1_normed = tensor1 / torch.max(tensor1_norm, eps * torch.ones_like(tensor1_norm))
    tensor2_normed = tensor2 / torch.max(tensor2_norm, eps * torch.ones_like(tensor2_norm))
    return torch.mm(tensor1_normed, tensor2_normed.T)

def get_bertscore_between_sent(sentence1: str, sentence2: str):
    P, R, F1 = bert_score.score([sentence1],[sentence2], lang='en', verbose=True)
    print('finished')

