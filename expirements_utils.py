import json
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
import bert_score
from torch import Tensor, logit
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import random
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
    masked_tokenid=tokenizer.convert_tokens_to_ids(sentence[location])
    sentence[location]=tokenizer.mask_token
    return sentence,masked_tokenid

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


def get_bertscore_between_sents(sentence1: str, sentence2: str):
    P, R, F1 = bert_score.score(sentence1,sentence2, lang='en', verbose=True)
    return P,R,F1

def get_similarity_between_two_states(states1:Tensor,states2:Tensor):
    cos_s=torch.nn.CosineSimilarity(dim=0)
    sims=[]
    for i,layer1 in enumerate(states1):
        s1=states1[i][0,:,:]
        s2=states2[i][0,:,:]
        res=cos_s(s1,s2)
        sims.append(np.mean(res.cpu().numpy()))
    return tuple(sims)

# checked for correctf for a single sentence:
def check_if_predicted_correct(logits:Tensor,mask_index:int,masked_tokenid:int):
    predicted_token_id=logits[:,mask_index].argmax(dim=-1)
    return (predicted_token_id == masked_tokenid).item()



def convert_k_to_pair(k):
    q = np.floor(np.sqrt(8*(k-1)+1)/2 +1.5)
    p = k - (q-1)*(q-2)/2
    return q,p
def generate_k_unique_pairs(n:int,k:int):
    num_of_pairs = (n/2)*(n-1) # = n*(n-1)/2 number of pairs from 0 to n
    assert (k < num_of_pairs)
    pairs_indexes = random.sample(range(1,int(num_of_pairs)),k)
    return pairs_indexes,k



    
if __name__ == "__main__":
    #test:
    a,k=generate_k_unique_pairs(4,5)
    for i in a:
        print(convert_k_to_pair(i))