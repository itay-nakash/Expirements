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
from expirements_utils import tokenizer, model, convert_sentences_list_to_model_input

from expirements_utils import read_dict_from_json,mask_word

STATES_NUM=13


def compare_layers_similarities(senteneces: List[List[str]],mask_loc: int, masked_word: str):
    input_ids_padded = convert_sentences_list_to_model_input(senteneces)
    #layers_sim[i][j] -> the similarity between layers i,j when 
    layers_sim=np.ones((13,13))
    with torch.no_grad():
        outputs = model(**input_ids_padded, output_hidden_states=True)

    # states[0].shape = (batch,tokens,hidden_d) - dim of each state, we have 13 states as number of layers
    states = outputs['hidden_states']
    # logits.shpae = (batch,tokens,voc_size)
    # logits = outputs['logits']
    cos_s=torch.nn.CosineSimilarity(dim=0)
    for i,state1 in enumerate(states):
        for j,state2 in enumerate(states):
            # run only on pairs that j<i - so we run once on each:
            if j<i:
                i_j_means=np.empty(state1.shape[0]) 
                for k in range(state1.shape[0]):
                    sim = cos_s(state1[k],state2[k])
                    sim = sim.cpu().numpy()
                    # mean over all the hidden-states dimentions:
                    mean_sim = np.mean(sim)
                    i_j_means[k]=mean_sim
                # mean over all the k's sentences
                layers_sim[i][j]=np.mean(i_j_means)
                # mul by k to save the number of examples:


    return layers_sim

def compare_layers(words_dict,sentences_list,num_of_iter=-1):
    layers_sim=np.zeros((STATES_NUM,STATES_NUM))
    num_of_sentences=0
    for word in words_dict:
        for mask_loc in words_dict[word]:
            current_sentences=[mask_word(sentences_list[i],int(mask_loc)) for i in words_dict[word][mask_loc]]
            num_of_sentences+=len(current_sentences)
            layers_sim+=len(current_sentences)*compare_layers_similarities(current_sentences,mask_loc,word)
            num_of_iter-=1
            if num_of_iter == 0:
                layers_sim=layers_sim/num_of_sentences
                return layers_sim
    layers_sim=layers_sim/num_of_sentences
    return layers_sim

def write_dict_to_excel(layers_sims ,states_num:int):
    results=[]
    for i in range(states_num):
        for j in range(states_num):
            if j<i:
                results.append({f'layers':f'({i},{j})',f'sim':layers_sims[i][j]})
    df = pd.DataFrame(results)
    df.to_csv('/home/itay.nakash/projects/smooth_language/layers_exp_result.csv', index=False)    


if __name__ == "__main__":
    word_to_sen=read_dict_from_json('word_to_sen_dict')
    sentences_list=read_dict_from_json('sentences_list')
    
    #create_table_from_results(word_to_sen,sentences_list)
    layers_sims=compare_layers(word_to_sen,sentences_list,5000)
    write_dict_to_excel(layers_sims,STATES_NUM)