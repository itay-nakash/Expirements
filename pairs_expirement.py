import itertools
from calendar import c
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import bert_score
from torch import Tensor, logit
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
from expirements_utils import tokenizer, model, convert_sentences_list_to_model_input

from expirements_utils import read_dict_from_json,mask_word,cosine_similarity

MIN_NUM_OF_SEN = 10
STATES_NUM= 13


class SameIndexExpi:
    # states[0].shpae = (batch,tokens,hidden_d)
    # returns states of only correct and only incorrect predicted by model
    def seperate_correct_incorrect_predict(states: Tensor, logits: Tensor,mask_index:int, masked_tokenid: float) -> Tuple[Tensor]:
        # get to token_id of the masked token:
        # get the predicted token_id of each sen:
        arg_max_tensor=logits[:,mask_index].argmax(dim=-1)
        correct_mask = arg_max_tensor==masked_tokenid
        incorrect_mask=arg_max_tensor!=masked_tokenid
        states_correct= tuple([state[correct_mask] for state in states])
        states_incorrect= tuple([state[incorrect_mask] for state in states])
        logits_correct= logits[correct_mask]
        # returns the logits on all the tokens, just on sentences predicted incorrect
        logits_incorrect_all= logits[incorrect_mask]
        # returns the values of logits on the (wrong token) in sentences that was predicted incorrect
        predicted_tokens_in_incorrect = arg_max_tensor[incorrect_mask]
        logits_of_predicted_token_incorrect= logits_incorrect_all[:,mask_index,predicted_tokens_in_incorrect]
        # TODO: find an efficieent way todo so:
        # keep just the wanted indeses, so at each entry - tell it to keep just the index of the entry:
        #logits_of_predicted_token_incorrect = logits_of_predicted_token_incorrect[:,torch.arange(0,predicted_tokens_in_incorrect.shape[0],1)]
        logits_of_predicted_token_incorrect = torch.FloatTensor([logits_of_predicted_token_incorrect[x,x] for x in range(logits_of_predicted_token_incorrect.shape[0])])

        return states_correct, states_incorrect,logits_correct,logits_incorrect_all,logits_of_predicted_token_incorrect

    def find_batch_similarities(senteneces: List[List[str]],mask_loc: int, masked_word: str):
        # convert the token str to token_id:
        masked_tokenid=tokenizer.convert_tokens_to_ids(masked_word)
        
        input_ids_padded = convert_sentences_list_to_model_input(senteneces)

        with torch.no_grad():
            outputs = model(**input_ids_padded, output_hidden_states=True)
        # states[0].shape = (batch,tokens,hidden_d) - dim of each state, we have 13 states as number of layers
        states = outputs['hidden_states']
        # logits.shpae = (batch,tokens,voc_size)
        logits = outputs['logits']
        corrent_states, incorrect_states,logits_c,\
        logits_nc_org_tok,logits_nc_for_pred_tok = SameIndexExpi.seperate_correct_incorrect_predict(states,logits,mask_loc,masked_tokenid)

        # calculate softmax:
        softmax_logits_c=torch.softmax(logits_c,dim=-1)
        softmax_logits_nc=torch.softmax(logits_nc_org_tok,dim=-1)
        softmax_logits_nc_for_pred_tok=torch.softmax(logits_nc_for_pred_tok,dim=-1)
        
        # calculate means:
        softmax_logits_c_mean=softmax_logits_c[:,mask_loc,masked_tokenid].mean().item()
        softmax_logits_nc_mean=softmax_logits_nc[:,mask_loc,masked_tokenid].mean().item()
        softmax_logits_nc_for_pred_tok_mean = softmax_logits_nc_for_pred_tok.mean().item()

        correct_mean_sims, correct_std_sims,correct_num_of_examples =SameIndexExpi.calculte_stats_on_states(corrent_states,mask_loc)
        incorrect_mean_sims, incorrect_std_sims,incorrect_num_of_examples=SameIndexExpi.calculte_stats_on_states(incorrect_states,mask_loc)


        return correct_mean_sims,incorrect_mean_sims,correct_std_sims,incorrect_std_sims,\
            correct_num_of_examples,incorrect_num_of_examples,softmax_logits_c_mean,\
                softmax_logits_nc_mean,softmax_logits_nc_for_pred_tok_mean

    def calculte_stats_on_states(states: Tensor,mask_loc =-1,mask_loc_list1=[],mask_loc_list2=[]) -> Tuple[Tensor,Tensor,int]:
        mean_sims = []
        std_sims=[]
        num_of_examples = 0
        for state in states:
            num_of_examples = state.shape[0]
            # state.shape = (batch,hidden_d)
            if mask_loc != -1: # we in the 'same-index-scenario' 
                state1 = state[:,mask_loc,:]
                state2 = state[:,mask_loc,:]
            else:
                state1 = state[:,mask_loc_list1,:]
                state2 = state[:,mask_loc_list2,:]
            # sims.shpae = (batch,batch)
            sims = cosine_similarity(state1, state2)
            sims = sims.cpu().numpy()
            sims = sims[np.triu_indices(sims.shape[0],k=1)]
            mean_sims.append(np.mean(sims))
            std_sims.append(np.std(sims))


        return mean_sims,std_sims,num_of_examples

    def create_table_from_results(words_dict:Dict ,sentences_list:List):
        results = []
        for word in words_dict:
            for mask_loc in words_dict[word]:
                current_sentences=[mask_word(sentences_list[i],int(mask_loc)) for i in words_dict[word][mask_loc]]
                sims_c,sims_nc, stds_c,stds_nc ,num_of_examples_c,\
                num_of_examples_nc,softmax_logits_c_mean,\
                softmax_logits_nc_mean,softmax_logits_nc_for_pred_tok = SameIndexExpi.find_batch_similarities(current_sentences,int(mask_loc),word)

                if num_of_examples_c < MIN_NUM_OF_SEN and num_of_examples_nc < MIN_NUM_OF_SEN:
                    continue
                print(f'new func: word: {word}, index: {mask_loc}, examples_c: {num_of_examples_c},examples_nc:{num_of_examples_nc}')
                for layer, (sim_c, std_c,sim_nc, std_nc) in enumerate(zip(sims_c, stds_c,sims_nc, stds_nc)):
                    results.append({'word': word, 'index': mask_loc, 'layer': layer, 'examples': num_of_examples_c, 'similarity': sim_c,'mean softmax masked_token':softmax_logits_c_mean,'mean softmax pred_token': softmax_logits_c_mean, 'std': std_c, 'predicted correct': 1})
                    results.append({'word': word, 'index': mask_loc, 'layer': layer, 'examples': num_of_examples_nc, 'similarity': sim_nc,'mean softmax masked_token':softmax_logits_nc_mean,'mean softmax pred_token': softmax_logits_nc_for_pred_tok, 'std': std_nc,'predicted correct': 0})
        df = pd.DataFrame(results)
        df.to_csv('/home/itay.nakash/projects/smooth_language/results/expirement_result.csv', index=False)    


class DifferentIndexExpi:

    def __init__(self,words_dict, indexes_sets_size:int):
        # the size of each :
        self.indexes_sets_size = indexes_sets_size
        self.words_dict = words_dict

    def choose_subsets(self,indexes_group):
        indexes1=random.sample(indexes_group,self.indexes_sets_size)
        # remove the choosen indexes - to make sure we dont have the same index in the expirement:
        indexes_group_new = [x for x in indexes_group if x not in indexes1]
        indexes2=random.sample(indexes_group_new,self.indexes_sets_size)
        return indexes1, indexes2

    def create_table_from_results(self,sentences_list):
        results = []
        for word in self.words_dict:
            indexes1,indexes2 = self.choose_subsets(self.words_dict[word].keys())
            for mask_loc1 in indexes1:
                current_sentences1=[mask_word(sentences_list[i],int(mask_loc1)) for i in self.words_dict[word][mask_loc1]]
                for mask_loc2 in indexes2:
                    current_sentences2=[mask_word(sentences_list[i],int(mask_loc2)) for i in self.words_dict[word][mask_loc2]]
                    # concate the lists to get all the sentences inputs:
                    all_current_sents = current_sentences1 + current_sentences2
                    SameIndexExpi.find_batch_similarities(all_current_sents,int(mask_loc1),int(mask_loc2),word)
        df = pd.DataFrame(results)
        df.to_csv('/home/itay.nakash/projects/smooth_language/expirement_result.csv', index=False)    



if __name__ == "__main__":
    most_freq_words=read_dict_from_json('n_most_frq')
    word_to_sen=read_dict_from_json('word_to_sen_dict')
    sentences_list=read_dict_from_json('sentences_list')
    
    SameIndexExpi.create_table_from_results(word_to_sen,sentences_list)