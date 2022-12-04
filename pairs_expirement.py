import itertools
import sys
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
        logits_c= logits[correct_mask]
        # returns the logits on all the tokens, just on sentences predicted incorrect
        logits_nc_all= logits[incorrect_mask]
        # returns the values of logits on the (wrong token) in sentences that was predicted incorrect
        predicted_tokens_in_incorrect = arg_max_tensor[incorrect_mask]
        logits_nc_for_pred_tok = torch.FloatTensor([logits_nc_all[i,mask_index,x] for i,x in enumerate(predicted_tokens_in_incorrect)])
        # TODO: find an efficieent way todo so:
        #logits_of_predicted_token_incorrect = logits_nc_all[:,mask_index,predicted_tokens_in_incorrect]
        # keep just the wanted indeses, so at each entry - tell it to keep just the index of the entry:
        #logits_of_predicted_token_incorrect = logits_of_predicted_token_incorrect[:,torch.arange(0,predicted_tokens_in_incorrect.shape[0],1)]
        # for debuging, removed it since it is run-time expensive:
        #for j in range(logits_nc_all.shape[0]):
        #    assert logits_nc_all[j,mask_index,masked_tokenid]<=logits_of_predicted_token_incorrect[j]
         # calculate softmax:
        softmax_logits_c=torch.softmax(logits_c,dim=-1)
        softmax_logits_nc_all=torch.softmax(logits_nc_all,dim=-1)
        softmax_logits_nc_masked_tok=torch.softmax(logits_nc_all,dim=-1)[:,mask_index,masked_tokenid]
        softmax_logits_nc_for_pred_tok = torch.FloatTensor([softmax_logits_nc_all[i,mask_index,x] for i,x in enumerate(predicted_tokens_in_incorrect)])


        return states_correct, states_incorrect,logits_c\
            ,logits_nc_all,logits_nc_for_pred_tok,softmax_logits_c,\
                softmax_logits_nc_masked_tok,softmax_logits_nc_for_pred_tok

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
        states_c, states_nc,logits_c,\
        logits_nc_all,logits_nc_for_pred_tok,softmax_logits_c,\
        softmax_logits_nc_masked_tok,softmax_logits_nc_for_pred_tok = SameIndexExpi.seperate_correct_incorrect_predict(states,logits,mask_loc,masked_tokenid)

        # calculate softmax means:
        softmax_c_mean=softmax_logits_c[:,mask_loc,masked_tokenid].mean().item()
        softmax_nc_masked_mean_masked=softmax_logits_nc_masked_tok.mean().item()
        softmax_nc_for_pred_tok_mean = softmax_logits_nc_for_pred_tok.mean().item()

        # calculate logits means:
        logits_c_mean = logits_c[:,mask_loc,masked_tokenid].mean().item()
        logits_nc_masked_tok_mean = logits_nc_all[:,mask_loc,masked_tokenid].mean().item()
        logits_nc_pred_tok_mean = logits_nc_for_pred_tok.mean().item()

        sims_c, stds_c,num_of_examples_c =SameIndexExpi.calculte_stats_on_states(states_c,mask_loc)
        sims_nc, stds_nc,num_of_examples_nc=SameIndexExpi.calculte_stats_on_states(states_nc,mask_loc)       

        return sims_c,sims_nc,stds_c,stds_nc,\
            num_of_examples_c,num_of_examples_nc,softmax_c_mean,\
                softmax_nc_masked_mean_masked,softmax_nc_for_pred_tok_mean,\
                    logits_c_mean,logits_nc_masked_tok_mean,logits_nc_pred_tok_mean

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

    def create_table_from_results(words_dict:Dict ,sentences_list:List,num_of_iter=sys.maxsize):
        results = []
        for word in words_dict:
            num_of_iter-=1
            if num_of_iter==0:
                break
            for mask_loc in words_dict[word]:
                current_sentences=[mask_word(sentences_list[i],int(mask_loc)) for i in words_dict[word][mask_loc]]
                sims_c,sims_nc, stds_c,stds_nc ,num_of_examples_c,\
                num_of_examples_nc,softmax_c_mean,\
                softmax_nc_masked_mean,softmax_nc_for_pred_tok_mean,\
                    logits_c_mean,logits_nc_masked_tok_mean,logits_nc_for_pred_tok_mean = SameIndexExpi.find_batch_similarities(current_sentences,int(mask_loc),word)

                if num_of_examples_c < MIN_NUM_OF_SEN and num_of_examples_nc < MIN_NUM_OF_SEN:
                    continue
                print(f'new func: word: {word}, index: {mask_loc}, examples_c: {num_of_examples_c},examples_nc:{num_of_examples_nc}')
                for layer, (sim_c, std_c,sim_nc, std_nc) in enumerate(zip(sims_c, stds_c,sims_nc, stds_nc)):
                    results.append({'word': word, 'index': mask_loc, 'layer': layer, 'examples': num_of_examples_c, 'similarity': sim_c,\
                        'mean logits masked_token':logits_c_mean,'mean logits pred_token': logits_c_mean,\
                            'mean softmax masked_token':softmax_c_mean,'mean softmax pred_token': softmax_c_mean,\
                                'std': std_c, 'predicted correct': 1})
                    
                    results.append({'word': word, 'index': mask_loc, 'layer': layer, 'examples': num_of_examples_nc, 'similarity': sim_nc,\
                        'mean logits masked_token': logits_nc_masked_tok_mean, 'mean logits pred_token': logits_nc_for_pred_tok_mean,\
                            'mean softmax masked_token':softmax_nc_masked_mean ,'mean softmax pred_token': softmax_nc_for_pred_tok_mean,\
                             'std': std_nc,'predicted correct': 0})
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

    def create_table_from_results(self,sentences_list,max_number_of_examples=sys.maxsize):
        results = []
        for word in self.words_dict:
            if len(results) > max_number_of_examples:
                break
            # not enough indexes for 2 seperate groups:
            if len(self.words_dict[word].keys()) < 2*self.indexes_sets_size:
                continue
            indexes1,indexes2 = self.choose_subsets(self.words_dict[word].keys())
            indexes_lens=[]
            all_current_sents=[]
            for mask_loc in indexes1+indexes2:
                current_sentences=[mask_word(sentences_list[i],int(mask_loc)) for i in self.words_dict[word][mask_loc]]
                indexes_lens.append(len(current_sentences))
                all_current_sents+=current_sentences
                # concate the lists to get all the sentences inputs:
            results += DifferentIndexExpi.find_batch_similarities(all_current_sents,indexes1,indexes2,indexes_lens,word)
        df = pd.DataFrame(results)
        df.to_csv('/home/itay.nakash/projects/smooth_language/results/different_index_exp.csv', index=False)

    
    def split_states_by_indexes(states:Tensor,logits:Tensor,indexes1:List[int],indexes2:List[int],indexes_lens:List[int]):
        n_states_list=[]
        for i,c_i in enumerate(indexes1+indexes2):
            n_states=[]
            for state in states:
                n_states.append(state[0:indexes_lens[i],int(c_i),:])
            n_states_list.append(tuple(n_states))
        return n_states_list


    def compare_states_pairs(states1:Tensor,states2:Tensor):
        mean_sims = []
        std_sims=[]
        num_of_examples = 0
        for i,state1 in enumerate(states1):
            state2=states2[i]
            # compare by the min number of sentences in the pairs:
            num_of_examples = min(state1.shape[0],state2.shape[0])
            if num_of_examples< MIN_NUM_OF_SEN:
                return 0,0,0
            # state.shape = (batch,hidden_d)
            state1 = state1[:num_of_examples,:]
            state2 = state2[:num_of_examples,:]
            # sims.shpae = (batch,batch)
            sims = cosine_similarity(state1, state2)
            sims = sims.cpu().numpy()
            sims = sims[np.triu_indices(sims.shape[0],k=1)]
            mean_sims.append(np.mean(sims))
            std_sims.append(np.std(sims))
        return mean_sims,std_sims,num_of_examples

     
    def find_batch_similarities(senteneces: List[List[str]],indexes1:List[int],indexes2:List[int],indexes_lens:List[int], masked_word: str):
        results=[]
        # convert the token str to token_id:
        masked_tokenid=tokenizer.convert_tokens_to_ids(masked_word)
        
        input_ids_padded = convert_sentences_list_to_model_input(senteneces)

        with torch.no_grad():
            outputs = model(**input_ids_padded, output_hidden_states=True)
        # states[0].shape = (batch,tokens,hidden_d) - dim of each state, we have 13 states as number of layers
        states = outputs['hidden_states']
        # logits.shpae = (batch,tokens,voc_size)
        logits = outputs['logits']
        n_states_list=DifferentIndexExpi.split_states_by_indexes(states,logits,indexes1,indexes2,indexes_lens)
        index_group_size=int(len(n_states_list)/2)
        states_group_1=n_states_list[0:index_group_size]
        states_group_2=n_states_list[index_group_size:index_group_size*2]
        # compare all 'states' (all different indexes) posible pairs
        for i,states1 in enumerate(states_group_1):
            for j,states2 in enumerate(states_group_2):
                mean_sims,std_sims,num_of_examples = DifferentIndexExpi.compare_states_pairs(states1,states2)
                if num_of_examples ==0:
                    continue
                for layer, (sim_c, std_c) in enumerate(zip(mean_sims, std_sims)):
                    results.append({'word': masked_word, 'index1': indexes1[i],'index2': indexes2[j], 'layer': layer, 'examples': num_of_examples*2, 'similarity': sim_c,'std': std_c\
                        })
        return results


if __name__ == "__main__":
    most_freq_words=read_dict_from_json('n_most_frq')
    word_to_sen=read_dict_from_json('word_to_sen_dict')
    sentences_list=read_dict_from_json('sentences_list')
    exp2 = DifferentIndexExpi(word_to_sen,5)
    exp2.create_table_from_results(sentences_list,160000)



''''
the run in screen: 

if __name__ == "__main__":
    most_freq_words=read_dict_from_json('n_most_frq')
    word_to_sen=read_dict_from_json('word_to_sen_dict')
    sentences_list=read_dict_from_json('sentences_list')
    SameIndexExpi.create_table_from_results(word_to_sen,sentences_list)

'''
