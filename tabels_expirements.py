import itertools
from calendar import c
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import random
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import bert_score
from torch import Tensor, logit
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
from expirements_utils import tokenizer, model, convert_sentences_list_to_model_input

from expirements_utils import read_dict_from_json,mask_word,cosine_similarity
import expirements_utils

MAX_ATTEMPTS_RND=10
MIN_NUM_OF_SEN=10
MAX_BERTSCORE_VALUE = 1
NUM_OF_MIXES=50

# TODO: currently it is implemented in a stright-foward way, but not effificet.
# I run the model each time on sentences matching instead of working with batch,
# needs to be changed.
class Table_Expirement:

    def __init__(self,most_freq_words:List, words_dict:Dict,sentences_list:List):
        # the size of each :
        self.most_freq_words = most_freq_words
        self.words_dict = words_dict
        self.sentences_list = sentences_list

    def get_sen_states(sen: str):
        input_ids_padded=convert_sentences_list_to_model_input([sen])
        with torch.no_grad():
            outputs = model(**input_ids_padded, output_hidden_states=True)
        return outputs['hidden_states'],outputs['logits']


    def create_same_sen_table(self,num_of_sentences=sys.maxsize):
        data = {'word1':[],'word2':[],'index1':[],'index2':[],'layer':[],\
            'n_examples':[],'sim':[],'std':[],'correct1':[],'correct2':[],\
                'logits1_masked':[],'logits1_pred':[],'softmax1_masked':[],'softmax1_pred':[],
                'logits2_masked':[],'logits2_pred':[],'softmax2_masked':[],'softmax2_pred':[],
                'mean_sim_to_maskedLM':[],'bertscore_n_m_r':[],'bertscore_n_m_p':[],'bertscore_n_m_f1':[]\
                ,'bertscore_m_r':[],'bertscore_m_p':[],'bertscore_m_f1':[],'sen_len1':[],'sen_len2':[]}
        values={}
        sen_list_batch = random.choices(self.sentences_list,k=1000)
        for i,sen in enumerate(sen_list_batch):
            c_sen1=sen[:]
            c_sen2=sen[:]
            print(f'-------------------------- sentence number: {i} num_of_sentences = {num_of_sentences}-----------------')
            if num_of_sentences<i:
                break
            sen_len = len(sen)
            pairs_indexes,current_num_of_mixes = expirements_utils.generate_k_unique_pairs(sen_len,NUM_OF_MIXES)
            for j in range(current_num_of_mixes):
                mask_index1,mask_index2=expirements_utils.convert_k_to_pair(pairs_indexes[j])
                # mask both words:
                c_sen_mask1,masked_tokenid1=mask_word(c_sen1,int(mask_index1))
                c_sen_mask2,masked_tokenid2=mask_word(c_sen2,int(mask_index2))
                states1,logits1=Table_Expirement.get_sen_states(c_sen_mask1)
                states2,logits2=Table_Expirement.get_sen_states(c_sen_mask2)

                # data for table:
                values['word1']=sen[mask_index1]
                values['word2']=sen[mask_index2]
                values['index1']=mask_index1
                values['index2']=mask_index2
                values['n_examples'] = 1
                values['correct1'] = expirements_utils.check_if_predicted_correct(logits1,mask_index1,masked_tokenid1)
                values['correct2'] = expirements_utils.check_if_predicted_correct(logits2,mask_index2,masked_tokenid2)
                
                values['sen_len1']=sen_len
                values['sen_len2']=sen_len
                
                r,p,f1= expirements_utils.get_bertscore_between_sents(c_sen_mask1,c_sen_mask2)
                values['bertscore_n_m_r'],values['bertscore_n_m_p'],values['bertscore_n_m_f1'] = (r.mean().item(),p.mean().item(),f1.mean().item()) 
                # here its identical sentences:
                values['bertscore_m_r'],values['bertscore_m_p'],values['bertscore_m_f1'] = (1,1,1)
                values['mean_sim_to_maskedLM'] = -1

                values['sim'] = expirements_utils.get_similarity_between_two_states(states1,states2)
                values['std'] = 0
                # logits of two sentences (TODO: split to function):
                values['logits1_masked']=logits1[0,mask_index1,masked_tokenid1].item()
                values['predicted_token_id1']=logits1[0,mask_index1].argmax(dim=-1).item()
                values['logits1_pred']=logits1[0,mask_index1,values['predicted_token_id1']].item()
                values['softmax1_masked']=torch.softmax(logits1,dim=-1)[0,mask_index1,masked_tokenid1].item()
                values['softmax1_pred']=torch.softmax(logits1,dim=-1)[0,mask_index1,values['predicted_token_id1']].item()

                values['logits2_masked']=logits2[0,mask_index2,masked_tokenid2].item()
                values['predicted_token_id2']=logits2[0,mask_index2].argmax(dim=-1).item()
                values['logits2_pred']=logits2[0,mask_index2,values['predicted_token_id1']].item()
                values['softmax2_masked']=torch.softmax(logits2,dim=-1)[0,mask_index2,masked_tokenid2].item()
                values['softmax2_pred']=torch.softmax(logits2,dim=-1)[0,mask_index2,masked_tokenid2].item()
                for i in range(len(states1)):
                    for key in data:
                        if key=='layer':
                            continue
                        if key!='sim':
                            data[key].append(values[key])
                        else:
                            data[key].append(values[key][i].item())
                    data['layer'].append(i)
            print(f'-------------------------- sentence number: {i} num_of_sentences = {num_of_sentences}-----------------')

        df=pd.DataFrame.from_dict(data)
        with open('/home/itay.nakash/projects/smooth_language/results/df_same_sen_'+str(num_of_sentences), 'a') as f:
            dfAsString = df.to_string(header=False, index=False)
            f.write(dfAsString)



class SameIndexExpiTable:

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
                # mask all words in the index:
                current_sentences=[mask_word(sentences_list[i],int(mask_loc)) for i in words_dict[word][mask_loc]]
                sims_c,sims_nc, stds_c,stds_nc ,num_of_examples_c,\
                num_of_examples_nc,softmax_c_mean,\
                softmax_nc_masked_mean,softmax_nc_for_pred_tok_mean,\
                    logits_c_mean,logits_nc_masked_tok_mean,logits_nc_for_pred_tok_mean = SameIndexExpiTable.find_batch_similarities(current_sentences,int(mask_loc),word)

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





if __name__ == "__main__":
    most_freq_words=read_dict_from_json('n_most_frq')
    word_to_sen=read_dict_from_json('word_to_sen_dict')
    sentences_list=read_dict_from_json('sentences_list')
    
    exp2 = Table_Expirement(most_freq_words,word_to_sen,sentences_list)

    exp2.create_same_sen_table(100)