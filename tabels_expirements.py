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
MAX_SEN_PAIRS=5
PAIRS_PER_WORD=40
''' expirements:
        1. same word, same index (SameIndexExpiTable)
        2. same word, different index
        3. different word, same index
        4. different word, different index

    features:
    sentence1, sentence2, word 1, word 2, sen1_len, sen2_len,
    index 1, index 2, layer, examples, similarity, std, correct 1,
    correct 2, mean logit masked, mean logit predicted, mean soft max masked,
    mean softmax predicted, mean similarity to masked LM head,
    BERT score between sentences without masking, BERTscore with masking
'''
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
        data = {'word1','word2','index1','index2','layer',\
            'n_examples','sim','std','correct1','correct2',\
                'logits1_masked','logits1_pred','softmax1_masked','softmax1_pred',
                'logits2_masked','logits2_pred','softmax2_masked','softmax2_pred',
                'sim_maskedLM','sim_maskedLM_not_normalize','bertscore_n_m_r','bertscore_n_m_p','bertscore_n_m_f1'\
                ,'bertscore_m_r','bertscore_m_p','bertscore_m_f1','sen_len1','sen_len2'}
        for key in data:
            data[key]=[]
        values={}
        sen_list_batch = random.choices(self.sentences_list,k=1000)
        for i,sen in enumerate(sen_list_batch):
            c_sen1=sen[:]
            c_sen2=sen[:]
            print(f'-------------------------- sentence number: {i} num_of_sentences = {num_of_sentences}-----------------')
            if num_of_sentences<i:
                break
            sen_len = len(sen)
            pairs_indexes = expirements_utils.generate_k_unique_pairs(sen_len,NUM_OF_MIXES)
            for j in range(NUM_OF_MIXES):
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



class SameIndexDiffWord:

    def verify_results(values:Dict,pred_correct:Tensor,pred_tensor:Tensor,sen1_index:int,sen2_index:int,mask_index:int,masked_tokenid:int):
        sen1=values['sen1']
        sen2=values['sen2']
        expirements_utils.test_pred_consistent(sen=sen1,pred_correct=pred_correct[sen1_index],pred_token=pred_tensor[sen1_index],mask_index=mask_index,masked_tokenid=masked_tokenid)
        expirements_utils.test_pred_consistent(sen=sen2,pred_correct=pred_correct[sen2_index],pred_token=pred_tensor[sen2_index],mask_index=mask_index,masked_tokenid=masked_tokenid)
        
        print(f'first sen:{sen1}\n second sen:{sen2} \
            \n sen1 pred_c:{pred_correct[sen1_index]}\n sen2 pred_c:{pred_correct[sen2_index]}\
            \n sen1 word:{pred_tensor[sen1_index]}\n sen2 pred_c:{pred_tensor[sen2_index]}')
   
    def create_data_dict():
        keys=['sen1','sen2','sen_len1','sen_len2','word1','word2','index1','index2','layer',\
    'n_exampels','sim','std','correct1','correct2',\
        'logits1_masked','logits1_pred','softmax1_masked','softmax1_pred',
        'logits2_masked','logits2_pred','softmax2_masked','softmax2_pred',
        'mean_sim_to_maskedLM','bertscore_nm_r','bertscore_nm_p','bertscore_nm_f1'\
        ,'bertscore_m_r','bertscore_m_p','bertscore_m_f1']
        data = dict.fromkeys(keys)
        for key in data:
            data[key]=[]
        values=dict.fromkeys(keys)
        return data,values
    
    def seperate_correct_incorrect_predict(states: Tensor, logits: Tensor,mask_index:int, masked_tokenid: float) -> Tuple[Tensor]:
        # get the predicted token_id of each sen:
        arg_max_tensor=logits[:,mask_index].argmax(dim=-1)
        correct_mask = arg_max_tensor==masked_tokenid
        incorrect_mask=arg_max_tensor!=masked_tokenid
        
        #TODO: find a way to mark what was correct and what not
        states_c= tuple([state[correct_mask] for state in enumerate(states)])
        states_nc= tuple([state[incorrect_mask] for state in states])
        logits_c= logits[correct_mask]
        # returns the logits on all the tokens, just on sentences predicted incorrect
        logits_nc= logits[incorrect_mask]
        predicted_tokens_in_nc = arg_max_tensor[incorrect_mask]

        return states_c, states_nc,logits_c\
            ,logits_nc,predicted_tokens_in_nc

    def indice_correct_incorrect_predict(states: Tensor, logits: Tensor,mask_index:int, masked_tokenid: float) -> Tuple[Tensor]:
        # get the predicted token_id of each sen:
        arg_max_tensor=logits[:,mask_index].argmax(dim=-1)
        #get the masks:
        correct_mask = arg_max_tensor==masked_tokenid
        incorrect_mask=arg_max_tensor!=masked_tokenid

        #tensor as size of batch:
        pred_correct=torch.zeros_like(states[0][:,0,0])
        for state in states:
            for i in range(state.shape[0]):
                if correct_mask[i]:
                    pred_correct[i]=1
                else:
                    pred_correct[i]=0

        return pred_correct,arg_max_tensor



    def create_table_from_results(words_dict:Dict ,sentences_list:List,num_of_iter=sys.maxsize):
        data,values = SameIndexExpiTable.create_data_dict()
        for masked_word in words_dict:
            num_of_iter-=1
            if num_of_iter==0:
                break
            masked_tokenid = tokenizer.convert_tokens_to_ids(masked_word)
            num_of_pairs = (len(words_dict[masked_word])/2)*(len(words_dict[masked_word])-1)
            maskes_paris=expirements_utils.generate_k_unique_pairs(len(words_dict[masked_word]),min(num_of_pairs-1,PAIRS_PER_WORD))
            for mask_pair in maskes_paris:
                mask_index1,mask_index2 = expirements_utils.convert_k_to_pair(mask_pair)
                current_sentences_m1=[(mask_word(sentences_list[i],int(mask_index1))[0]) for i in words_dict[masked_word][mask_index1]]
                current_sentences_m2= [(mask_word(sentences_list[i],int(mask_index2))[0]) for i in words_dict[masked_word][mask_index2]]
                states1,logits1=expirements_utils.run_model_on_batch(current_sentences_m1)
                states2,logits2=expirements_utils.run_model_on_batch(current_sentences_m2)
                pred_correct1,pred_tensor1 = SameIndexExpiTable.indice_correct_incorrect_predict(states1,logits1,int(mask_index1),masked_tokenid)
                pred_correct2,pred_tensor = SameIndexExpiTable.indice_correct_incorrect_predict(states1,logits1,int(mask_index1),masked_tokenid)
                
            print('------------------------- finished word -----------------------')
        df = pd.DataFrame(data)
        df.to_csv('/home/itay.nakash/projects/smooth_language/results/expirement_result.csv', index=False)    




class SameIndexExpiTable:

    def verify_results(values:Dict,pred_correct:Tensor,pred_tensor:Tensor,sen1_index:int,sen2_index:int,mask_index:int,masked_tokenid:int):
        sen1=values['sen1']
        sen2=values['sen2']
        expirements_utils.test_pred_consistent(sen=sen1,pred_correct=pred_correct[sen1_index],pred_token=pred_tensor[sen1_index],mask_index=mask_index,masked_tokenid=masked_tokenid)
        expirements_utils.test_pred_consistent(sen=sen2,pred_correct=pred_correct[sen2_index],pred_token=pred_tensor[sen2_index],mask_index=mask_index,masked_tokenid=masked_tokenid)
        
        print(f'first sen:{sen1}\n second sen:{sen2} \
            \n sen1 pred_c:{pred_correct[sen1_index]}\n sen2 pred_c:{pred_correct[sen2_index]}\
            \n sen1 word:{pred_tensor[sen1_index]}\n sen2 pred_c:{pred_tensor[sen2_index]}')
    def create_data_dict():
        keys=['sen1','sen2','sen_len1','sen_len2','word1','word2','index1','index2',
        'layer', 'n_exampels','sim','std','correct1','correct2',
        'bertscore_nm_r','bertscore_nm_p','bertscore_nm_f1'
        ,'bertscore_m_r','bertscore_m_p','bertscore_m_f1',
        'logits1_masked', 'logits1_pred','softmax1_masked','softmax1_pred',
        'logits2_masked','logits2_pred','softmax2_masked','softmax2_pred',
        'logits1_masked_no_norm', 'logits1_pred_no_norm','softmax1_masked_no_norm','softmax1_pred_no_norm',
        'logits2_masked_no_norm','logits2_pred_no_norm','softmax2_masked_no_norm','softmax2_pred_no_norm',
        'logits1_masked_cl', 'logits1_pred_cl','softmax1_masked_cl','softmax1_pred_cl',
        'logits2_masked_cl','logits2_pred_cl','softmax2_masked_cl','softmax2_pred_cl',
        'logits1_masked_cl_no_norm', 'logits1_pred_cl_no_norm','softmax1_masked_cl_no_norm','softmax1_pred_cl_no_norm',
        'logits2_masked_cl_no_norm','logits2_pred_cl_no_norm','softmax2_masked_cl_no_norm','softmax2_pred_cl_no_norm',
        'predicted_token1_cl','predicted_token1_ll','predicted_token2_cl','predicted_token2_ll',
        'predicted_token1_cl_no_norm','predicted_token1_ll_no_norm','predicted_token2_cl_no_norm','predicted_token2_ll_no_norm']
        data = dict.fromkeys(keys)
        for key in data:
            data[key]=[]
        values={}
        return data,values
    
    def seperate_correct_incorrect_predict(states: Tensor, logits: Tensor,mask_index:int, masked_tokenid: float) -> Tuple[Tensor]:
        # get the predicted token_id of each sen:
        arg_max_tensor=logits[:,mask_index].argmax(dim=-1)
        correct_mask = arg_max_tensor==masked_tokenid
        incorrect_mask=arg_max_tensor!=masked_tokenid
        
        #TODO: find a way to mark what was correct and what not
        states_c= tuple([state[correct_mask] for state in enumerate(states)])
        states_nc= tuple([state[incorrect_mask] for state in states])
        logits_c= logits[correct_mask]
        # returns the logits on all the tokens, just on sentences predicted incorrect
        logits_nc= logits[incorrect_mask]
        predicted_tokens_in_nc = arg_max_tensor[incorrect_mask]

        return states_c, states_nc,logits_c\
            ,logits_nc,predicted_tokens_in_nc

    def indice_correct_incorrect_predict(states: Tensor, logits: Tensor,mask_index:int, masked_tokenid: float) -> Tuple[Tensor]:
        # get the predicted token_id of each sen:
        arg_max_tensor=logits[:,mask_index].argmax(dim=-1)
        #get the masks:
        correct_mask = arg_max_tensor==masked_tokenid
        incorrect_mask=arg_max_tensor!=masked_tokenid

        #tensor as size of batch:
        pred_correct=torch.zeros_like(states[0][:,0,0])
        for state in states:
            for i in range(state.shape[0]):
                if correct_mask[i]:
                    pred_correct[i]=1
                else:
                    pred_correct[i]=0

        return pred_correct,arg_max_tensor



    def create_table_from_results(words_dict:Dict ,sentences_list:List,num_of_iter=sys.maxsize):
        org_num_of_iter=num_of_iter
        data,values = SameIndexExpiTable.create_data_dict()
        for iter,masked_word in enumerate(words_dict):
            if len(masked_word)<2 or iter<41: # patch 
                continue
            num_of_iter-=1
            if num_of_iter==0:
                break
            masked_tokenid = tokenizer.convert_tokens_to_ids(masked_word)
            for mask_index in words_dict[masked_word]:
                # mask all words in the index:
                current_sentences_m=[(mask_word(sentences_list[i],int(mask_index))[0]) for i in words_dict[masked_word][mask_index]]
                states,logits=expirements_utils.run_model_on_batch(current_sentences_m)
                pred_correct,pred_tensor = SameIndexExpiTable.indice_correct_incorrect_predict(states,logits,int(mask_index),masked_tokenid)
                num_sens=len(current_sentences_m)
                max_sen_indx=num_sens-1
                num_pairs = int((max_sen_indx*(max_sen_indx-1)/2)-1)
                if max_sen_indx<MIN_NUM_OF_SEN:
                    continue
                pairs_indexes = expirements_utils.generate_k_unique_pairs(n=max_sen_indx,k=min(num_pairs,MAX_SEN_PAIRS))
                # bertscore with mask:
                m_P, m_R, m_F1=expirements_utils.get_bertscores_all_sents(pairs_indexes,current_sentences_m)
                # bertscore without mask
                nm_P,nm_R,nm_F1=expirements_utils.get_bertscores_all_sents(pairs_indexes,sentences_list)
                for i,pair_indx in enumerate(pairs_indexes):
                    sen1_indx,sen2_indx=expirements_utils.convert_k_to_pair(pair_indx)
                    states1=tuple([state[sen1_indx,:,:] for state in states])
                    states2=tuple([state[sen2_indx,:,:] for state in states])

                    # basic information:
                    values['sen1']=current_sentences_m[sen1_indx]
                    values['sen2']=current_sentences_m[sen2_indx]
                    values['sen_len1']=len(current_sentences_m[sen1_indx])
                    values['sen_len2']=len(current_sentences_m[sen2_indx])
                    values['word1']=masked_word
                    values['word2']=masked_word
                    values['index1']=sen1_indx
                    values['index2']=sen2_indx
                    values['n_exampels']=1
                    
                    #similarity:
                    values['std']=0
                    values['correct1']=pred_correct[int(sen1_indx)].item()
                    values['correct2']=pred_correct[int(sen2_indx)].item()

                    # logtits and softmax:
                    values['logits1_masked'],values['logits1_pred'],\
                        values['softmax1_masked'],values['softmax1_pred']=expirements_utils.fill_masked_pred_logits_softmax(logits[sen1_indx],mask_index,masked_tokenid,sen1_indx,pred_tensor)
                        
                    values['logits2_masked'], values['logits2_pred'],\
                        values['softmax2_masked'], values['softmax2_pred']=expirements_utils.fill_masked_pred_logits_softmax(logits[sen2_indx],mask_index,masked_tokenid,sen2_indx,pred_tensor)
                    
                    # without norm:
                    logits_no_norm1=expirements_utils.layer_predict_without_norm(states1[12])
                    logits_no_norm2=expirements_utils.layer_predict_without_norm(states2[12])
                    values['logits1_masked_no_norm'],values['logits1_pred_no_norm'],\
                        values['softmax1_masked_no_norm'],values['softmax1_pred_no_norm']=expirements_utils.fill_masked_pred_logits_softmax(logits_no_norm1,mask_index,masked_tokenid,sen1_indx,pred_tensor)
                    values['logits2_masked_no_norm'],values['logits2_pred_no_norm'],\
                        values['softmax2_masked_no_norm'],values['softmax2_pred_no_norm']=expirements_utils.fill_masked_pred_logits_softmax(logits_no_norm2,mask_index,masked_tokenid,sen2_indx,pred_tensor)
                    
                    #predicted_token1_cl','predicted_token1_ll','predicted_token2_cl','predicted_token2_ll',
                    #'predicted_token1_cl_no_norm','predicted_token1_ll_no_norm','predicted_token2_cl_no_norm','predicted_token2_ll_no_norm']
                    # predicted token ll:
                    values['predicted_token1_ll']=expirements_utils.get_predicted_token_str(logits[sen1_indx],mask_index)
                    values['predicted_token1_ll_no_norm']=expirements_utils.get_predicted_token_str(logits_no_norm1,mask_index)
                    values['predicted_token2_ll']=expirements_utils.get_predicted_token_str(logits[sen2_indx],mask_index)
                    values['predicted_token2_ll_no_norm']=expirements_utils.get_predicted_token_str(logits_no_norm2,mask_index)


                    #bertscores:
                    values['bertscore_m_r']=m_R[i].item()
                    values['bertscore_m_p']=m_P[i].item()
                    values['bertscore_m_f1']=m_F1[i].item()
                    values['bertscore_nm_r']=nm_R[i].item()
                    values['bertscore_nm_p']=nm_P[i].item()
                    values['bertscore_nm_f1']=nm_F1[i].item()

                    c_sim=expirements_utils.get_similarity_between_two_states(states1,states2)
                    for i in range(len(states)):
                        for key in values:
                            data[key].append(values[key])

                        data['layer'].append(i)
                        data['sim'].append(c_sim[i].item())

                        # collect states information:
                        state_no_norm1=expirements_utils.layer_predict_without_norm(states1[i])
                        state_no_norm2=expirements_utils.layer_predict_without_norm(states2[i])
                        state_with_norm1=expirements_utils.layer_predict_with_norm(states1[i])
                        state_with_norm2=expirements_utils.layer_predict_with_norm(states2[i])
                        logits1_masked_cl,logits1_pred_cl,softmax1_masked_cl,softmax1_pred_cl= expirements_utils.fill_masked_pred_logits_softmax(state_with_norm1,mask_index,masked_tokenid,sen1_indx,pred_tensor)
                        logits2_masked_cl,logits2_pred_cl,softmax2_masked_cl,softmax2_pred_cl= expirements_utils.fill_masked_pred_logits_softmax(state_with_norm2,mask_index,masked_tokenid,sen2_indx,pred_tensor)
                        logits1_masked_cl_no_norm,logits1_pred_cl_no_norm,softmax1_masked_cl_no_norm,softmax1_pred_cl_no_norm = expirements_utils.fill_masked_pred_logits_softmax(state_no_norm1,mask_index,masked_tokenid,sen1_indx,pred_tensor)
                        logits2_masked_cl_no_norm,logits2_pred_cl_no_norm,softmax2_masked_cl_no_norm,softmax2_pred_cl_no_norm = expirements_utils.fill_masked_pred_logits_softmax(state_no_norm2,mask_index,masked_tokenid,sen2_indx,pred_tensor)
                        
                        data['logits1_masked_cl'].append(logits1_masked_cl)
                        data['logits1_pred_cl'].append(logits1_pred_cl)
                        data['softmax1_masked_cl'].append(softmax1_masked_cl)
                        data['softmax1_pred_cl'].append(softmax1_pred_cl)

                        data['logits2_masked_cl'].append(logits2_masked_cl)
                        data['logits2_pred_cl'].append(logits2_pred_cl)
                        data['softmax2_masked_cl'].append(softmax2_masked_cl)
                        data['softmax2_pred_cl'].append(softmax2_pred_cl)
                        
                        data['logits1_masked_cl_no_norm'].append(logits1_masked_cl_no_norm)
                        data['logits1_pred_cl_no_norm'].append(logits1_pred_cl_no_norm)
                        data['softmax1_masked_cl_no_norm'].append(softmax1_masked_cl_no_norm)
                        data['softmax1_pred_cl_no_norm'].append(softmax1_pred_cl_no_norm)

                        data['logits2_masked_cl_no_norm'].append(logits2_masked_cl_no_norm)
                        data['logits2_pred_cl_no_norm'].append(logits2_pred_cl_no_norm)
                        data['softmax2_masked_cl_no_norm'].append(softmax2_masked_cl_no_norm)
                        data['softmax2_pred_cl_no_norm'].append(softmax2_pred_cl_no_norm)

                        data['predicted_token1_cl'].append(expirements_utils.get_predicted_token_str(state_with_norm1,mask_index))
                        data['predicted_token1_cl_no_norm'].append(expirements_utils.get_predicted_token_str(state_no_norm1,mask_index))
                        data['predicted_token2_cl'].append(expirements_utils.get_predicted_token_str(state_with_norm2,mask_index))
                        data['predicted_token2_cl_no_norm'].append(expirements_utils.get_predicted_token_str(state_no_norm2,mask_index))

                        k=1
                        if k==2:
                            df = pd.DataFrame(data)
                            df.to_csv('/home/itay.nakash/projects/smooth_language/results/df_same_sen_'+str(org_num_of_iter))
                            return


                    print('---finish pair---')
            print('------------------------- finished word -----------------------')

        df = pd.DataFrame(data)
        df.to_csv('/home/itay.nakash/projects/smooth_language/results/df_same_sen_'+str(org_num_of_iter))




if __name__ == "__main__":
    most_freq_words=read_dict_from_json('n_most_frq')
    word_to_sen=read_dict_from_json('word_to_sen_dict')
    sentences_list=read_dict_from_json('sentences_list')
    
    SameIndexExpiTable.create_table_from_results(word_to_sen,sentences_list,60)
