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

    def sample_original_couple(sen:List,already_been:List):
        num_of_sampeling=0
        mask_index1,mask_index2=random.sample(range(0,len(sen)),2)
        already_been+=[(mask_index1,mask_index2),(mask_index2,mask_index1)]
        while num_of_sampeling<MAX_ATTEMPTS_RND and (mask_index1,mask_index2) in already_been:
            mask_index1,mask_index2=random.sample(range(0,len(sen)),2)
        if num_of_sampeling == MAX_ATTEMPTS_RND:
            print('got to dobule sen!')
        return mask_index1,mask_index2,already_been,num_of_sampeling

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
            already_been=[]
            for j in range(NUM_OF_MIXES):
                mask_index1,mask_index2,already_been,num_of_sampeling=Table_Expirement.sample_original_couple(sen,already_been)
                if num_of_sampeling==MAX_ATTEMPTS_RND:
                    break
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


    def create_same_index_diff_sen_table(self):
        
        print('finished')



if __name__ == "__main__":
    most_freq_words=read_dict_from_json('n_most_frq')
    word_to_sen=read_dict_from_json('word_to_sen_dict')
    sentences_list=read_dict_from_json('sentences_list')
    
    exp2 = Table_Expirement(most_freq_words,word_to_sen,sentences_list)

    exp2.create_same_sen_table(100)