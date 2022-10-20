from calendar import c
import json
import torch
import itertools
import pandas as pd
import numpy as np
import torch.nn.functional as F
from typing import List,Dict,Set,Union,Optional,Tuple,Any
from torch import Tensor
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForMaskedLM.from_pretrained("roberta-base")
MIN_NUM_OF_SEN = 10

def read_dict_from_json(fname):
    with open(fname) as json_file:
        return json.load(json_file)

# for batch tensors:
def cosine_similarity(tensor1: Tensor, tensor2: Tensor, eps: float = 1e-8):
    tensor1_norm = tensor1.norm(dim=1).unsqueeze(1)
    tensor2_norm = tensor2.norm(dim=1).unsqueeze(1)
    tensor1_normed = tensor1 / torch.max(tensor1_norm, eps * torch.ones_like(tensor1_norm))
    tensor2_normed = tensor2 / torch.max(tensor2_norm, eps * torch.ones_like(tensor2_norm))
    return torch.mm(tensor1_normed, tensor2_normed.T)


# states[0].shpae = (batch,tokens,hidden_d)
def remove_incorrect(states: Tensor, logits: Tensor,mask_index:int, masked_tokenid: float) -> Tensor:
    # get to token_id of the masked token:
    # get the predicted token_id of each sen:
    arg_max_tensor=logits[:,mask_index].argmax(dim=-1)
    
    #create tensors for conditions:
    ones_tensor = torch.ones_like(arg_max_tensor)
    zeros_tensor = torch.zeros_like(arg_max_tensor)

    # I probably wrote it pretty bad and messy, might check about it in the future  
    logits_mod = torch.where(logits[:,mask_index].argmax(dim=-1)==masked_tokenid,ones_tensor,zeros_tensor)
    indices=logits_mod.nonzero()
    new_states=[]
    #indices is a list of a list of indices, so I need to merge them to one list:
    list_of_indices=list(itertools.chain.from_iterable(indices))
    #if it empty, return an empty tuple:
    if list_of_indices==[]:
        return tuple()
    for state in states:
        new_states.append(torch.index_select(state,0,torch.tensor(list_of_indices)))
    return tuple(new_states)

def find_batch_similarities(senteneces: List[List[str]],mask_index: int, masked_word: str, only_correct_flag :bool = False) -> Tuple[Tensor,Tensor,float,int]:
    # convert the sentences list to input_ids:
    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in senteneces]
    # convert the token str to token_id:
    masked_tokenid=tokenizer.convert_tokens_to_ids(masked_word)
    
    #add padding:
    input_ids_padded = tokenizer(tokenizer.batch_decode(input_ids), add_special_tokens=False, padding=True, return_tensors="pt")
    size = input_ids_padded.input_ids.shape
    with torch.no_grad():
        outputs = model(**input_ids_padded, output_hidden_states=True)
    # states[0].shape = (batch,tokens,hidden_d) (dim of each state, we have 13 states as number of layers)
    states = outputs['hidden_states']
    logits = outputs['logits']
    if only_correct_flag:
        states = remove_incorrect(states,logits,mask_index,masked_tokenid)

    #calculated to all enries, not just correct ones:
    masked_token_logits_list=[logits[:,mask_index,:][i][masked_tokenid].item() for i in range(logits.shape[0])]
    mean_logic_on_masked=sum(masked_token_logits_list)/len(masked_token_logits_list)
    mean_sims = []
    std_sims=[]
    num_of_examples = 0
    for state in states:
        # its here since we dont want to check it if state is an empty tuple: 
        num_of_examples = state.shape[0]
        # state.shape = (batch,hidden_d)
        state = state[:,mask_index,:]
        # sims.shpae = (batch,batch)
        sims = cosine_similarity(state, state)
        sims = sims.cpu().numpy()
        sims = sims[np.triu_indices(sims.shape[0],k=1)]
        mean_sims.append(np.mean(sims))
        std_sims.append(np.std(sims))

    return mean_sims,std_sims,mean_logic_on_masked,num_of_examples

def get_mask_features(sentence:str, position=None):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt")
    mask_index = int(torch.where(inputs.input_ids == tokenizer.mask_token_id)[1])

    if position is None:
        position = mask_index

    outputs = model(**inputs, output_hidden_states=True)
    states = list(outputs.hidden_states) + [outputs.logits]
    hidden_states = [state[:, position].view(-1) for state in states]
    # print the predicted word
    predicted_word = tokenizer.decode(outputs.logits[:,mask_index].argmax(dim=-1)[0])
    return predicted_word, hidden_states

def compare_dists(sentence1,sentence2):
    feats1 = get_mask_features(sentence1)
    feats2 = get_mask_features(sentence2)
    cosine_dists = [float(F.cosine_similarity(f1,f2, dim=0)) for (f1,f2) in zip(feats1, feats2)]
    return cosine_dists

def mask_word(sentence,location):
    sentence[location]=tokenizer.mask_token
    return sentence


def create_table_from_results(words, only_correct_flag :bool = False):
    results = []
    for word in words:
        for mask_loc in words[word]:
            current_sentences=[mask_word(sentences_list[i],int(mask_loc)) for i in words[word][mask_loc]]
            if len(current_sentences) < MIN_NUM_OF_SEN:
                continue    
            sims, stds, mean_logic_on_masked ,n= find_batch_similarities(current_sentences,int(mask_loc),word, only_correct_flag)
            # added a minor fix, since I remove the incorrect examples in find_batch_similarities, I want to check again that its enough examples
            # before I write it down. if not, just continue and dont save it.
            if n < MIN_NUM_OF_SEN:
                continue
            print(f'word: {word}, index: {mask_loc}, examples: {n}')
            for layer, (sim, std) in enumerate(zip(sims, stds)):
                results.append({'word': word, 'index': mask_loc, 'layer': layer, 'examples': n, 'similarity': sim,'mean logic':mean_logic_on_masked, 'std': std})
    df = pd.DataFrame(results)
    if only_correct_flag:
        df.to_csv('/home/itay.nakash/projects/smooth_language/expirement_result_only_correct_predict.csv', index=False)    
    else:
        df.to_csv('/home/itay.nakash/projects/smooth_language/expirement_result_all.csv', index=False)    


if __name__ == "__main__":
    most_freq_words=read_dict_from_json('n_most_frq')
    word_to_sen=read_dict_from_json('word_to_sen_dict')
    sentences_list=read_dict_from_json('sentences_list')
    
    #sentence1= sentences_list[word_to_sen['Ġmust']['2'][0]]    
    #sentence2= sentences_list[word_to_sen['Ġmust']['2'][1]]
    #current_sentences=[mask_word(sentences_list[index],z2) for index in word_to_sen['Ġmust']['2']]    
    #get_batch_mean_similarity(current_sentences,2)

    create_table_from_results(word_to_sen,True)
    create_table_from_results(word_to_sen,False)