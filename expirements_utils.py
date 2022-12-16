import json
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import bert_score
from torch import Tensor, logit
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import random
import faiss
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForMaskedLM.from_pretrained("roberta-base")


def write_dict_to_json(dict,fname):
    file_path='/home/itay.nakash/projects/smooth_language/data/'+fname
    json_str = json.dumps(dict)
    with open(file_path,'w') as f:
        f.write(json_str)

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

def get_similarity_between_two_states(states1:Tensor,states2:Tensor,mask_index:int):
    mask_index=int(mask_index)
    cos_s=torch.nn.CosineSimilarity(dim=0)
    sims=[]
    for i,_ in enumerate(states1):
        s1=states1[i][mask_index,:]
        s2=states2[i][mask_index,:]
        res=cos_s(s1,s2)
        # TODO: change in the states calc to 'no_grad' I think, and than remove teh detach from here:
        sims.append(np.mean(res.cpu().detach().numpy()))
    return tuple(sims)


def get_predicted_token_str(logits:Tensor,mask_index:int)->str:
    predicted_token_id=logits[int(mask_index),:].argmax(dim=-1).item()
    return tokenizer.decode(token_ids=predicted_token_id)
    

# checked for correct for a single sentence:
def check_if_predicted_correct(logits:Tensor,mask_index:int,masked_tokenid:str):
    if get_predicted_token_str(logits,mask_index)[1:] == masked_tokenid:
        return 1
    else:
        return 0

# run the model on a sentence, and return the states and logits as output
def get_sen_states(sen: str):
    input_ids_padded=convert_sentences_list_to_model_input([sen])
    with torch.no_grad():
        outputs = model(**input_ids_padded, output_hidden_states=True)
    return outputs['hidden_states'],outputs['logits']

# decode a pair encoded as a number (k)
def convert_k_to_pair(k:int)-> Tuple[int,int]:
    q = np.floor(np.sqrt(8*(k-1)+1)/2 +1.5)
    p = k - (q-1)*(q-2)/2
    return int(q),int(p)

# encode k pairs from 1 to n, as k pairs.
def generate_k_unique_pairs(n:int,k:int)-> List[int]:
    num_of_pairs = (n/2)*(n-1) # = n*(n-1)/2 number of pairs from 0 to n
    if (k > num_of_pairs):
        return [] #can't sample a bigger group than num of pairs
    pairs_indexes = random.sample(range(1,int(num_of_pairs)),k)
    return pairs_indexes

# run the model on a batch of sentences, and return the states and logits as output
def run_model_on_batch(sen_list:List[str])->Tuple[Tensor,Tensor]:
    input_ids_padded = convert_sentences_list_to_model_input(sen_list)
    with torch.no_grad():
        outputs = model(**input_ids_padded, output_hidden_states=True)
    # states[0].shape = (batch,tokens,hidden_d) - dim of each state, we have 13 states as number of layers
    return outputs['hidden_states'],outputs['logits']

# make sure that the prediction in pred_correct is correct
def test_pred_consistent(sen:List[str],pred_correct:int,pred_token:int,mask_index:int,masked_tokenid:int):
    _,logits=get_sen_states(sen)
    # make sure it predict the same token:
    assert pred_token==logits[:,int(mask_index)].argmax(dim=-1)
    # make sure it says it predicted correctly only if it did
    assert pred_correct== (pred_token==masked_tokenid)
    
# calculate bertscore on a batch
def get_bertscores_all_sents(pairs_indexes:List[int],current_sentences:List[List[str]]):
    sen_list1,sen_list2=[],[]
    for pair_indx in pairs_indexes:
        sen1_index,sen2_index=convert_k_to_pair(pair_indx)
        sen_list1.append(" ".join(current_sentences[sen1_index]))
        sen_list2.append(" ".join(current_sentences[sen2_index]))
    
    return bert_score.score(sen_list1,sen_list2, lang='en', verbose=True)

def layer_predict_with_norm(state:Tensor):
    lm_head=model.lm_head
    output=lm_head.forward(state)
    return output

def layer_predict_without_norm(state:Tensor):
    lm_head_without_norm=model.lm_head.decoder
    output=lm_head_without_norm.forward(state)
    return output


lm_norm_org_dim = nn.Sequential(
    model.lm_head.dense,
    nn.GELU(),
    model.lm_head.layer_norm,
)

def dense_layer_org_dim(states:Tuple[Tensor]):     
    n_states=[]
    for state in states:
        n_states.append(lm_norm_org_dim.forward(state))
    return tuple(n_states)

def fill_masked_pred_logits_softmax(logits:Tensor,mask_index:int,masked_tokenid:int,sen_indx:int,pred_tensor:Tensor):
    return logits[int(mask_index),masked_tokenid].item(),\
        logits[int(mask_index),pred_tensor[sen_indx]].item(),\
            torch.softmax(logits,dim=-1)[int(mask_index),masked_tokenid].item(),\
                torch.softmax(logits,dim=-1)[int(mask_index),pred_tensor[sen_indx]].item()

# To add:
# predicted word for the last layer
# predicted word for the current layer

# elads code:
# with layer norm:
# lm_head = roberta.lm_head
# run over the hidden states over the

# without layer norm:
# lm_head.decoder(hidden)

#def lmHead():
#    res = faiss.StandardGpuResources()  # use a single GPU
#    embeddings = model.roberta.embeddings.word_embeddings.weight
#    N, d = embeddings.shape
#    avoid_tokens = []
#    with torch.no_grad():
#        embeddings = embeddings.detach().clone()
#        # embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
#    #TODO: why it creates a copy of the embedding?
#        for tok in avoid_tokens:
#            embeddings[tok].fill_(0.)
#    index = faiss.IndexFlatIP(d) #TODO: ???
#    # index = faiss.IndexIVFFlat(index, d, 8192, faiss.METRIC_INNER_PRODUCT)#faiss.METRIC_L2)
#    # here we specify METRIC_L2, by default it performs inner-product search
#    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
#    gpu_index.train(embeddings) # add vectors to the index
#    gpu_index.add(embeddings) # add vectors to the index
#
#def classify_with_lm_head(x, k=1):
#        # logits = model.lm_head(x)
#        x = lm_norm(x)
#        logits = F.linear(x, model.lm_head.decoder.weight, model.lm_head.decoder.bias)
#        return logits.topk(k=k, dim=-1)
#
#def decode_inner_feats(sentence, k=1, index=None, normalize=False, classifier=None):
#    device = "cuda" if torch.cuda.is_available() else "cpu"
#    model.eval()
#    inputs = tokenizer(sentence, return_tensors="pt").to(device)
#    outputs = model(**inputs, output_hidden_states=True)
#    decoded = []
#    for layer, state in enumerate(outputs.hidden_states):
#        if index is not None:
#            if normalize:
#                state = lm_norm(state)
#                state = state / state.norm(dim=-1, keepdim=True)
#            D, I = index.search(state.view(-1, d), k)
#            # print(I)
#            # flatten I list of lists
#            for i in range(k):
#                nearest = [item[i] for item in I]
#                decoded.append({
#                    "sentence": tokenizer.decode(nearest),
#                    "distance": sum([d[i] for d in D]) / len(D),
#                    "layer": layer,
#                    'metric': f'{i}-NN'})
#        if classifier is not None:
#            distance, nearest = classifier(state, k=k)
#            for i in range(k):
#                decoded.append({
#                    "sentence": tokenizer.decode(nearest[:, :, i].view(-1)),
#                    "distance": distance[:, :, i].mean().item(),
#                    "layer": layer})
#    return decoded

if __name__ == "__main__":
    #test:
    a=generate_k_unique_pairs(4,5)
    for i in a:
        print(convert_k_to_pair(i))