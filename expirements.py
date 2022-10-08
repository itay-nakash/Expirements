import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForMaskedLM.from_pretrained("roberta-base")



def read_dict_from_json(fname):
    with open(fname) as json_file:
        return json.load(json_file)


def get_mask_features(sentence, position=None):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt")
    mask_index = int(torch.where(inputs.input_ids == tokenizer.mask_token_id)[1])

    if position is None:
        position = mask_index

    outputs = model(**inputs, output_hidden_states=True)
    states = list(outputs.hidden_states) + [outputs.logits]
    hidden_states = [state[:, position].view(-1) for state in states]
    # print(outputs.logits[:,mask_index].argmax(dim=-1))
    print(tokenizer.decode(outputs.logits[:,mask_index].argmax(dim=-1)[0]))
    return hidden_states

def compare_dists(sentence1,sentence2):
    feats1 = get_mask_features(sentence1)
    feats2 = get_mask_features(sentence2)
    cosine_dists = [float(F.cosine_similarity(f1,f2, dim=0)) for (f1,f2) in zip(feats1, feats2)]
    return cosine_dists

def mask_word(sentence,location):
    sentence[location]='<mask>'

if __name__ == "__main__":
    most_freq_words=read_dict_from_json('n_most_frq')
    word_to_sen=read_dict_from_json('word_to_sen_dict')
    sentences_list=read_dict_from_json('sentences_list')
    
    #sentence1 is the first sentence(0) that has the word visit in location 7:
    sentence1= sentences_list[word_to_sen['visit']['7'][0]]
    
    #sentence1 is the second sentence(1) that has the word visit in location 7:
    sentence2= sentences_list[word_to_sen['visit']['7'][1]]
    
    print(sentence1)
    print(sentence2)

    print(compare_dists('I learned some of them in my <mask> to New Liberty.','As real estate professionals, we get to <mask> people’s homes every day and talk to buyers and sellers about their favorite features.'))
