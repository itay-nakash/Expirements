import json
from xml.etree.ElementPath import find
from transformers import AutoTokenizer
import numpy as np
import torch
import itertools
from datasets import load_dataset
import nltk
from collections import Counter



def find_all_in_list(word, list):
    return [i for i, x in enumerate(list) if x == word]

def find_all_in_string(word, sent):
    indexex=[]
    words_list = sent.split()
    for i,w in enumerate(words_list):
        if w ==word:
            indexex.append(i)
    return indexex


def write_dict_to_json(dict,fname):
    with open(fname, "w") as outfile:
        json.dump(dict, outfile)

def read_dict_from_json(fname):
    with open(fname) as json_file:
        return json.load(json_file)

class ExamplesGenerator:
    def __init__(self, dataset_name='nthngdy/oscar-mini', dataset_subset_name=''):
        # the dataset we choose to extract sentences from:
        self.dataset_name = dataset_name
        # the dataset subset name (if we use a subset)
        self.dataset_subset_name = dataset_subset_name
        # all the dataset as a sentences list
        self.sentences_list=[]
        # a list of the n most freq sentences
        self.n_most_freq=[]
        # dict from spesific word and location to all of the sentences it apear in
        # etc. self.word_to_senteces['because'][2] will give us a list of all the sentences with the word 'because' in the 12 position. 
        self.word_to_senteces={} 

        # minimum and maximum number of tokens in saved sentences:
        self.min_word_in_sen = 2
        self.max_word_in_sen = 32
        self.max_tok_in_sen = 64
        #load the dataset itself:
        if dataset_subset_name == '':
            self.data = load_dataset(dataset_name)
        else:
            self.data = load_dataset(dataset_name,dataset_subset_name)
    

    # takes the text in the datasets and splites it to sentences, according to the length conditions:
    def split_to_sentences(self):
        nltk.download('punkt')
        for example in self.data['train']:
            example_sentences = nltk.tokenize.sent_tokenize(example['text'])
            #filter sentences according to condition: (#TODO: might be a more efficient way to do this in the tokenize min_word_in_sen, and not splitting it to two interations)
            example_sentences = [sentence for sentence in example_sentences if len(sentence.split(' ')) < self.max_word_in_sen and len(sentence.split(' ')) > self.min_word_in_sen ]
            self.sentences_list += example_sentences
        ExamplesGenerator.print_in_format(f"Finished generating exampels, you have generated {len(self.sentences_list)} sentences")

    #returning wierd words - to ask nitay and eyal
    def tokenize_sentences(self):
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        tokens = [tokenizer.convert_ids_to_tokens(_ids) for _ids in tokenizer(self.sentences_list)['input_ids'] if len(tokenizer.convert_ids_to_tokens(_ids))<=self.max_tok_in_sen]
        self.sentences_list = tokens
        print('finished tokenize sentences')


    # return the n most frequent words in a the dataset sentences    
    def find_n_to_k_most_frequent_for_lists(self,n,k):
        list_of_tokens=list(itertools.chain.from_iterable(self.sentences_list))
        counter = Counter(list_of_tokens)
        
        self.n_most_freq = counter.most_common(n)
        self.n_most_freq = self.n_most_freq[k:]
        ExamplesGenerator.print_in_format(f"{n} most frequent strings are: {self.n_most_freq}")


    # return the n most frequent words in a the dataset sentences    
    def find_n_to_k_most_frequent_for_strings(self,n,k):
        #concatenate string to use split easily
        concatenated_string=' '.join(self.sentences_list)
        # split to words list for Counter
        words_list=concatenated_string.split()
        counter = Counter(words_list)
        
        self.n_most_freq = counter.most_common(n)
        self.n_most_freq = self.n_most_freq[k:]
        ExamplesGenerator.print_in_format(f"{n} most frequent strings are: {self.n_most_freq}")

    def create_dict_of_most_common(self):

        for i,sentence in enumerate(self.sentences_list):
            #check for each common word if its in the sentece:
            for c_tup in self.n_most_freq:
                c_word = c_tup[0]
                word_indexes = find_all_in_list(c_word, sentence)
                # iterate over all the word appearances in the sen:
                for index in word_indexes: 
                    if c_word in self.word_to_senteces:
                        if index in self.word_to_senteces[c_word]:
                            self.word_to_senteces[c_word][index].append(i)
                        else:
                            #needs to create a list to the sentences in this index:
                            self.word_to_senteces[c_word][index]=[i]
                    else:
                        #needs to create a new dict for the word index
                        self.word_to_senteces[c_word]={index:[i]}

    def create_json_from_most_common(self,n,k):
        self.split_to_sentences()
        self.tokenize_sentences()
        write_dict_to_json(self.sentences_list,'sentences_list')
        self.find_n_to_k_most_frequent_for_lists(n,k)
        self.print_in_format('writing n_to_k_most_frq:')
        write_dict_to_json(self.n_most_freq,'n_most_frq')
        self.print_in_format('writing word_to_sen_dict:')
        self.create_dict_of_most_common()
        write_dict_to_json(self.word_to_senteces,'word_to_sen_dict')


    @staticmethod
    def print_in_format(string_to_print):
        print(f" ------- {string_to_print} -------")




if __name__ == "__main__":
    examples_generator = ExamplesGenerator(dataset_name='nthngdy/oscar-mini', dataset_subset_name='unshuffled_original_en')
    examples_generator.create_json_from_most_common(900,450)

# TODO:
# in n_most_freq - find out how to remove all the 'unwanted' words (and if they are indeed unwanted) - e.g., the, and, to.. 
