from json import load
from xml.etree.ElementPath import find
import numpy as np
import torch
from datasets import load_dataset
import nltk
from collections import Counter



def find_all(word, sent):
    indexex=[]
    words_list = sent.split()
    for i,w in enumerate(words_list):
        if w ==word:
            indexex.append(i)
    return indexex


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
        self.min_tok_in_sen = 2
        self.max_tok_in_sen = 32

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
            #filter sentences according to condition: (#TODO: might be a more efficient way to do this in the tokenize iteration, and not splitting it to two interations)
            example_sentences = [sentence for sentence in example_sentences if len(sentence.split(' ')) < self.max_tok_in_sen and len(sentence.split(' ')) > self.min_tok_in_sen ]
            self.sentences_list += example_sentences
        ExamplesGenerator.print_in_format(f"Finished generating exampels, you have generated {len(self.sentences_list)} sentences")

    # return the n most frequent words in a the dataset sentences    
    def find_n_most_frequent(self,n):
        #concatenate string to use split easily
        concatenated_string=' '.join(self.sentences_list)
        # split to words list for Counter
        words_list=concatenated_string.split()
        counter = Counter(words_list)
        
        self.n_most_freq = counter.most_common(n)
        ExamplesGenerator.print_in_format(f"{n} most frequent strings are: {self.n_most_freq}")

    def create_dict_of_most_common(self):

        for sentence in self.sentences_list:
            #check for each common word if its in the sentece:
            for c_tup in self.n_most_freq:
                c_word = c_tup[0]
                word_indexes = find_all(c_word, sentence)
                # iterate over all the word appearances in the sen:
                for index in word_indexes: 
                    if c_word in self.word_to_senteces and index in self.word_to_senteces[c_word]:
                            self.word_to_senteces[c_word][index].append(sentence)
                            continue
                    #needs to create a new dict for the word index
                    self.word_to_senteces[c_word]={index:[sentence]}
                    #self.word_to_senteces[c_word][index]=sentence
                    

    @staticmethod
    def print_in_format(string_to_print):
        print(f" ------- {string_to_print} -------")




if __name__ == "__main__":
    examples_generator = ExamplesGenerator(dataset_name='nthngdy/oscar-mini', dataset_subset_name='unshuffled_original_en')
    examples_generator.split_to_sentences()
    print(examples_generator.find_n_most_frequent(20))
    examples_generator.create_dict_of_most_common()
    examples_generator.print_in_format('finished')


# TODO:
# in n_most_freq - find out how to remove all the 'unwanted' words (and if they are indeed unwanted) - e.g., the, and, to.. 
