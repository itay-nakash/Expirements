from json import load
import numpy as np
import torch
from datasets import load_dataset
import nltk
from collections import Counter



class ExamplesGenerator:
    def __init__(self, dataset_name='nthngdy/oscar-mini', dataset_subset_name=''):
        # the dataset we choose to extract sentences from:
        self.dataset_name = dataset_name
        self.dataset_subset_name = dataset_subset_name
        self.sentences_list=[]
        self.n_most_freq=[]

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
        concatenated_string=' '.join(self.sentences_list)
        words_list=concatenated_string.split()
        counter = Counter(words_list)
        
        self.n_most_freq = counter.most_common(n)
        ExamplesGenerator.print_in_format(f"{n} most frequent strings are: {self.n_most_freq}")

    @staticmethod
    def print_in_format(string_to_print):
        print(f" ------- {string_to_print} -------")




if __name__ == "__main__":
    examples_generator = ExamplesGenerator(dataset_name='nthngdy/oscar-mini', dataset_subset_name='unshuffled_original_en')
    examples_generator.split_to_sentences()
    print(examples_generator.find_n_most_frequent(1000))



# TODO:
# in n_most_freq - find out how to remove all the 'unwanted' words (and if they are indeed unwanted) - e.g., the, and, to.. 
# Create dict with all the sentences each word is appeared in + the location it appears in