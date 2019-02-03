
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # Define supporting functions for preparing results list for writing to disk # # # # # # # # # # #
def prepare_results_by_sentences(results,token_spacer,word_spacer):
    '''Returns a list of word-tag pairs, one for each sentence'''
    from utils_NLP import splitsentences_allenResults
    results_list = splitsentences_allenResults(results)
    Nsentences = len(results_list)

    to_write = []
    for r in results_list:
        sentence = ''
        for i in range(len(r['words'])):
            sentence = sentence + r['words'][i] + token_spacer + str(r['tags'][i]) + word_spacer
        to_write.append(sentence)

    return to_write

def prepare_results_by_sentences_forwardbackwards(results,token_spacer,word_spacer):
    '''Returns a list of word-tag pairs, one for each sentence'''
    from utils_NLP import splitsentences_allenResults

    def do_mirror(s):
        '''transforms s=[a,b,c] into s2[a,b,c,c,b,a] '''
        s2 = s.copy()
        s2.reverse()
        return s + s2

    results_list = splitsentences_allenResults(results)
    Nsentences = len(results_list)

    to_write = []
    for r in results_list:
        # Transform r into r + its mirror
        r['words'] = do_mirror(r['words'])
        r['tags'] = do_mirror(r['tags'])

        # Start building sentences as before
        sentence = ''
        for i in range(len(r['words'])):
            sentence = sentence + r['words'][i] + token_spacer + str(r['tags'][i]) + word_spacer
        to_write.append(sentence)

    return to_write

def prepare_results_by_paragraph(r,token_spacer,word_spacer):
    '''Returns a single list of word-tag pairs'''

    sentence = ''
    for i in range(len(r['words'])):
        sentence = sentence + r['words'][i] + token_spacer + str(r['tags'][i]) + word_spacer
    to_write = [sentence]
    return to_write
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Set flags for code
newline_method = 1     # Method for defining new lines in text file_exists
                       # 1 - Each sentence is a new line
                       # 2 - Repeat each sentence forward and backwards, and then make this a new line
                       # 3 - Each paragraph is a new line

# Set up and load data
# Includes
import sys
import os
import numpy as np
import json
import os


# Setup paths containing utility
curr_folder = os.getcwd()
sys.path.insert(0, os.path.join(curr_folder,'../app'))

# Utils imports for loading data
from utils import save_data, load_data, exists_datafolder,get_data_root
from utils import load_SQuAD_train, load_SQuAD_dev
from utils import get_foldername
from utils import merge_artfiles

# Allen NLP processing import
from utils_NLP import allenNLP_split_words, splitsentences_allenResults


# Load the training data
arts_train = load_SQuAD_train()

# Load the testing data
arts_dev = load_SQuAD_dev()

# Merge all articles together
Ntrain = len(arts_train)
Ndev = len(arts_dev)
print ("Narticles in train = " +  str(len(arts_train)))
print ("Narticles in dev = " +  str(len(arts_dev)))

# Merge arts together
arts= arts_train + arts_dev
Narticles = len(arts)

# Load blanks data (ground truth)
foldername = get_foldername('sq_pp_training')
arts3 = load_data('train.json',foldername) + load_data('dev.json',foldername)

# Make sure all titles match
all_title_pairs = [(a1['title'],a3['title']) for a1,a3 in zip(arts,arts3)]
titles_match_bool = [a1['title'] == a3['title'] for a1,a3 in zip(arts,arts3)]
print("Matching titles: {} \nTotal articles {}".format(sum(titles_match_bool),len(titles_match_bool)))

# Merge ground truth blanks with original data to get full dataset
from utils_SQuAD import merge_arts_paragraph_fields
list_of_fields = ['context_blanked','blank_classification']
arts = merge_arts_paragraph_fields(arts,arts3,list_of_fields)

# Choose a subset of articles
inds = [0,4,6]
art = [arts[i] for i in inds]


# Save in same folder as loaded from
foldername = get_foldername('sq_pp_training')

# # Pick out sample paragraph (testing)
# a = art[0]
# p = a['paragraphs'][0]

# Write blank char to file to clear
with open(os.path.join(get_data_root(),foldername,'allenTrain.txt'), 'w') as f:
    f.write('')

# Start writing new text
for a in art:
    for p in a['paragraphs']:

        context_split = allenNLP_split_words(p['context'])
        bc = p['blank_classification']
        if not len(context_split) == len(bc):
            print('Warning - Mismatch between # words and labels')

        # Format as allenNLP results list
        results = {'words': context_split,
                   'tags': bc}


        token_spacer = "//"
        word_spacer = ' '

        if newline_method == 1: to_write = prepare_results_by_sentences(results,token_spacer,word_spacer);
        elif newline_method == 2: to_write = prepare_results_by_sentences_forwardbackwards(results,token_spacer,word_spacer);
        elif newline_method == 3: to_write = prepare_results_by_paragraph(results,token_spacer,word_spacer)
        else: print("Unknown newline method");



        with open(os.path.join(get_data_root(),foldername,'allenTrain.txt'), 'a') as f:
            for w in to_write:
                f.write(w + '\n')
