# This file preprocesses the original SQuAD data by running AllenNLP
# named entity recognition on all articles

# Set up and load data
# Includes
import sys
import os
import numpy as np
import json
import os

# Import Allen
from allennlp.predictors import Predictor

# Setup paths containing utility
curr_folder = os.getcwd()
sys.path.insert(0, os.path.join(curr_folder,'../app'))

# Import custom utilities
from utils import load_SQuAD_train, load_SQuAD_dev, exists_SQuAD_pp
from utils import load_SQuAD_train_pp, load_SQuAD_dev_pp
from utils import save_SQuAD_train_pp, save_SQuAD_dev_pp

# Flags
verbose_on = True       # Verbose comments
verbose2_on = False      # Detailed verbose comments - show results of NLP
testing_mode = False

# Set up AllenNLP
allenNERmodel = os.path.join(os.getenv("HOME"),'src','allennlp','ner-model-2018.12.18.tar.gz')
if not testing_mode: predictor = Predictor.from_path(allenNERmodel)


# Load the training data
arts = load_SQuAD_train()

art = arts
# art = arts[105:107]         # A few short articles


# Loop through and add results field to data
art2 = art.copy()
for i,a in enumerate(art):
    filename = 'train_art_' + str(i).zfill(3) + '.json'

    # Do a short test to see if file exists
    file_exists = exists_SQuAD_pp(filename)
    if file_exists:
        print("File: " + filename + " already exists. Skipping...")
        continue        # If file already exists, skip over to the next file

    # Otherwise with operation
    print("Article number:" + str(i).zfill(3) + ". Saving to: " + filename)
    for j,p in enumerate(a['paragraphs']):
        print("\tParagraph number: " + str(j))
        if not testing_mode:
            results = predictor.predict(sentence=p['context'])
            if verbose2_on:
                for word, tag in zip(results["words"], results["tags"]):
                    print(f"{word}\t{tag}")

            # Merge words and tags together into 1 long sentence, for more efficient json storage
            results2 = {
                    'words': ' '.join(results['words']),
                    'tags': ' '.join(results['tags']),
                }
        else:
            results = 'asdf'
            results2 = 'asdf'
        art2[i]['paragraphs'][j]['allenNER']=results2

    # Save individual articles
    save_SQuAD_train_pp(art2[i],filename)

# Save new data to json
save_SQuAD_train_pp(art2)

# Test loading data
arts3 = load_SQuAD_train_pp()

train_art_0 = load_SQuAD_train_pp('train_art_000.json')

# Testing
# save_SQuAD_train_pp(arts,filename='test.json',verbose=False,do_overwrite=False)



# # Import all
# # for i in range(len(arts)):
# out = []
# for i in range(105,107):
#     art = arts[i]
#     if verbose_on: print("Article #:" + str(i))
#     out.append({'paragraph':[]})
#     for j in range(len(art['paragraphs']):
#         p = art['paragraphs'][j]
#         if verbose_on: print("Paragraph #:" + str(j))
#         out[i]['paragraph'].append
#         for k in range(len(p['qas']):
#             if verbose_on: print("Question #:" + str(k))
#             q = p['qas'][k]['question']


# out = []
# art2 = art.copy()
# [art2[i]['paragraphs'][j]['qas'][k]['b']='asdf' for i,a in enumerate(art2) for j,p in enumerate(a['paragraphs']) for k,q in enumerate(p['qas'])]



# for i,a in enumerate(art):
#     print(i)
#     print(art['paragraph'][0])
#
if False:
    # Set up and test AllenNLP
    predictor = Predictor.from_path("/home/davestanley/src/allennlp/ner-model-2018.12.18.tar.gz")
    results = predictor.predict(sentence="Did Uriah honestly think he could beat The Legend of Zelda in under three hours?")
    for word, tag in zip(results["words"], results["tags"]):
        print(f"{word}\t{tag}")
