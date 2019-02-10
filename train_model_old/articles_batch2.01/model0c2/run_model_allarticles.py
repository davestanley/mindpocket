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
sys.path.insert(0, os.path.join(curr_folder,'../../../app'))

# Import custom utilities
from utils import save_data, load_data, exists_datafolder
from utils import load_SQuAD_train, load_SQuAD_dev
from utils import get_foldername
from utils_allenNLP import run_predictor

# Set up folder names
foldername = os.path.join('SQ_pp_b4m0c2')

# Flags
verbose_on = True       # Verbose comments
verbose2_on = False      # Detailed verbose comments - show results of NLP
testing_mode = False
skip_save = False

# Set up AllenNLP
currmodel = os.path.join('.','model.tar.gz')
if not testing_mode: predictor = Predictor.from_path(currmodel)

# # # # # # # # # # # # # # # # # # # # Process training data # # # # # # # # # # # # # # # # #
# Load the training data
arts = load_SQuAD_train()
art = arts
# art = arts[105:107]         # A few short articles
run_predictor(art,predictor,foldername,'train',testing_mode=False,skip_save=False,prepend_data_folder=False)



# # # # # # # # # # # # # # # # # # # # Process DEV data # # # # # # # # # # # # # # # # #
# Load the dev data
arts = load_SQuAD_dev()
art = arts
run_predictor(art,predictor,foldername,'dev',testing_mode=False,skip_save=False,prepend_data_folder=False)
