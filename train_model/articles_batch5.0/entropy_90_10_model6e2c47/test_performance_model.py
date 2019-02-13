
# Script for testing model performance. Should do the same as
# test_performance_model.ipynb, but also just display averages

# Set up and load data
# Includes
import sys
import os
import numpy as np
import json
import os


# Setup paths containing utility
curr_folder = os.getcwd()
sys.path.insert(0, os.path.join(curr_folder,'../../../app'))

# Utils imports for loading data
from utils import save_data, load_data, exists_datafolder
from utils import load_SQuAD_train, load_SQuAD_dev
from utils import get_foldername
from utils_NLP import text2sentences,words2words_blanked,words2answers
from utils_NLP import words2text
from utils_SQuAD import OR_arts_paragraph_fields,merge_arts_paragraph_fields

# Plotting includes
from utils_EDAplots import plotbar_train_dev,plothist_train_dev,plotbar_train_dev2,plothist_train_dev2

# Stats saving stuff
from utils_EDA import calcstats_train_dev

# Import fig stuff
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# # # # # # # # # # BEGIN MASSIVE FUNCTION THAT RUNS EVERYTHING # # # # # # # # # #

def run_full_analysis(resultsfname,merge_in_NER_data):
    # Option for merging NER data into combined model
    # merge_in_NER_data = False

    # Load data containing MODEL (predictions)
    foldername = os.path.join('SQ_pp_b4m0c2')
    arts_train = load_data('train.json',foldername,prepend_data_folder=False)
    arts_dev = load_data('dev.json',foldername,prepend_data_folder=False)

    # All articles
    Ntrain = len(arts_train)
    Ndev = len(arts_dev)

    arts = arts_train + arts_dev
    print(arts[1]['title'])
    # print(arts[1]['paragraphs'][0]['context'])

    # Trim down newly loaded articles to match Narticles in training set
    ind_train = slice(0,Ntrain)
    ind_dev = slice(0,Ndev)

    # Chosen display articles
    ind_ex_train = 1                   # Example from training set - Chopin
    ind_ex_dev = Ntrain + (467-442)    # Example from dev set - Immune system

    # Load blanks data (ground truth)
    foldername = get_foldername('sq_pp_training')
    arts3 = load_data('train.json',foldername)[ind_train] + load_data('dev.json',foldername)[ind_dev]
    # print(arts3[1]['title'])
    # print(arts3[1]['paragraphs'][0]['context_blanked'])

    # Make sure all titles match
    all_title_pairs = [(a1['title'],a3['title']) for a1,a3 in zip(arts,arts3)]
    titles_match_bool = [a1['title'] == a3['title'] for a1,a3 in zip(arts,arts3)]
    print("Matching titles: {} \nTotal articles {}".format(sum(titles_match_bool),len(titles_match_bool)))
    if not sum(titles_match_bool) == len(titles_match_bool):
        raise ValueError('Articles mismatch.')

    # Merge ground truth blanks with original data to get full dataset
    list_of_fields = ['context_blanked','blank_classification']
    arts = merge_arts_paragraph_fields(arts,arts3,list_of_fields)

    # print(arts[1]['title'])
    # print(arts[1]['paragraphs'][0]['context'])
    # print(arts[1]['paragraphs'][0]['context_blanked'])

    # Convert AllenNLP Model blanks classification into standard format

    # If doing merge, use unique name for this model result. Otherwise, use generic name
    if merge_in_NER_data: fieldname = 'blank_classified_allenMODEL'
    else: fieldname = 'blank_classified_allen'


    from utils_NLP import allenNLP_classify_blanks
    arts = allenNLP_classify_blanks(arts,'0',fieldname)
    arts[0]['paragraphs'][0].keys()



    if merge_in_NER_data:
        # Load data containing NEP (predictions)
        foldername = get_foldername('sq_pp_ner')

        arts_NER = load_data('train.json',foldername)[ind_train] + load_data('dev.json',foldername)[ind_dev]
        print(arts[1]['title'])
        # print(arts[1]['paragraphs'][0]['context'])

        # Make sure all titles match
        all_title_pairs = [(a1['title'],a3['title']) for a1,a3 in zip(arts,arts_NER)]
        titles_match_bool = [a1['title'] == a3['title'] for a1,a3 in zip(arts,arts_NER)]
        print("Matching titles: {} \nTotal articles {}".format(sum(titles_match_bool),len(titles_match_bool)))
        if not sum(titles_match_bool) == len(titles_match_bool):
            raise ValueError('Articles mismatch.')

        # Convert AllenNLP Model blanks classification into standard format
        from utils_NLP import allenNLP_classify_blanks
        arts_NER = allenNLP_classify_blanks(arts_NER,'O','blank_classified_allenNER')
        print(arts_NER[0]['paragraphs'][0].keys())

        # Merge NER data into full dataset
        list_of_fields = ['blank_classified_allenNER']
        arts = merge_arts_paragraph_fields(arts,arts_NER,list_of_fields)
        print(arts[0]['paragraphs'][0].keys())

    # OR operation on blank_classified_allenMODEL and blank_classified_allenNER into blank_classified_allenMODEL

    if merge_in_NER_data:
        destination_fieldname = 'blank_classified_allen'
        arts = OR_arts_paragraph_fields(arts,['blank_classified_allenMODEL','blank_classified_allenNER'],destination_fieldname)

        p = arts[0]['paragraphs'][1]
        print(p['blank_classified_allenNER'])
        print(p['blank_classified_allenMODEL'])
        print(p['blank_classified_allen'])


    # Initialize stuff
    TPR0 = []
    FPR0 = []
    ACC0 = []
    Nsentences0 = []
    TP0 = []
    FP0 = []
    FN0 = []
    TN0 = []
    TPpersent0 = []
    FPpersent0 = []
    abads = []            # Article-level bads
    sbc0 = []
    st0 = []
    Nwords0 = []

    art = arts[:]

    i=-1
    for a in art:
        i=i+1
        # AllenNLP results
        words = [w for p in a['paragraphs'] for w in p['allenNER']['words'].split()]
    #     tags = [t for p in a['paragraphs'] for t in p['allenNER']['tags'].split()]
    #     tags = [not t == '0' for t in tags]   # Convert to binary
        tags = [t for p in a['paragraphs'] for t in p['blank_classified_allen']]

        # Ground truth
        blank_classification = [bc for p in a['paragraphs'] for bc in p['blank_classification']]
        blank_classification = [b == 1 for b in blank_classification] # Convert to binary

        Nsentences2 = len(text2sentences(words2text(words)))

        sbc = sum(blank_classification)
        st = sum(tags)
        if sbc == 0 or st == 0:
            print("Warning article {} contains {} ground truth blanks and {} tags. Likely bad".format(str(i),str(sbc),str(st)))

            # Make up some dummy values so don't confuse for a REAL outlier in plots. Should just drop this data in the future
            # This is ok because we'll skip them later if want to do stats - that's what abads is for
            TPR = 0.0
            FPR = 0.0
            ACC = 0.0
            TP = 100
            FP = 100
            FN = 100
            TN = 100

            TPpersent = 1
            FPpersent = 1
            abads.append(i)
        else:
            TP = sum([b and t for b,t in zip(blank_classification,tags)])
            FP = sum([not b and t for b,t in zip(blank_classification,tags)])
            FN = sum([b and not t for b,t in zip(blank_classification,tags)])
            TN = sum([not b and not t for b,t in zip(blank_classification,tags)])
            ACC = (TP+TN)/(TP+FP+FN+TN)
            ACC2 = sum([b == t for b,t in zip(blank_classification,tags)]) / len(tags)

            # Sensitivity, hit rate, recall, or true positive rate
            TPR = TP/(TP+FN)
            # Specificity or true negative rate
            TNR = TN/(TN+FP)
            # Precision or positive predictive value
            PPV = TP/(TP+FP)
            # Negative predictive value
            NPV = TN/(TN+FN)
            # Fall out or false positive rate
            FPR = FP/(FP+TN)
            # False negative rate
            FNR = FN/(TP+FN)
            # False discovery rate
            FDR = FP/(TP+FP)

            # Per sententance values
            TPpersent = TP / Nsentences2
            FPpersent = FP / Nsentences2

        TPR0.append(TPR)
        FPR0.append(FPR)
        ACC0.append(ACC)
        TP0.append(TP)
        FP0.append(FP)
        FN0.append(FN)
        TN0.append(TN)
        TPpersent0.append(TPpersent)
        FPpersent0.append(FPpersent)
        sbc0.append(sbc)
        st0.append(st)
        Nwords0.append(len(tags))




    # Print tiles of bad articles
    for ab in abads:
        print(art[ab]['title'])

    # Calculate how this affects Ntrain / Ndev
    Ntrain_bad = len([b for b in abads if b < Ntrain])
    Ndev_bad = len([b for b in abads if b >= Ntrain])
    print('Ntrain={}'.format(str(Ntrain)))
    print('Ndev={}'.format(str(Ndev)))
    print('Ntrain_bad={}'.format(str(Ntrain_bad)))
    print('Ndev_bad={}'.format(str(Ndev_bad)))



    # Calculate all statistics and save
    # resultsfname = 'results.json'

    # Remove the file if it already exists, since we will be appending below
    if os.path.isfile(resultsfname):
        os.remove(resultsfname)

    # Calculate stats
        # Scale up by factor of 100 for easier reading

    # Write title
    with open(resultsfname, 'a') as f:
        f.write('\"==' + "New drop bads" + '==\"\n')

    # ax = plothist_train_dev2(myvar2,Ntrain-Ntrain_bad,Ndev-Ndev_bad,xlabel=varname,ylabel='N Articles',devbins='auto')
    # This excludes the bad files.... but unfortunately i didn't figure out how to separate testing and dev (x-validation) sets
    myvar = TPR0; myvar2 = [v for i, v in enumerate(myvar) if i not in abads]; out = calcstats_train_dev([x*100 for x in myvar2],resultsfname,Ntrain=Ntrain-Ntrain_bad,Ndev=Ndev-Ndev_bad,Ntest=None,mytitle='TPR')
    myvar = FPR0; myvar2 = [v for i, v in enumerate(myvar) if i not in abads]; out = calcstats_train_dev([x*100 for x in myvar2],resultsfname,Ntrain=Ntrain-Ntrain_bad,Ndev=Ndev-Ndev_bad,Ntest=None,mytitle='FPR')
    myvar = ACC0; myvar2 = [v for i, v in enumerate(myvar) if i not in abads]; out = calcstats_train_dev([x*100 for x in myvar2],resultsfname,Ntrain=Ntrain-Ntrain_bad,Ndev=Ndev-Ndev_bad,Ntest=None,mytitle='ACC')
    myvar = sbc0; myvar2 = [v for i, v in enumerate(myvar) if i not in abads]; out = calcstats_train_dev([x*100 for x in myvar2],resultsfname,Ntrain=Ntrain-Ntrain_bad,Ndev=Ndev-Ndev_bad,Ntest=None,mytitle='Trues')
    myvar = st0; myvar2 = [v for i, v in enumerate(myvar) if i not in abads]; out = calcstats_train_dev([x*100 for x in myvar2],resultsfname,Ntrain=Ntrain-Ntrain_bad,Ndev=Ndev-Ndev_bad,Ntest=None,mytitle='Positives')
    myvar = FPR0; myvar2 = [v for i, v in enumerate(myvar) if i not in abads]; out = calcstats_train_dev([x*100 for x in myvar2],resultsfname,Ntrain=Ntrain-Ntrain_bad,Ndev=Ndev-Ndev_bad,Ntest=None,mytitle='FPR')


    # Write title
    with open(resultsfname, 'a') as f:
        f.write('\"==' + "Original" + '==\"\n')
    # OLD
    out = calcstats_train_dev([x*100 for x in TPR0],resultsfname,Ntrain=Ntrain,Ndev=Ndev-15,Ntest=15,mytitle='TPR')
    out = calcstats_train_dev([x*100 for x in FPR0],resultsfname,Ntrain=Ntrain,Ndev=Ndev-15,Ntest=15,mytitle='FPR')
    out = calcstats_train_dev([x*100 for x in ACC0],resultsfname,Ntrain=Ntrain,Ndev=Ndev-15,Ntest=15,mytitle='ACC0')
    out = calcstats_train_dev(sbc0,resultsfname,Ntrain=Ntrain,Ndev=Ndev-15,Ntest=15,mytitle='Trues')             # Sum of all true values in each article (e.g., ground truth true blanks)
    out = calcstats_train_dev(st0,resultsfname,Ntrain=Ntrain,Ndev=Ndev-15,Ntest=15,mytitle='Positives')        # Sum of all positive IDs in each article (tags)
    out = calcstats_train_dev(TP0,resultsfname,Ntrain=Ntrain,Ndev=Ndev-15,Ntest=15,mytitle='TP0')
    out = calcstats_train_dev(FP0,resultsfname,Ntrain=Ntrain,Ndev=Ndev-15,Ntest=15,mytitle='FP0')
    out = calcstats_train_dev(FN0,resultsfname,Ntrain=Ntrain,Ndev=Ndev-15,Ntest=15,mytitle='FN0')
    out = calcstats_train_dev(TN0,resultsfname,Ntrain=Ntrain,Ndev=Ndev-15,Ntest=15,mytitle='TN0')
    out = calcstats_train_dev(TPpersent0,resultsfname,Ntrain=Ntrain,Ndev=Ndev-15,Ntest=15,mytitle='TPpersent0')
    out = calcstats_train_dev(FPpersent0,resultsfname,Ntrain=Ntrain,Ndev=Ndev-15,Ntest=15,mytitle='FPpersent0')
# # # # # # # # # # END MASSIVE FUNCTION  # # # # # # # # # #



run_full_analysis(resultsfname='results1_mymodel.json',merge_in_NER_data=False)
run_full_analysis(resultsfname='results2_mymodel_and_NER.json',merge_in_NER_data=True)
