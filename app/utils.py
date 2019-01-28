

import numpy as np
import json
import os

def load_SQuAD_data(filename,SQuAD_foldername='SQuAD',verbose=False):
    import json
    import os

    # Set up path info
    repodir = os.path.join(os.getenv("HOME"),'src','animated-succotash')
    datadir = os.path.join(repodir,'data')
    squaddir = os.path.join(datadir,SQuAD_foldername)

    fullname = os.path.join(squaddir,filename)
    if verbose: print(fullname)

    # Load the data
    json_data=open(fullname,'r')
    data = json.load(json_data)
    json_data.close()

    arts = data['data']

    if verbose:
        # What's in data?
        print("Version: " + data['version'])

        # All articles
        Narticles = len(arts)
        print ("Narticles = " +  str(Narticles))

        # List all article titles
        print ("Narticles = " +  str(Narticles))
        for ind in range(Narticles):
            art = arts[ind]
            print(art['title'])

    return arts

def save_SQuAD_data(arts,filename,SQuAD_foldername='SQuAD_postprocessed',verbose=False):
    import json
    import os

    # Set up path info
    repodir = os.path.join(os.getenv("HOME"),'src','animated-succotash')
    datadir = os.path.join(repodir,'data')
    squaddir = os.path.join(datadir,SQuAD_foldername)

    # Create directory if doesn't already exist
    if not os.path.exists(squaddir):
        print("Directory " + squaddir + " doesn't exist. Creating..")
        os.makedirs(squaddir)

    fullname = os.path.join(squaddir,filename)
    if verbose: print(fullname)

    data = {'data': arts, 'version':['v2.0']}

    # Save the data
    with open(fullname, 'w') as outfile:
        json.dump(data, outfile)


# # # # # Loading data # # # # #
# Raw data
def load_SQuAD_train(filename='train-v2.0.json',verbose=False):
    return load_SQuAD_data(filename,'SQuAD',verbose)

def load_SQuAD_dev(filename='dev-v2.0.json',verbose=False):
    return load_SQuAD_data(filename,'SQuAD',verbose)

# Postprocessed data
def load_SQuAD_train_pp(filename='train-v2.0.json',verbose=False):
    # Load data from post processing folder
    return load_SQuAD_data(filename,'SQuAD_postprocessed',verbose)

def load_SQuAD_dev_pp(filename='dev-v2.0.json',verbose=False):
    # Load data from post processing folder
    return load_SQuAD_data(filename,'SQuAD_postprocessed',verbose)


# # # # # Saving data # # # # #
def save_SQuAD_train_pp(arts,filename='train-v2.0.json',verbose=False):
    save_SQuAD_data(arts,filename,'SQuAD_postprocessed',verbose)

def save_SQuAD_dev_pp(arts,filename='dev-v2.0.json',verbose=False):
    save_SQuAD_data(arts,filename,'SQuAD_postprocessed',verbose)
