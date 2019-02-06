
# Mainly file IO utilities

import numpy as np
import json
import os

# # # # # Path information # # # # # # #
def get_foldername (mystring):
    if mystring == 'sq':
        foldername = 'SQuAD'    # Original SQuAD dataset
    if mystring == 'sq_pp_ner':
        foldername = 'SQuAD_pp_NER'
    if mystring == 'sq_pp_training':            # Ground truth
        foldername = 'SQuAD_pp_trainingblanks'
    return foldername



# # # # # Core File IO functions # # # # # # #
def get_data_root():
    ''' Gets the root folder for storing all data '''
    import os

    # Set up path info
    repodir = os.path.join(os.getenv("HOME"),'src','mindpocket')
    datadir = os.path.join(repodir,'data')

    return datadir


def load_data(filename,foldername='SQuAD',verbose=False,prepend_data_folder=True):
    import json
    import os

    datadir = get_data_root()
    if prepend_data_folder: squaddir = os.path.join(datadir,foldername)
    else: squaddir = foldername

    fullname = os.path.join(squaddir,filename)
    if verbose: print(fullname)

    # Load the data
    with open(fullname,'r') as f:
        data = json.load(f)

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

def save_data(arts,filename,foldername='SQuAD_postprocessed',verbose=False,do_overwrite=False,prepend_data_folder=True):
    # prepend_data_folder - Adds ~/src/mindpocket/data prefix to squaddir folder name
    import json
    import os

    # Set up path info
    datadir = get_data_root()
    if prepend_data_folder: squaddir = os.path.join(datadir,foldername)
    else: squaddir = foldername

    # Create directory if doesn't already exist
    if not os.path.exists(squaddir):
        print("Directory " + squaddir + " doesn't exist. Creating..")
        os.makedirs(squaddir)

    fullname = os.path.join(squaddir,filename)
    if verbose: print('Saving' + fullname)

    # Check if path exists. If exists, and overwrite is false, return
    if os.path.exists(fullname):
        file_exists = True
        if not do_overwrite:
            # Break out of code if we're not overwriting
            print("File " + fullname + " exists...skipping.")
            return file_exists
        else:
            print("File " + fullname + " exists...overwriting.")
    else:
        file_exists = False

    # Save the data
    data = {'data': arts, 'version':['v2.0']}
    with open(fullname, 'w') as f:
        json.dump(data, f)

    return file_exists


# # # # # # Merge individual art files into single large json # # # # # #
def merge_artfiles(filename_prefix,foldername,outname,verbose = False,do_overwrite=False,prepend_data_folder=True):
    # prepend_data_folder - Adds ~/src/mindpocket/data prefix to squaddir folder name
    import os
    import glob
    datadir = get_data_root()
    if prepend_data_folder: squaddir = os.path.join(datadir,foldername)
    else: squaddir = foldername
    searchterm = os.path.join(squaddir,filename_prefix)
    if verbose: print("Searching" + searchterm)
    allfiles = glob.glob(searchterm)
    allfiles.sort()         # Need to sort because for some reason, glob returns in a random order
    Nfiles = len(allfiles)

    # Start looping through and loading articles
    arts = []
    # print(Nfiles)
    for i in range(Nfiles):
        # Load the data
        #fullname = os.path.join(squaddir,allfiles[i])
        # print(fullname)
        with open(allfiles[i],'r') as f:
            data = json.load(f)
        art = data['data']
        arts.append(art)

    # Save the merged arts
    # fullname = os.path.join(squaddir,outname)
    # print("Saving merged articles to " + fullname)
    save_data(arts,outname,foldername,verbose,do_overwrite,prepend_data_folder)
    return(arts)


# # # # # Test if file exists # # # # #
def exists_datafolder(filename,foldername='SQuAD_postprocessed',prepend_data_folder=True):
    # prepend_data_folder - Adds ~/src/mindpocket/data prefix to squaddir folder name

    import os

    datadir = get_data_root()
    if prepend_data_folder: squaddir = os.path.join(datadir,foldername)
    else: squaddir = foldername
    fullname = os.path.join(squaddir,filename)

    if os.path.exists(fullname):
        file_exists = True
    else: file_exists = False

    return file_exists



# # # # # Loading data # # # # #
# Raw data
def load_SQuAD_train(filename='train-v2.0.json',verbose=False):
    return load_data(filename,get_foldername('sq'),verbose)

def load_SQuAD_dev(filename='dev-v2.0.json',verbose=False):
    return load_data(filename,get_foldername('sq'),verbose)

# # Postprocessed data
# def load_SQuAD_train_pp(filename='train-v2.0.json',verbose=False):
#     # Load data from post processing folder
#     return load_data(filename,'SQuAD_postprocessed',verbose)
#
# def load_SQuAD_dev_pp(filename='dev-v2.0.json',verbose=False):
#     # Load data from post processing folder
#     return load_data(filename,'SQuAD_postprocessed',verbose)
#
#
# # # # # # Saving data # # # # #
# def save_SQuAD_train_pp(arts,filename='train-v2.0.json',verbose=False,do_overwrite=False):
#     file_exists = save_data(arts,filename,'SQuAD_postprocessed',verbose,do_overwrite)
#     return file_exists
#
# def save_SQuAD_dev_pp(arts,filename='dev-v2.0.json',verbose=False,do_overwrite=False):
#     file_exists = save_data(arts,filename,'SQuAD_postprocessed',verbose,do_overwrite)
#     return file_exists
