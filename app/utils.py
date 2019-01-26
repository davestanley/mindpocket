

import numpy as np
import json
import os

def load_SQuAD_data(filename,verbose=False):
    import json
    import os

    # Set up path info
    repodir = os.path.join(os.getenv("HOME"),'src','animated-succotash')
    datadir = os.path.join(repodir,'data')
    squaddir = os.path.join(datadir,'SQuAD')

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


def load_SQuAD_train(verbose=False):
    return load_SQuAD_data('train-v2.0.json',verbose)


def load_SQuAD_dev(verbose=False):
    return load_SQuAD_data('dev-v2.0.json',verbose)
