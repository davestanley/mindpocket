

import numpy as np
import json
import os


def load_SQuAD_train(verbose=False):
    import json
    import os


    # Set up path info
    repodir = os.path.join(os.getenv("HOME"),'src','animated-succotash')
    datadir = os.path.join(repodir,'data')
    squaddir = os.path.join(datadir,'SQuAD')

    filename = os.path.join(squaddir,'train-v2.0.json')

    # Load the data
    json_data=open(filename,'r')
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
