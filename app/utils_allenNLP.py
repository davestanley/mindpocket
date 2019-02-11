# Utilities for running and testing AllenNLP code

def run_predictor(art,predictor,foldername,filename_prefix,testing_mode=False,skip_save=False,prepend_data_folder=True):
    # prepend_data_folder - Adds ~/src/mindpocket/data prefix to squaddir folder name
    from utils import save_data, load_data, exists_datafolder
    from utils import merge_artfiles

    verbose2_on=False

    # Loop through and add results field to data
    art2 = art.copy()
    for i,a in enumerate(art):
        filename = filename_prefix + '_art_' + str(i).zfill(3) + '.json'

        # Do a short test to see if file exists
        file_exists = exists_datafolder(filename,foldername,prepend_data_folder)
        if file_exists:
            print("File: " + filename + " already exists. Skipping...")
            continue        # If file already exists, skip over to the next file

        # Otherwise with operation
        print("Article number:" + str(i).zfill(3) + ". Saving to: " + filename)
        for j,p in enumerate(a['paragraphs']):
            if verbose2_on:
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
        if not skip_save: save_data(art2[i],filename,foldername,[],[],prepend_data_folder)

    # Once all individual files have been saved, merge into 1 large json file
    merge_artfiles(filename_prefix + '_art_*',foldername,filename_prefix + '.json',verbose=True,do_overwrite=[],prepend_data_folder=prepend_data_folder)
