
# Utilities specifically for working with the SQuAD dataset

def classify_blanks_from_answers(art,maxWords_per_FITB=2,return_full=False,verbose_on=False,warning_level=1):
    """Creates keys context_blanked and blank_classification under paragraph, based on keywords present in answers"""
    # Inputs:
    # return_full = True / False - If true, returns a copy of art with new fields added.
    #        If false, just returns the new fields in an otherwise empty struct
    # Imports
    from utils_NLP import extract_no_stopwords, join_punctuation, allenNLP_split_words
    from copy import deepcopy

    # Local settings
    verbose_on2 = False # Set to true to display full context for each paragraph

    # Copy so don't overwrite original
    art2 = deepcopy(art)    # Direct copy of original art
    art3 = []               # Will contain only new entries to save space. Merge these in later.

    # Loop through articles
    for i,a in enumerate(art):
        art3.append({'paragraphs':[]}) # Grow art3
        filename = 'train_art_' + str(i).zfill(3) + '.json'
        if verbose_on: print("Article number:" + str(i).zfill(3))
        for j,p in enumerate(a['paragraphs']):
            art3[i]['paragraphs'].append([]) # Grow art3
            art3[i]['title'] = art[i]['title'] # Copy over title
            if verbose_on: print("\tParagraph number: " + str(j))

            # ID all answers
            all_questions = [qa['question'] for qa in p['qas']]
            all_answers = [a['text'] for qa in p['qas'] for a in qa['answers']]

            context = p['context']

            # Manual cleaning
            context = context.replace('  ',' ')               # Remove any double spaces!
            context = context.strip()                         # Remove excess white space at start and end

            # Split context in preparation for marking words
            context_split = allenNLP_split_words(context)

            # Create blanks
            blank_classification = [0] * len(context_split)

            # # Check to make sure reconstruction works
            # # # Commenting this out because there will almost always be a mismatch
            # # # due to using allenNLP_split_words above (causes issues with brackets, etc)
            # context_reassembled = ' '.join(join_punctuation(context_split))
            # if not context == context_reassembled:
            #     # import pdb;
            #     # pdb.set_trace()
            #     if warning_level == 1: print("Warning: Article #" + str(i) + " something's wrong - mismatch between original context and re-assembled context")
            #     if warning_level == 2:
            #         print(len(context))
            #         print(len(context_reassembled))
            #     if warning_level == 3:
            #         print(context)
            #         print(context_reassembled)

            #context_split_lower = context.lower().split()
            if verbose_on2: print('\tContext: ' + context)

            for a in all_answers:
                asplit = a.split()
                asplit = extract_no_stopwords(asplit)

                if len(asplit) <= maxWords_per_FITB and a.lower() in context.lower():
                    if verbose_on: print('\t\tAnswer <' + a + '> is present verbatim in context and has sufficiently few words')

                    # If answer a is found verbatim in the context, then switch all matching words over to blanks
                    for w in asplit:
                        for k,w2 in enumerate(context_split):
                            if w.lower() == w2.lower():
                                context_split[k] = '______'
                                blank_classification[k] = 1

                else:
                    if verbose_on: print('\t**FAIL** Answer <' + a + '> fails - either out of context or too many words')

            context_blanked = ' '.join(context_split)

            # Add entries to art2
            art2[i]['paragraphs'][j]['context_blanked'] = context_blanked
            art2[i]['paragraphs'][j]['blank_classification'] = blank_classification

            # Add entries to art3
            art3[i]['paragraphs'][j] = {'context_blanked':context_blanked,
                                        'blank_classification':blank_classification
                                       }
    if return_full:
        return art2
    else:
        return art3

def merge_arts_paragraph_fields(arts,arts2,list_of_fields,verbose_on=False):
    ''' Merges 2 lists of arts together'''
    from copy import deepcopy

    art_merged = deepcopy(arts)
    for i,a in enumerate(arts):
        if verbose_on: print("Article number:" + str(i).zfill(3))
        # Check that the titles match
        if not arts[i]['title'] == arts2[i]['title']:
            print('Warning - titles mismatch. Art1 title:' + arts[i]['title'] + ' vs ' + art2[i]['title'] + '.')
        for j,p in enumerate(a['paragraphs']):
            if verbose_on: print("\tParagraph number: " + str(j))
            for field in list_of_fields:
                art_merged[i]['paragraphs'][j][field] = arts2[i]['paragraphs'][j][field]
    return art_merged


def OR_arts_paragraph_fields(art,list_of_fields,destination_fieldname):
    ''' OR together two paragraph fields in art '''
    for i,a in enumerate(art):
        for j,p in enumerate(a['paragraphs']):
            lista=p[list_of_fields[0]]
            for k,l in enumerate(list_of_fields):
                if k>0:
                    # Skip 1st entry, sorry programming efficiency Gods
                    listb=p[list_of_fields[k]]
                    # Continually mix in new listb's until complete
                    lista = [a or b for a,b in zip(lista,listb)]
            art[i]['paragraphs'][j][destination_fieldname] = lista      # Assign ORed list

    return art
