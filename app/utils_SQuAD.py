
# Utilities specifically for working with the SQuAD dataset

def classify_blanks_from_answers(art,maxWords_per_FITB=2,verbose_on=False):
    """Creates keys context_blanked and blank_classification under paragraph, based on keywords present in answers"""
    # Imports
    from utils_NLP import extract_no_stopwords
    from copy import deepcopy

    # Local settings
    verbose_on2 = False # Set to true to display full context for each paragraph

    # Copy so don't overwrite original
    art2 = deepcopy(art)
    #art2 = art.copy()

    # Loop through articles
    for i,a in enumerate(art):
        filename = 'train_art_' + str(i).zfill(3) + '.json'
        if verbose_on: print("Article number:" + str(i).zfill(3))
        for j,p in enumerate(a['paragraphs']):
            if verbose_on: print("\tParagraph number: " + str(j))

            # ID all answers
            all_questions = [qa['question'] for qa in p['qas']]
            all_answers = [a['text'] for qa in p['qas'] for a in qa['answers']]

            context = p['context']
            context = context.replace('  ',' ')
            context_split = context.split()
            blank_classification = [False] * len(context_split)

            # Check to make sure reconstruction works
            context_reassembled = ' '.join(context_split)
            if not context == context_reassembled:
                print("Warning: Article #" + str(i) + " somethings wrong - mismatch between original context and re-assembled context")
#                 print(context)
#                 print(context_reassembled)
#                 print(len(context))
#                 print(len(context_reassembled))

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
                                blank_classification[k] = True

                else:
                    if verbose_on: print('\t**FAIL** Answer <' + a + '> fails - either out of context or too many words')

            context_blanked = ' '.join(context_split)
            art2[i]['paragraphs'][j]['context_blanked'] = context_blanked
            art2[i]['paragraphs'][j]['blank_classification'] = blank_classification

    return art2
