# Exploration of data sets


# # # # Printing stuff
def p_list_qas(paragraphs):
    # List all questions and answers present in input paragraphs
    for p in paragraphs:
        print('Context: ' + p['context'])
        for j,qa in enumerate(p['qas']):
            print('Question ' + str(j) + ': ' + qa['question'])
            for k,a in enumerate(qa['answers']):
                print('\tAnswer ' + str(k) + ': ' + a['text'])

# # # # Returning stuff
def a_sentences_per_article(art):
    # Number of sentences per article
    from nltk.tokenize import sent_tokenize

    asentences = []
    for a in art:
        psentences = [len(sent_tokenize(p['context'])) for p in a['paragraphs']]
        asentences.append(sum(psentences))

    return asentences

def p_return_words_blanked(words,is_blanked,blanked_str='______'):
    words_blanked = [words[i] if not t else blanked_str for i,t in enumerate(tags)]
    return words_blanked

def p_return_words_upper(words,is_blanked):
    words_upper = [words[i] if not t else words[i].upper() for i,t in enumerate(tags)]
    return words_upper

def p_return_blanks(words,is_blanked):
    words_upper = [words[i] for i,t in enumerate(tags) if t]
    return words_upper
