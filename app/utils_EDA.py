# Exploration of data sets

def p_list_qas(paragraphs):
    # List all questions and answers present in input paragraphs
    for p in paragraphs:
        print('Context: ' + p['context'])
        for j,qa in enumerate(p['qas']):
            print('Question ' + str(j) + ': ' + qa['question'])
            for k,a in enumerate(qa['answers']):
                print('\tAnswer ' + str(k) + ': ' + a['text'])

def a_sentences_per_article(art):
    # Number of sentences per article
    from nltk.tokenize import sent_tokenize

    asentences = []
    for a in art:
        psentences = [len(sent_tokenize(p['context'])) for p in a['paragraphs']]
        asentences.append(sum(psentences))

    return asentences
