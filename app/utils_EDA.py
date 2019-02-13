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

def calcstats_train_dev(myvar,filename=None,Ntrain=0,Ndev=0,Ntest=0,mytitle=None):
    import statistics
    import json

    st = {}
    if Ntrain and Ntrain > 0:
        st['train'] = {
                    'median': statistics.median(myvar[0:Ntrain-1]),
                    'mean': statistics.mean(myvar[0:Ntrain-1]),
                    #'sum': sum(myvar[0:Ntrain-1])
                    }

    if Ndev and Ndev > 0:
        st['dev'] = {
                    'median': statistics.median(myvar[Ntrain:Ntrain+Ndev-1]),
                    'mean': statistics.mean(myvar[Ntrain:Ntrain+Ndev-1]),
                    #'sum': sum(myvar[Ntrain:Ntrain+Ndev-1])
                    }
    if Ntest and Ntest > 0:
        st['test'] = {
                    'median': statistics.median(myvar[Ntest:]),
                    'mean': statistics.mean(myvar[Ntest:]),
                    #'sum': sum(myvar[Ntest:])
                    }
    #
    # if filename:
    #     with open(filename, 'w') as f:
    #         json.dump(st, f)

    if filename:
        with open(filename, 'a') as f:
            if mytitle:
                f.write('\"==' + mytitle + '==\"\n')
            for k in st.keys():
                json.dump([k, st[k]], f)
                f.write('\n')

    return st
