# Exploration of data sets

def p_list_qas(paragraphs):
    # List all questions and answers present in input paragraphs
    for p in paragraphs:
        print('Context: ' + p['context'])
        for j,qa in enumerate(p['qas']):
            print('Question ' + str(j) + ': ' + qa['question'])
            for k,a in enumerate(qa['answers']):
                print('\tAnswer ' + str(k) + ': ' + a['text'])
