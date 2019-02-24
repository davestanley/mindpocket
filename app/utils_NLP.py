


def extract_no_stopwords(tokens):
    from nltk.corpus import stopwords
    out = list()
    out = [w for w in tokens if not w in stopwords.words('english')]
    return out



def find_inds_of_NE(tags):
    # This function is out-dated now. Use instead:
    # allenNLP_classify_blanks_fromResults(results,failterm='O')
    # Get indices of all named entity tags in the tags list
    l = [i for i, t in enumerate(tags) if not t == 'O']
    return(l)

def join_punctuation(seq, characters='.,;?!'):
    # For joining lists of words together with correct
    # punctuation spacings
    characters = set(characters)
    seq = iter(seq)
    current = next(seq)

    for nxt in seq:
        if nxt in characters:
            current += nxt
        else:
            yield current
            current = nxt

    yield current


def tag_paragraph_NER(paragraph,predictor=[],verbose_mode=False):
    # Use AllenNLP to run NER on paragraph
    from allennlp.predictors import Predictor



    # Run it on my test sentence
    #predictor = Predictor.from_path("/home/davestanley/src/allennlp/ner-model-2018.12.18.tar.gz")
    import os

    # Only load predictor if not already passed and pre-loaded
    if not predictor: predictor = Predictor.from_path(os.path.join(os.getenv("HOME"),'src','allennlp','ner-model-2018.12.18.tar.gz'))
    results = predictor.predict(sentence=paragraph)
    for word, tag in zip(results["words"], results["tags"]):
        if verbose_mode:
            print(f"{word}\t{tag}")

    return results


def get_two_preceeding_sentences(sentences,ind,verbose_mode=False):
    # Returns a subset sentences:
        # sentences[ind] and sentences[ind-1] and sentences[ind-2]
        # only if they exist
    sentences_subset = []

    # Store current and previous sentences
    sentences_subset.append(sentences[ind-2] if ind >= 2 else str(''))
    sentences_subset.append(sentences[ind-1] if ind >= 1 else str(''))
    sentences_subset.append(sentences[ind])

    if verbose_mode:
        print("Current sentence number=" + str(ind))
        print("Sentence i-2 = " + sentences_subset[0])
        print("Sentence i-1 = " + sentences_subset[1])
        print("Sentence i = " + sentences_subset[2])

    return sentences_subset

def text2sentences(text):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def sent2text(sentences_subset):
    return ' '.join(join_punctuation([sent for sent in sentences_subset if sent]))  # Joins only if sentence is non-empty


def words2text(words):
    return ' '.join(join_punctuation(words))

def words2words_blanked(words,bc,blankstr='______'):
    '''Takes in words and blanks info, bc, and returns a list of blanked words'''
    return [words[i] if not t else blankstr for i,t in enumerate(bc)]

def words2words_hashblank(words,bc,hashsymbol='___'):
    '''Takes in words and blanks info, bc, and indicate blanks by hases (e.g., #Normandy#)'''
    return [words[i] if not t else hashsymbol+words[i]+hashsymbol for i,t in enumerate(bc)]

def words2answers(words,bc):
    '''Takes in words and blanks info, bc, and returns a list of answers corresponding to blanked words'''
    myanswers = []
    for i,c in enumerate(bc):
        if c == 1: myanswers.append(words[i])
    return myanswers

def allenNLP_split_words(context):
    """Uses AllenNLP to conduct word splitting, as an alternative to mystring.split()"""
    # # Split context (alternative method)
    # from allennlp.predictors import SentenceTaggerPredictor
    # predictor = SentenceTaggerPredictor([],[])
    # context_split = predictor._tokenizer.split_words(context);
    #
    from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
    ws = SpacyWordSplitter()
    context_split = ws.split_words(context);
    # Convert context_split to str's
    for i,c in enumerate(context_split):
        context_split[i] = str(context_split[i])

    return context_split


def splitsentences_allenResults(results,verbose=False):
    # Divides the results dictionary supplied by AllenNLP into a list based on
    # sentences. Output is results_list[0:Nsentences-1], each entry containing
    # fields 'words' and 'tags'
    ind_sentence_starts = [i+1 for i,w in enumerate(results['words']) if ('.' in w or '?' in w or '!' in w)]
    ind_sentence_starts.insert(0, 0)

    if len(ind_sentence_starts) < 2:
        if verbose: print("Warning no punctuation found, so cannot split sentence. Returning original as list")
        return [results]

#     # Dict, then list
#     out_words = []
#     out_tags = []
#     for i in range(len(ind_sentence_starts)-1):
#         out_words.append(results['words'][ind_sentence_starts[i]:ind_sentence_starts[i+1]])
#         out_tags.append(results['tags'][ind_sentence_starts[i]:ind_sentence_starts[i+1]])
#     return {'words': out_words, 'tags': out_tags}

    # List, then dict
    results_list = []
    for i in range(len(ind_sentence_starts)-1):
        results_list.append({
            'words': results['words'][ind_sentence_starts[i]:ind_sentence_starts[i+1]],
            'tags' : results['tags'][ind_sentence_starts[i]:ind_sentence_starts[i+1]]
        })
    return(results_list)

def merge_allenResults(results_list):
    # Merge results_list[0:Nsentences] into a single results dict
    Nsent = len(results_list)
    return({
        'words': [w for r in results_list for w in r['words']],
        'tags': [t for r in results_list for t in r['tags']]
    })

def allenNLP_classify_blanks_fromResults(results,failterm='O'):
    """Like allenNLP_classify_blanks, but works on general text, not arts"""
    # Failterm = term associated with a false classification (non-blank)

    tagsout = [0 if t == failterm else 1 for t in results['tags']]   # Convert to binary
    return tagsout

def allenNLP_classify_blanks(art,failterm='O',fieldname='blank_classified_allenNER'):
    """For all articles, converts allen NER tags into a list of 0's or 1's and stores these in field: blank_classified_allenNER"""
    # Failterm = term associated with a false classification (non-blank)
    for i,a in enumerate(art):
        for j,p in enumerate(a['paragraphs']):
            tags = p['allenNER']['tags'].split()
            art[i]['paragraphs'][j][fieldname] = [0 if t == failterm else 1 for t in tags]   # Convert to binary
    return(art)

def extract_blanked_out_sentences(results,failterm='O',easiness=0,verbose_mode = False):
    # From results word/token list, blank out a random word, and then return
    # the sentences containing that blanked out word, plus a few preceding
    # centences for context.
    from nltk.tokenize import sent_tokenize
    import numpy as np
    import random
    import spacy # For non-random word selection only

    words = results['words']
    tags = results['tags']


    tags = allenNLP_classify_blanks_fromResults(results,failterm)
    l_NE = [i for i, t in enumerate(tags) if t == 1]

    #l_NE = find_inds_of_NE(tags)
    if verbose_mode:
        print("Indices of named entities:" + str(l_NE))

    # Catch if nothing found, so program doesn't crash
    if not l_NE:
        print('Warning - no named entities found. Returning empty')
        text_blanks = dict()
        text_blanks['text'] = 'No suitable blanks found'
        text_blanks['removed_word'] = 'error'
        text_blanks['removed_word_tag'] = 'error'

        return text_blanks

    choose_word_randomly = False # if false, tries to choose
                        # The word that is least similar to the document
                        # overall. The goal here is to ensure that you don't
                        # for example, blank out "Rome" in an article about
                        # Rome (e.g., too obvious)
    if choose_word_randomly:
        # Choose one at random
        ind = random.choice(l_NE)
    else:
        try:
            # Choose the word based on how similar it is to other words in the sentence
            # Similar words are easy, dissimilar words are hard
            nlp = spacy.load('en_core_web_sm')
            candidate_blanks = [words[i] for i in l_NE]
            # candidate_blanks = []
            # for i in l_NE: candidate_blanks.append(words[i])
            doc = nlp(words2text(results['words']))
            doc2 = nlp(candidate_blanks[0])
            doc.similarity(doc2)

            # Calculate similarity to sentence for each candidate blank
            sims = []
            for i,cb in enumerate(candidate_blanks): sims.append(doc.similarity(nlp(cb)))
            ind_min = sims.index(min(sims))
            ind_max = sims.index(max(sims))
            candidate_blanks[ind_min]
            candidate_blanks[ind_max]


            # Choose entry in candidate_blanks depending on difficulty
            # This works by finding the blank corresponding to the "easiness" percentile
            # difficulty = 100
            # easiness = 100 - difficulty
            # Find value of the easiness percentile
            print(easiness)
            arr = np.array(sims)
            per = np.percentile(arr, easiness) # easiest blanks have the closest similarity to the document overall

            # Match to the closest one in the actual list
            arr.sort()
            per_orig = arr[np.where(arr>=per)[0][0]]

            # Now, return index of original one
            arr_orig = np.array(sims)
            ind_cb = np.where(arr_orig==per_orig)[0].tolist()[0] # Take first index matching per]

            # Find where it is in our original list of blanks
            ind = l_NE[ind_cb]

        except:
            print("###Warning - could not intelligently choose random blank. Reverting to random selection###")
            ind = random.choice(l_NE)


    # Back up blanked out word and word type
    removed_word = words[ind]
    removed_word_tag = tags[ind]

    # Replace chosen word with blank
    words_new = words.copy()
    blank_token = '____'
    words_new[ind] = blank_token

    # Rebuild the sentence with appropriate punctuation
    #paragraph_new = ' '.join(words_new).replace(" ,", ",").replace(" .", ".")
    paragraph_new = words2text(words_new)

    # Print this sentence along with the previous sentence together
    if verbose_mode: print(paragraph_new)
    if verbose_mode: print("Answer: " + removed_word)



    # Finally, figure out which sentence contains the blank and only present it and the previous two
    # ===

    # First, figure out the index of the sentence containing the blank
    sentences_new = text2sentences(paragraph_new)


    curr_word = 0
    i=0
    for sent in sentences_new:
        i=i+1
        curr_word = curr_word + len(sent.split())
        if ind < curr_word:
            break

    ind_sentence_containing_blank = i-1
    if verbose_mode: print(ind_sentence_containing_blank)


    # if verbose_mode:
    #     # This method searches for the token directly - it is slower, but guaranteed to work)
    #     # Never mind, punctuation messes this method up. No time to fix
    #     sentences_new = sent_tokenize(paragraph_new)


    #     i=0
    #     for sent in sentences_new:
    #         if blank_token in sent.split():
    #             break
    #         i=i+1
    #     ind_sentence_containing_blank2 = i
    #     if verbose_mode: print(ind_sentence_containing_blank2)
    #     if not ind_sentence_containing_blank2 == ind_sentence_containing_blank: print('Error occurred')




    sentences_subset = get_two_preceeding_sentences(sentences_new,ind_sentence_containing_blank)
    paragraph_subset = sent2text(sentences_subset)

    # Fix any spacing issues
    paragraph_subset2 = paragraph_subset
    paragraph_subset2 = paragraph_subset2.replace(' )',')')
    paragraph_subset2 = paragraph_subset2.replace('( ','(')
    paragraph_subset2 = paragraph_subset2.replace('[ ','[')
    paragraph_subset2 = paragraph_subset2.replace(' ]',']')
    paragraph_subset2 = paragraph_subset2.replace(' - ','-')

    text_blanks = dict()
    text_blanks['text'] = paragraph_subset
    text_blanks['removed_word'] = removed_word
    text_blanks['removed_word_tag'] = removed_word_tag

    return text_blanks
