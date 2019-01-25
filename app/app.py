
def find_inds_of_NE(tags):
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


def tag_paragraph_NER(paragraph,verbose_mode=False):
    # Use AllenNLP to run NER on paragraph
    from allennlp.predictors import Predictor


    # Set up other global variables
    verbose_mode = False


    # Run it on my test sentence
    predictor = Predictor.from_path("/home/davestanley/src/allennlp/ner-model-2018.12.18.tar.gz")
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


def extract_blanked_out_sentences(results,verbose_mode = False):
    # From results word/token list, blank out a random word, and then return
    # the sentences containing that blanked out word, plus a few preceding
    # centences for context.
    from nltk.tokenize import sent_tokenize

    words = results['words']
    tags = results['tags']

    l_NE = find_inds_of_NE(tags)
    if verbose_mode:
        print("Indices of named entities:" + str(l_NE))


    # Choose one at random
    import random
    ind = random.choice(l_NE)


    # Back up blanked out word and word type
    removed_word = words[ind]
    removed_word_tag = words[ind]

    # Replace chosen word with blank
    words_new = words.copy()
    blank_token = '____'
    words_new[ind] = blank_token

    # Rebuild the sentence with appropriate punctuation
    #paragraph_new = ' '.join(words_new).replace(" ,", ",").replace(" .", ".")
    paragraph_new = ' '.join(join_punctuation(words_new))

    # Print this sentence along with the previous sentence together
    if verbose_mode: print(paragraph_new)
    if verbose_mode: print("Answer: " + removed_word)



    # Finally, figure out which sentence contains the blank and only present it and the previous two
    # ===

    # First, figure out the index of the sentence containing the blank
    sentences_new = sent_tokenize(paragraph_new)


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
    paragraph_subset = ' '.join(join_punctuation([sent for sent in sentences_subset if sent]))  # Joins only if sentence is non-empty

    text_blanks = dict()
    text_blanks['text'] = paragraph_subset
    text_blanks['removed_word'] = removed_word
    text_blanks['removed_word_tag'] = removed_word_tag

    return text_blanks



# Set up and load data
# Includes
import sys
import os


# Setup paths containing utility
curr_folder = os.getcwd()
sys.path.insert(0, os.path.join(curr_folder,'../app'))

# Load the data
from utils import load_SQuAD_train
arts = load_SQuAD_train()


# Choose a paragraph
paragraph = arts[15]['paragraphs'][1]['context']
print(paragraph)

# Tag the paragraph
results = tag_paragraph_NER(paragraph)


# Print the sentences
text_blanks = extract_blanked_out_sentences(results)

blanked_sentence = text_blanks['text']
removed_word = text_blanks['removed_word']
removed_word_tag = text_blanks['removed_word_tag']

print(blanked_sentence)
print('Answer: ' + removed_word)
