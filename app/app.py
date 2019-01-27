

######################### NLP Functions #########################

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

def text2sentences(text):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def sent2text(sentences_subset):
    return ' '.join(join_punctuation([sent for sent in sentences_subset if sent]))  # Joins only if sentence is non-empty


def words2text(words):
    return ' '.join(join_punctuation(words))

def split_allenResults(results):
    # Divides the results dictionary supplied by AllenNLP into a list based on
    # sentences
    ind_sentence_starts = [i+1 for i,w in enumerate(results['words']) if (w=='.' or w=='?' or w=='!')]
    ind_sentence_starts.insert(0, 0)

    if len(ind_sentence_starts) < 2:
        print("Warning no punctuation found, so cannot split sentence. Returning original")
        return results

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
    # Merge list of resutls into a single results dict
    Nsent = len(results_list)
    return({
        'words': [w for r in results_list for w in r['words']],
        'tags': [t for r in results_list for t in r['tags']]
    })

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

    # Catch if nothing found, so program doesn't crash
    if not l_NE:
        print('Warning - no named entities found. Returning empty')
        text_blanks = dict()
        text_blanks['text'] = 'No suitable blanks found'
        text_blanks['removed_word'] = 'error'
        text_blanks['removed_word_tag'] = 'error'

        return text_blanks



    # Choose one at random
    import random
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

    text_blanks = dict()
    text_blanks['text'] = paragraph_subset
    text_blanks['removed_word'] = removed_word
    text_blanks['removed_word_tag'] = removed_word_tag

    return text_blanks

######################### App GUI Functions #########################



# General imports
import sys
import os

# Imports for dash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Setup local paths
curr_folder = os.getcwd()
sys.path.insert(0, os.path.join(curr_folder,'../app'))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#DDEEFF',
    'text': '#000000',
    'align': 'center'
}

app.layout = html.Div(style={'backgroundColor': colors['background']},children=[
    html.H1(children='MindPocket', style={'textAlign': colors['align'],'color': colors['text']}),
    html.Div(style={'textAlign': colors['align'],'color': colors['text']},children='''
        Enter text to generate questions
    '''),
    dcc.Textarea(
        id='input-box',
        placeholder='Enter a value...',
        value='',
        style={'textAlign': 'left','color': '#000000','width': '100%'},
        rows=20
    ),
    html.Button('Generate!', id='button'),
    html.Div(id='output-container-button',
             children='Enter a value and press submit',
             style={'textAlign': 'left','color': colors['text']})
])


@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])
def update_output(n_clicks, value):
    testing_mode = True
    verbose_mode = True

    paragraph = value

    # Testing: see if we can identify paragraph breaks
    if testing_mode:
        print(paragraph)
        print("Paragraph type: " + str(type(paragraph)))
        if '\n' in paragraph:
            print('pass')


    if not testing_mode:
        # Tag the paragraph
        results = tag_paragraph_NER(paragraph)

        if verbose_mode:
            for word, tag in zip(results["words"], results["tags"]):
                print(f"{word}\t{tag}")

        # Subdivide results into separate sentences, based on punctuation
        results_list = split_allenResults(results)

        # Loop through sentences in steps of 4 sentences at a time
        Nsent = len(results_list)
        step_size=2
        out = []
        count=0
        for i in range(0,Nsent,step_size):
            count=count+1
            results_curr = merge_allenResults(results_list[i:min(i+step_size,Nsent)])
            print (results_curr)

            # Generate the sentences
            text_blanks = extract_blanked_out_sentences(results_curr)
            blanked_sentence = text_blanks['text']
            removed_word = text_blanks['removed_word']
            removed_word_tag = text_blanks['removed_word_tag']

            out.append(html.P('Question {}: Fill in the blank(s)'.format(str(count))))
            out.append(html.P(blanked_sentence))
            out.append(html.P('Answer: {}; (Question type: {})'.format(removed_word,removed_word_tag)))

        #print(blanked_sentence)
        #print('Answer: ' + removed_word)
    else:
        # Fill in some default values
        blanked_sentence = 'as created by Jordan ___, a software engi'
        removed_word = 'Walke'
        removed_word_tag = 'B-ORG'
        out = []
        count=0
        out.append(html.P('Question {}: Fill in the blank(s)'.format(str(count))))
        out.append(html.P(blanked_sentence))
        out.append(html.P('Answer: {}; (Question type: {})'.format(removed_word,removed_word_tag)))
        count=1
        out.append(html.P('Question {}: Fill in the blank(s)'.format(str(count))))
        out.append(html.P(blanked_sentence))
        out.append(html.P('Answer: {}; (Question type: {})'.format(removed_word,removed_word_tag)))





    return out
    # return 'Fill in the blank:\n"{}"\n Answer:{} \n'.format(
    #     blanked_sentence,
    #     removed_word
    # )




if __name__ == '__main__':
    app.run_server(debug=True)
