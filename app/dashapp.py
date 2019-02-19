

######################### App GUI Functions #########################



# General imports
import sys
import os

# Setup app path
curr_folder = os.getcwd()
sys.path.insert(0, os.path.join(curr_folder,'app'))
# sys.path.insert(0, os.path.join(os.getenv("HOME"),'src','mindpocket','app'))

# Set up genanki path
sys.path.insert(0, os.path.join(curr_folder,'submodules','genanki'))

# Imports for dash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app

# Import flask
import flask

# Imports for anki
import random
import genanki

# File management
import shutil
import glob

# Import AllenNLP
from allennlp.predictors import Predictor

# Include custom
import myallennlp
from myallennlp import *
from myallennlp.models.simple_tagger2 import SimpleTagger2
from myallennlp.dataset_readers import sequence_tagging2
from myallennlp.data.tokenizers.word_splitter import SpacyWordSplitter

# Set up temporary folder for saving contents of this session
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
ankiout_path = os.path.join('downloads',timestr)
if not os.path.exists(ankiout_path):
    os.makedirs(ankiout_path)


# Delete all download that have expired (right now, default expiration time is
# 1 hour. Comment out other lines to change)
# timestr_tokeep = time.strftime("%Y%m%d-%H%M")           # 1 minute
timestr_tokeep = time.strftime("%Y%m%d-%H")           # 1 hour
# timestr_tokeep = time.strftime("%Y%m%d")              # 1 day

# Delete files that are past the expiration
allfiles = glob.glob(os.path.join('downloads','*'))
allfiles_tokeep = glob.glob(os.path.join('downloads',timestr_tokeep + '*'))
for f in allfiles:
    if f not in allfiles_tokeep:
        print('Deleting expired downloads folder ' + f)
        shutil.rmtree(f)


# Test
@app.route('/index')
def sayHi():
    return "Hi from my Flask App!"

# Flask coding for downloading files
@app.route('/downloads/<path>/<filename>')
def return_downloads(path = None, filename = None):
    #path = '/output.apkg'
    #path = '/home/davestanley/Dropbox/git/mindpocket/output.apkg'
    # print('original path')
    # print(path)
    path = os.path.join(os.getenv("HOME"),'src','mindpocket','downloads',path,filename)
    #path = '/home/davestanley/Dropbox/git/mindpocket/downloads/' + path + '/' + filename
    # print('new path')
    # print(path)
    return flask.send_file(path, as_attachment=True)

# Define for IIS module registration.
wsgi_app = app.wsgi_app

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



######################### NLP Functions #########################
# Import local functions
from app.utils_NLP import tag_paragraph_NER, splitsentences_allenResults, merge_allenResults, extract_blanked_out_sentences

# Connect dash to flask
dashapp = dash.Dash(__name__, server=app, url_base_pathname='/',external_stylesheets=external_stylesheets)
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#DDEEFF',
    # 'background': '#FFFFFF',
    'text': '#000000',
    'align': 'center'
}

dashapp.title='MindPocket: Optimizing Learning'

dashapp.layout = html.Div(
    [
        html.H1(children='MindPocket', style={'textAlign': colors['align'],'color': colors['text']}),
        html.Div(style={'textAlign': colors['align'],'color': colors['text']},children='''
            Enter text to generate questions
        '''),
        dcc.Textarea(
            id='input-box',
            placeholder='Enter a value...',
            value='',
            style={'textAlign': 'left','color': '#000000','width': '95%'},
            rows=20
        ),
        # html.Div(style={'textAlign': colors['align'],'color': colors['text']},children='''
        #     Select difficulty:
        # '''),
        html.Div(
            [
                html.P('Select difficulty:'),
                dcc.Slider(
                    id='difficulty-slider',
                    min=1,
                    max=10,
                    marks={i: 'Level {}'.format(i) if i == 1 or i == 10 else '{}'.format(i) for i in range(1,11)},
                    value=10
                )
            ],
            style={'margin-bottom': '40'}
        ),
        html.Button('Generate!', id='button'),
        # html.Div(style={'textAlign': colors['align'],'color': colors['text']},children='''
        #     blah
        # '''),
        html.Div(
            [
                html.A(
                    children='Download Anki file',
                    id='download-link',
                    download='file.apkg',
                    href='/downloads/' + timestr + '/' + 'mindpocket_deck.apkg'
                )
            ],
            style={'margin-bottom': '10','margin-top': '10'}
        ),
        html.Div(id='output-container-button',
                 children='Enter a value and press submit',
                 style={'textAlign': 'left','color': colors['text']})
     ],
     style={'width': '80%',
           'margin-left': 'auto',
           'margin-right' : 'auto',
           'line-height' : '30px',
           'backgroundColor': colors['background'],
           'padding': '40px'
           }
)


# Only open if predictor doesn't exist
# import inspect
# predictor = []
# print(inspect.stack()[1].function)
# print('Starting up')
testing_mode = False
use_allenNLP_NER_as_model = False     # if true, uses AllenNLP pre-trained model; if false, uses my model trained on SQUAD data

if not testing_mode:
    try:
        predictor;
    except:
        if use_allenNLP_NER_as_model:
            # AllenNER model
            predictor = Predictor.from_path(os.path.join(os.getenv("HOME"),'src','allennlp','ner-model-2018.12.18.tar.gz'))
            failterm = 'O'    # Model output associated with a "false" classification (e.g., do not blank)
        else:
            # My model
            mymodel = os.path.join(curr_folder,'app','model.tar.gz')
            predictor = Predictor.from_path(mymodel,predictor_name='sentence-tagger')
            failterm = '0'    # Model output associated with a "false" classification (e.g., do not blank)

@dashapp.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value'),
    dash.dependencies.State('difficulty-slider', 'value')])
def update_output(n_clicks, value, difficulty):
    verbose_mode = False

    paragraph = value

    if paragraph.strip() == '':
        print('Empty run.')
        out = []
        return out

    # Testing: see if we can identify paragraph breaks
    if testing_mode:
        print(paragraph)
        print("Paragraph type: " + str(type(paragraph)))
        if '\n' in paragraph:
            print('pass')


    if not testing_mode:
        # Tag the paragraph
        results = tag_paragraph_NER(paragraph,predictor)

        if verbose_mode:
            for word, tag in zip(results["words"], results["tags"]):
                print(f"{word}\t{tag}")

        # Subdivide results into separate sentences, based on punctuation
        results_list = splitsentences_allenResults(results)

        # Loop through sentences in steps of 2 sentences at a time
        Nsent = len(results_list)
        step_size=2
        out = []
        count=0
        question_list = []
        answers_list = []
        answers_tag_list = []
        answers_text = ''
        for i in range(0,Nsent,step_size):
            count=count+1
            results_curr = merge_allenResults(results_list[i:min(i+step_size,Nsent)])
            print (results_curr)

            # Generate the sentences
            easiness = 100-10*difficulty
            text_blanks = extract_blanked_out_sentences(results_curr,failterm,easiness)
            blanked_sentence = text_blanks['text']
            removed_word = text_blanks['removed_word']
            removed_word_tag = text_blanks['removed_word_tag']

            # Append to running tally of all Q & A's
            question_list.append(blanked_sentence)
            answers_list.append(removed_word)
            answers_tag_list.append(removed_word_tag)

            # Append to text output for questions
            out.append(html.P('Question {}: {}'.format(str(count),blanked_sentence)))
            if count == 1: answers_text = answers_text + 'Answers: {}. {}'.format(str(count),removed_word)
            else: answers_text = answers_text + ', {}. {}'.format(str(count),removed_word)

        # Append to text output for answers
        answers_text = answers_text + '.'
        out.append(html.P(answers_text))

        #print(blanked_sentence)
        #print('Answer: ' + removed_word)
    else:
        # Initialize lists
        question_list = []
        answers_list = []
        answers_tag_list = []

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

        # Append to running tally of all Q & A's
        question_list.append(blanked_sentence)
        answers_list.append(removed_word)
        answers_tag_list.append(removed_word_tag)

    # # Save to Anki # #
    # Build model
    myrand = random.randrange(1 << 30, 1 << 31)
    my_model = genanki.Model(
        myrand,
        'Simple Model',
        fields=[
            {'name': 'Question'},
            {'name': 'Answer'}
        ],
        templates=[
            {
              'name': 'Card 1',
              'qfmt': '{{Question}}',
              'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
            }
        ]
    )

    # Build deck
    deck_name = 'Mindpocket Deck'
    my_deck = genanki.Deck(
        myrand+1,
        deck_name)

    # Add notes to deck
    for i in range(len(question_list)):
        my_note = genanki.Note(
            model=my_model,
            fields=[question_list[i], answers_list[i]])
        my_deck.add_note(my_note)

    # # Save deck to file # #
    # First, get string containing current time
    deckname = "mindpocket_deck"
    ankiout_file = deckname + '.apkg'
    ankiout_full = os.path.join(ankiout_path,ankiout_file)
    genanki.Package(my_deck).write_to_file(ankiout_full)


    # import pdb
    # pdb.set_trace()

    return out
    # return 'Fill in the blank:\n"{}"\n Answer:{} \n'.format(
    #     blanked_sentence,
    #     removed_word
    # )




# if __name__ == '__main__':
#     dashapp.run_server(debug=True)
