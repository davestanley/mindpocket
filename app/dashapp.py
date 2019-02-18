

######################### App GUI Functions #########################



# General imports
import sys
import os

# Setup app path
curr_folder = os.getcwd()
sys.path.insert(0, os.path.join(curr_folder,'app'))
# sys.path.insert(0, os.path.join(os.getenv("HOME"),'src','mindpocket','app'))


# Imports for dash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app

# Import AllenNLP
from allennlp.predictors import Predictor

# Include custom
import myallennlp
from myallennlp import *
from myallennlp.models.simple_tagger2 import SimpleTagger2
from myallennlp.dataset_readers import sequence_tagging2
from myallennlp.data.tokenizers.word_splitter import SpacyWordSplitter


# Test
@app.route('/index')
def sayHi():
    return "Hi from my Flask App!"

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
    style={'width': '80%',
          'margin-left': 'auto',
          'margin-right' : 'auto',
          'line-height' : '30px',
          'backgroundColor': colors['background'],
          'padding': '20px'
          },
    children=[
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
    html.Div(style={'textAlign': colors['align'],'color': colors['text']},children='''
        Select difficulty:
    '''),
    dcc.Slider(
        min=1,
        max=10,
        marks={i: 'Level {}'.format(i) if i == 1 or i == 10 else '{}'.format(i) for i in range(1,11)},
        value=10
    ),
    html.Div(style={'textAlign': colors['align'],'color': colors['text']},children='''

    '''),
    html.Button('Generate!', id='button'),
    html.Div(id='output-container-button',
             children='Enter a value and press submit',
             style={'textAlign': 'left','color': colors['text']})
])


# Only open if predictor doesn't exist
# import inspect
# predictor = []
# print(inspect.stack()[1].function)
# print('Starting up')
testing_mode = True
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
    [dash.dependencies.State('input-box', 'value')])
def update_output(n_clicks, value):
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
            easiness = 0
            text_blanks = extract_blanked_out_sentences(results_curr,failterm,easiness)
            blanked_sentence = text_blanks['text']
            removed_word = text_blanks['removed_word']
            removed_word_tag = text_blanks['removed_word_tag']

            out.append(html.P('Question {}: {}'.format(str(count),blanked_sentence)))
            answers_list.append(removed_word)
            answers_tag_list.append(removed_word_tag)
            if count == 1: answers_text = answers_text + 'Answers: {}. {}'.format(str(count),removed_word)
            else: answers_text = answers_text + ', {}. {}'.format(str(count),removed_word)
        answers_text = answers_text + '.'
        out.append(html.P(answers_text))

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



    # import pdb
    # pdb.set_trace()

    return out
    # return 'Fill in the blank:\n"{}"\n Answer:{} \n'.format(
    #     blanked_sentence,
    #     removed_word
    # )




# if __name__ == '__main__':
#     dashapp.run_server(debug=True)
