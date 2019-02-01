

######################### App GUI Functions #########################



# General imports
import sys
import os

# Imports for dash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app

# Test
@app.route('/index')
def sayHi():
    return "Hi from my Flask App!"

# Define for IIS module registration.
wsgi_app = app.wsgi_app

# Setup local paths
# curr_folder = os.getcwd()
# print(curr_folder)
# sys.path.insert(0, os.path.join(curr_folder,'../app'))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



######################### NLP Functions #########################
# Import local functions
from app.utils_NLP import tag_paragraph_NER, split_allenResults, merge_allenResults, extract_blanked_out_sentences

# Connect dash to flask
dashapp = dash.Dash(__name__, server=app, url_base_pathname='/',external_stylesheets=external_stylesheets)
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#DDEEFF',
    'text': '#000000',
    'align': 'center'
}

dashapp.layout = html.Div(style={'backgroundColor': colors['background']},children=[
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


@dashapp.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])
def update_output(n_clicks, value):
    testing_mode = False
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



    # import pdb
    # pdb.set_trace()

    return out
    # return 'Fill in the blank:\n"{}"\n Answer:{} \n'.format(
    #     blanked_sentence,
    #     removed_word
    # )




# if __name__ == '__main__':
#     dashapp.run_server(debug=True)
