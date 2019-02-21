# MindPocket

Mindpocket generates fill-in-the-blank questions from any body of text. These questions can then be imported into Anki (https://apps.ankiweb.net/), a popular open-source spaced repetition flashcard program. Try it out here: [mindpocket.co](mindpocket.co)

**Google slides presentation**: https://docs.google.com/presentation/d/1xPXH58mQQloVF6xvLjpK8aK8PnqofJTZtT6BnSYMKjY

**Insight project page**: https://platform.insightdata.com/projects/mindpocket

# To run

Run: `python run.py`
Or: `gunicorn app:app -D`

## Requirements
- AllenNLP (http://allennlp.org)
- Dash (conda install dash)
- spaCy (https://spacy.io/usage/)
- genanki (https://github.com/kerrickstaley/genanki)

