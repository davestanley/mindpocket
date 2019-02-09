
import allennlp
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
spw = SpacyWordSplitter(pos_tags = True, ner = True)


sentence = 'I am going ot the store in France. George Washington is awesome.'
sentences = ['I am going to the store in France.', 'George Washington is awesome', 'I like ice cream.','The vikings are awesome people from Normandy']

# Single split words
out = spw.split_words(sentence)
for o in out:
    print(o.pos_)

for o in out:
    print(o.ent_type_)


# Batch split words
sent = spw.batch_split_words(sentences)


for out in sent:
    for o in out:
        print(o.pos_)

for out in sent:
    for o in out:
        print(o.ent_type_)
