



import numpy as np
import json
import os

# Set up path info
repodir = os.path.join(os.getenv("HOME"),'src','animated-succotash')
datadir = os.path.join(repodir,'data')
squaddir = os.path.join(datadir,'SQuAD')


filename_SQuAD_train = os.path.join(squaddir,'train-v2.0.json')
filename_SQuAD_dev = os.path.join(squaddir,'dev-v2.0.json')



###################### EXPLORE dev dataset ######################

json_data=open(filename_SQuAD_dev,'r')
data = json.load(json_data)
json_data.close()
Nemails = len(data)

# What's in data?
print(data.keys())
print(data['version'])

# All articles
art = data['data']
Narticles = len(art)
print ("Narticles = " +  str(Narticles)) 

# Single article
a1 = art[0]
a1.keys()
print(a1['title'])
par = a1['paragraphs']
Nparagraphs = len(par)
print ("Nparagraphs = " +  str(Nparagraphs)) 

# Single paragraph
p1 = par[0]
p1.keys()
p1['qas']
p1['context']

# Paragraph Text
text = p1['context']

# Paragraph questions
qas = p1['qas']
print ("Nquestions = " +  str(len(qas)))

# Question1
q1 = qas[0]
q1.keys()
print(q1['question'])
print(q1['id'])
print(q1['answers'])
print(q1['is_impossible'])


# List all article titles
art = data['data']
Narticles = len(art)
print ("Narticles = " +  str(Narticles)) 
for ind in range(Narticles):
    art_curr = art[ind]
    print(art_curr['title'])

    
###################### EXPLORE train dataset ######################

json_data=open(filename_SQuAD_train,'r')
data = json.load(json_data)
json_data.close()


# What's in data?
print(data.keys())
print(data['version'])

# All articles
art = data['data']
Narticles = len(art)
print ("Narticles = " +  str(Narticles)) 


# List all article titles
art = data['data']
Narticles = len(art)
print ("Narticles = " +  str(Narticles)) 
for ind in range(Narticles):
    art_curr = art[ind]
    print(art_curr['title'])


