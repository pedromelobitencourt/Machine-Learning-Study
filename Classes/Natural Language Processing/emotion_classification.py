import pandas as pd 
import string
import spacy
import random
import seaborn as sns
import numpy as np
from spacy.lang.pt.stop_words import STOP_WORDS

def preprocessing(text):
    text = text.lower()
    document = nlp(text)

    list = []

    for token in document:
        list.append(token.lemma_)

    list = [word for word in list if word not in STOP_WORDS and word not in punctuations]
    list = ' '.join([str(element) for element in list if not element.isdigit()])

    return list

## Loading database

database = pd.read_csv('../../Databases/training_base.txt', encoding='utf-8')
print(database.shape)

punctuations = string.punctuation

nlp = spacy.load('pt_core_news_sm')

test = preprocessing('Estou aprendendo processamento de linguagem natural, curso em Belo Horizonte')
print(test)


## Pre-processing
database['texto'] = database['texto'].apply(preprocessing)
final_database = []

for text, emotion in zip(database['texto'], database['emocao']):
    if emotion == 'alegria':
        dic = ({'ALEGRIA': True, 'MEDO': False})
    elif emotion == 'medo':
        dic = ({'ALEGRIA': False, 'MEDO': True})

    final_database.append([text, dic.copy()])

print(len(final_database), final_database[0])


## Creating the classifier
model = spacy.blank('pt')
category = model.create_pipe('textcat')
category.add_label('ALEGRIA')
category.add_label('MEDO')

model.add_pipe(category)
history = []

model.begin_training()

for episode in range(1000):
    random.shuffle(final_database)

    losses = {}

    for batch in spacy.util.minibatch(final_database, 30): # train 30 by 30
        texts = [model(text) for text, entities in batch]
        annotations = [{'cats': entities} for text, entities in batch]
        model.update(texts, annotations, losses=losses)

    if episode % 100 == 0:
        print(losses) # the error
        history.append(losses)