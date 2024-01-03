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

    token_list = []

    for token in document:
        token_list.append(token.lemma_)

    token_list = [word for word in token_list if word not in STOP_WORDS and word not in punctuations]
    token_list = ' '.join([str(element) for element in token_list if not element.isdigit()])

    return token_list

# Loading database
database = pd.read_csv('../../Databases/training_base.txt', encoding='utf-8')
print(database.shape)

punctuations = string.punctuation

nlp = spacy.load('pt_core_news_sm')

test = preprocessing('Estou aprendendo processamento de linguagem natural, curso em Belo Horizonte')
print(test)

# Pre-processing
database['texto'] = database['texto'].apply(preprocessing)
final_database = []

for text, emotion in zip(database['texto'], database['emocao']):
    if emotion == 'alegria':
        dic = {'ALEGRIA': True, 'MEDO': False}
    elif emotion == 'medo':
        dic = {'ALEGRIA': False, 'MEDO': True}

    final_database.append((text, dic.copy()))

print(len(final_database), final_database[0])

# Creating the classifier
model = spacy.blank('pt')

categories = model.create_pipe('textcat')
categories.add_label('ALEGRIA')
categories.add_label('MEDO')

model.add_pipe(categories)

history = []

model.begin_training()

for episode in range(1000):
    random.shuffle(final_database)
    losses = {}

    for batch in spacy.util.minibatch(final_database, 30):  # train 30 by 30
        texts = [model(text) for text, entities in batch]
        annotations = [{'cats': entities} for text, entities in batch]
        model.update(texts, annotations, losses=losses)

    if episode % 100 == 0:
        print(losses)
        history.append(losses)

loss_history = []

for i in history:
    loss_history.append(i.get('textcat'))

loss_history = np.array(loss_history)

import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.title('Error progression')
plt.xlabel('Episode')
plt.ylabel('Error') # view the graph to know how many episodes to run

model.to_disk('model') # saving the model


## Testing with a phrase
loaded_model = spacy.load('model') # if you saved previously

positive_text = "i love your eye's color"
positive_text = preprocessing(positive_text)

forecast = loaded_model(positive_text)
print(forecast.cats) # the probability of each category

negative_text = "i'm afraid of him"
negative_text = preprocessing(negative_text)

forecast = loaded_model(negative_text)
print(forecast.cats)


## Model Assessing on Training Database
forecasts = []

for text in database['texto']:
    forecast = loaded_model(text)
    forecasts.append(forecast.cats)

final_forecast = []

for forecast in forecasts:
    if forecast['ALEGRIA'] > forecast['MEDO']:
        final_forecast.append('alegria')
    else:
        final_forecast.append('medo')
    
final_forecast = np.array(final_forecast)

real_answers = database['emocao'].values

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(real_answers, final_forecast) # 100%: overfitting or...

cm = confusion_matrix(real_answers, final_forecast)


## Model Assessing on Test Database

test_database = pd.read_csv('../../Databases/test_base.txt', encoding='utf-8')
test_database['texto'] = database['texto'].apply(preprocessing)

forecasts = []

for text in test_database['texto']:
    forecast = loaded_model(text)
    forecasts.append(forecast.cats)

final_forecast = []

for forecast in forecasts:
    if forecast['ALEGRIA'] > forecast['MEDO']:
        final_forecast.append('alegria')
    else:
        final_forecast.append('medo')
    
final_forecast = np.array(final_forecast)

real_answers = test_database['emocao'].values

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(real_answers, final_forecast) # 100%: overfitting or...

cm = confusion_matrix(real_answers, final_forecast)