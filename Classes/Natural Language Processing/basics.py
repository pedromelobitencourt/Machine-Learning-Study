import bs4 as bs
import nltk
import spacy

# tokens: https://spacy.io/usage/linguistic-features

## POS (part-of-speech) Marking

nlp = spacy.load('en_core_web_sm')
import en_core_web_sm

nlp = en_core_web_sm.load()
document = nlp('I am learning natural language processing')
print(type(document), '\n=============================================================')

for token in document:
    print(token.text, token.pos_)
print('==========================================================================================================================')

## Lemmatization and Stemming
    
for token in document:
    print(token.text, token.lemma_) # lemma: radical
print('==========================================================================================================================')

doc = nlp('find found learning learned drive drove driven')
print([token.lemma_ for token in doc]) # it makes easier to interprete something
print('==========================================================================================================================')

nltk.download('rslp')

stemmer = nltk.stem.RSLPStemmer()
print(stemmer.stem('learn')) # it's harder to use
print('==========================================================================================================================')

for token in document:
    print(token.text, token.lemma_, stemmer.stem(token.text))