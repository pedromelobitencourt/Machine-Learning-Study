# find and classify some text entities, based on the used training database
# used in chatbots

import bs4 as bs
import spacy
import urllib.request

data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Machine_learning')
data = data.read()
html_data = bs.BeautifulSoup(data, 'lxml')

paragraphs = html_data.find_all('p')

content = ''

for p in paragraphs:
    content += p.text

content = content.lower() # usually we work with lowercase only

nlp = spacy.load('en_core_web_sm')
import en_core_web_sm

nlp = en_core_web_sm.load()
doc = nlp(content)

for entity in doc.ents:
    print(entity.text, entity.label_)

from spacy import displacy

displacy.render(doc, style='ent', jupyter=True)