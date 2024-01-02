import bs4 as bs
import spacy
import urllib.request

def prBlue(skk):
    return "\033[34m{}\033[00m".format(skk)

def prBlueWithBackground(skk):
    return "\033[34;43m{}\033[00m".format(skk)

data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Machine_learning')
data = data.read()

print('Data: ', data)
print('==========================================================================================================================')

html_data = bs.BeautifulSoup(data, 'lxml')
print(html_data)
print('==========================================================================================================================')

paragraphs = html_data.find_all('p')

content = ''

for p in paragraphs:
    content += p.text

print(content)
print('==========================================================================================================================')

content = content.lower() # usually we work with lowercase only

nlp = spacy.load('en_core_web_sm')
import en_core_web_sm

nlp = en_core_web_sm.load()

string = 'computer'
search_token = nlp(string)

from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab) # nlp.vocab (the language vocabulary)
matcher.add('SEARCH', None, search_token)

doc = nlp(content)
matches = matcher(doc)
print(matches) # 0: (search id, initial position - word, final position - word)
print('==========================================================================================================================')

print(doc[784:785])
print(doc[784 - 5:785 + 5])
print('==========================================================================================================================')

from IPython.core.display import HTML
from IPython import display

word_number = 50
doc = nlp(content)
matches = matcher(doc)

display.display(display.HTML(f'<h1>{string.upper()}</h1>'))
display.display(display.HTML(f"""<p><strong>Results: </strong>{len(matches)}</p>"""))
print('==========================================================================================================================')

print(string.upper())
print(f'Found results: {len(matches)}')
text = ''

for i in matches:
    start = i[1] - word_number

    if start < 0:
        start = 0

    text += str(doc[start:i[2] + word_number]).replace(string, prBlueWithBackground(string)) + '\n\n'

print(text)