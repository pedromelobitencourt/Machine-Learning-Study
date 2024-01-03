from wordcloud import WordCloud
import matplotlib.pyplot as plt
import bs4 as bs
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import urllib.request

# Seu código existente para obter e processar dados da Wikipedia
data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Machine_learning')
data = data.read()
html_data = bs.BeautifulSoup(data, 'lxml')

paragraphs = html_data.find_all('p')

content = ''

for p in paragraphs:
    content += p.text

content = content.lower()  # Usually we use only lowercase letters

cloud = WordCloud(background_color='white', max_words=100)

nlp = spacy.load('en_core_web_sm')
import en_core_web_sm

nlp = en_core_web_sm.load()
doc = nlp(content)

token_list = []

for token in doc:
    token_list.append(token.text)

no_stop = []

for word in token_list:
    if nlp.vocab[word].is_stop == False:
        no_stop.append(word)

print(len(token_list))
print(len(no_stop))

try:
    cloud = cloud.generate(''.join(no_stop))
    plt.figure(figsize=(15, 15))
    plt.imshow(cloud)
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"Erro ao gerar a nuvem de palavras: {e}")
