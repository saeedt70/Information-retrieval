import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from num2words import num2words
import numpy as np
from nltk.tokenize import word_tokenize
#####################################read corpus
df = pd.read_csv('corpus.csv',header=None)
ds = pd.read_csv('TIME.STP',header=None)
corpus=df[4]
print('total number of words')
print(corpus.apply(len))
print('sum total number of words')
print(corpus.apply(len).sum())
print('number of unique terms')
print(corpus.apply(set).apply(len))
print('sum number of unique terms')
print(corpus.apply(set).apply(len).sum())
####lower_case
def convert_lower_case(data):
    return np.char.lower(data)
####remove_punctuation
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data  
####stop_words
def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text
####remove_apostrophe
def remove_apostrophe(data):
    return np.char.replace(data, "'", "")
####convert_numbers
def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text  
#### this function tokenized
def get_tokenized_list(doc_text):
    tokens = nltk.word_tokenize(doc_text)
    return tokens
####stemming
def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text
#####################Preprocessing on our Corpus
cleaned_corpus = []
for doc in enumerate(corpus):
    data = convert_lower_case(str(doc))
    data = remove_punctuation(data) 
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    doc_text = ' '.join(data)
    cleaned_corpus.append(data)


################################transform it to vector
vectorizerX = TfidfVectorizer()
vectorizerX.fit(cleaned_corpus)
doc_vector = vectorizerX.transform(cleaned_corpus)
print('\nWord indexes:')
print(vectorizerX.vocabulary_)  
print('\ntf-idf values:')
print(doc_vector)
####Preprocess the Query and transform it to vector
query = """OPPOSITION OF INDONESIA TO THE NEWLY-CREATED MALAYSIA ."""
rel='[61 155 156 242 269 315 339 358]'
query = convert_lower_case(str(query))
query = remove_punctuation(query)
query = remove_apostrophe(query)
query = remove_stop_words(query)
query = convert_numbers(query)
query = remove_stop_words(query)
query = stemming(query)
query = get_tokenized_list(query)
q = []
for w in query:
 q.append(w)
q = ' '.join(q)
print('Query:')
print(q)
query_vector = vectorizerX.transform([q])
####calculate cosine similarities
cosineSimilarities = cosine_similarity(doc_vector,query_vector).flatten()
related_docs_indices = cosineSimilarities.argsort()[:-11:-1]
print('related docs_id:')
print(related_docs_indices)

#for i in related_docs_indices:
   # data = [cleaned_corpus[i]]
   # print(data)


################################################evalution
def preprocess(s):
    s = s.strip()[1:-1]
    nums = [int(n) for n in s.split(' ') if n.strip()!='']
    return nums

def precision(relevant, retrieved):
    relevant = preprocess(relevant)
    retrieved = preprocess(retrieved)
    return np.shape(np.intersect1d(relevant, retrieved))[0] / np.shape(retrieved)[0]

def recall(relevant, retrieved):
    relevant = preprocess(relevant)
    retrieved = preprocess(retrieved)
    
    return np.shape(np.intersect1d(relevant, retrieved))[0] / np.shape(relevant)[0]
####
def precision1(relevant, retrieved):
    relevant = preprocess(relevant)
    retrieved = preprocess(retrieved)
    r = [(np.shape(np.intersect1d(relevant, retrieved[:i+1]))[0] / np.shape(retrieved[:i+1])[0]) for i in range(0,len(retrieved))]
    return r

def recall1(relevant, retrieved):
    relevant = preprocess(relevant)
    retrieved = preprocess(retrieved)
    r = [(np.shape(np.intersect1d(relevant, retrieved[:i+1]))[0] / np.shape(relevant)[0]) for i in range(0,len(retrieved))]
    return r

a1=precision1(str(rel),str(related_docs_indices))
b1=recall1(str(rel),str(related_docs_indices))
a=precision(str(rel),str(related_docs_indices))
b=recall(str(rel),str(related_docs_indices))

print('precision arry:')
print(a1)
print('recall arry:')
print(b1)
print('precision:')
print(a)
print('recall:')
print(b)

#####plot
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
plt.plot(a1,b1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.show()
