from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

sentence = "hi my name is saeed taheri . this is a test"
words = word_tokenize(sentence)

wordsFiltered = []
for w in words:
 if w not in stopWords:
  wordsFiltered.append(w)
print(wordsFiltered)
