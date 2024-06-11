from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
stemmer = PorterStemmer()
examples = ["cars", "eating", "games","playing","programmer","Stemmers "]

for w in examples:
    #print(stemmer.stem(w))
    print(SnowballStemmer("english").stem(w))
