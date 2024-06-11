import pandas as pd
from collections import Counter
##############################################
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
import xlsxwriter
workbook = xlsxwriter.Workbook('3.csv')
worksheet = workbook.add_worksheet()
###############################
df = pd.read_csv('corpus.csv',header=None)
###############################
df['BOW'] = df[4].apply(lambda x: Counter(x.split(" ")))
print(df['BOW'])
id=0
words=df['BOW']
for word in words:  
  worksheet.write(id, 1, str(word))
  id=id+1              
workbook.close()





  











