import pandas as pd
import nltk
from nltk import pos_tag 
import xlsxwriter
from nltk.tokenize import word_tokenize 
#####################################
workbook = xlsxwriter.Workbook('pos.csv')
worksheet = workbook.add_worksheet()
###############################
df = pd.read_csv('2.csv',header=None)

###############################

def chunking(text, grammar): 
    word_tokens = word_tokenize(text)   
    word_pos = pos_tag(word_tokens) 
    chunkParser = nltk.RegexpParser(grammar) 
    tree = chunkParser.parse(word_pos) 
      
    for subtree in tree.subtrees():
      
       return(subtree)       
################################
df[5] = df[4].astype(str)
grammar = "NP: {<DT>?<JJ>*<NN>}"
words=df[5]
id=0
for w in words:
     worksheet.write(id, 1, str( chunking(w, grammar)))
     id=id+1              
workbook.close()









  



