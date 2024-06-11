import nltk
import re
from nltk.corpus import PlaintextCorpusReader
import xlsxwriter
workbook = xlsxwriter.Workbook('corpus.xlsx')
worksheet = workbook.add_worksheet()
#######################################
corpus_root = 'C:/Users/saeed/Desktop/saeed taheri-ir/corpus'
filelists = PlaintextCorpusReader(corpus_root, 'TIME.ALL')
c=filelists.raw()
print('total number of words')
print(len(c.split()))
print('number of unique terms')
print(len(set(c.split())))

id=0
for word in c.split("*TEXT"):
         worksheet.write(id,0,id)
         worksheet.write(id,1,str(word[:4]))
         worksheet.write(id,2,str(word[4:14]))
         worksheet.write(id,3,str(word[14:23]))
         worksheet.write(id,4,str(word[24:]))
         
         id=id+1        
workbook.close()
