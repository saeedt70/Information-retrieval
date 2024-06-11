import pandas as pd
import nltk
import xlsxwriter
workbook = xlsxwriter.Workbook('postlist.csv')
worksheet = workbook.add_worksheet()

df = pd.read_csv('corpus.csv',header=None)
docs = df[4].astype(str)
vocab = []
postings = {}
for i,doc in enumerate(docs):

    words = doc.split(" ")
    for word in words:
        if word not in vocab:
            vocab.append(word)
        wordId = vocab.index(word)
        if wordId not in postings:
            postings[wordId] = [i]
        else:
            postings[wordId].append(i)
i=0
for w in enumerate(postings):
    # print(str(vocab[i])+':'+str(postings[i]))
      worksheet.write(i, 0, str(vocab[i]))
      worksheet.write(i, 1, str(postings[i]))
      i=i+1              
workbook.close()
     


