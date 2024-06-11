import sys

filename = sys.argv[1]
fp = open("New Text Document.txt")
data = fp.read()
words = data.split()
fp.close()

unwanted_chars = ".,-_ (and so on)"
wordfreq = {}
for raw_word in words:
    word = raw_word.strip(unwanted_chars)
    if word not in wordfreq:
        wordfreq[word] = 0 
    wordfreq[word] += 1
