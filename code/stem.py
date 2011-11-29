#! /usr/bin/python
from nltk.stem.porter import PorterStemmer
a = PorterStemmer(); 
for line in open('mapping_new_words.txt') : 
    word = line.strip() ; 
    print word, " ",a.stem_word(word)
    
