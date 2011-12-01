#! /usr/bin/python
from nltk.stem.porter import PorterStemmer
a = PorterStemmer(); 
for line in open('mapping_correct_words.txt') : 
    word = line.strip() ; 
    a.b = word; 
    a.k0 = 0 ; 
    a.k = len(word) - 1 
    a.step1ab()
    result = a.b[a.k0:a.k+1]
    print word, " ",result
    
