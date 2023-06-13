import argparse
import codecs
from collections import defaultdict
import nltk
import codecs
from trigram.trigram_tester import NgramTester

nltk.download('punkt')

#read the trigram model
ngram = NgramTester()
ngram.read_model('trigram/ngrams/ngram 2.txt')


# evaluation function
def evaluate_keystrokes(filename,ngram):
    print('start evaluation')
    keystrokes=0
    characters=0
    
    with codecs.open(filename, 'r', 'utf-8',errors='ignore') as f:
        lines = f.read().split('\n')

        
        i=0
        for line in lines :
            #print(i)
            sentence=''
            for word in nltk.word_tokenize(line):
                # print(word,len(word))
                characters+=len(word)
                if nltk.word_tokenize(line)!=[]:
                    if ngram.predict(nltk.word_tokenize(line))[0]==word:
                        sentence+=word+' '

                    else:
                        new_char=''
                        test=0

                        for char in word:
                            keystrokes+=1
                            new_char+=char
                            tokens = nltk.word_tokenize(sentence+new_char)
                            pred = ngram.predict_end_word(tokens)

                            if pred!=None and pred!=[]:
                                if pred[0]==word.lower():
                                    sentence+=word+' '
                                    keystrokes+=1
                                    test=len(word)
                                    print('k',keystrokes)
                                    print('c',characters)
                                    break
            i+=1
                     
    saved_keystrokes=characters-keystrokes
    print('saved key strokes : ', saved_keystrokes)
    print('total number of characters : ', characters)
    print('percentage saved keystrokes : ', saved_keystrokes/characters)
    return saved_keystrokes/characters,saved_keystrokes,characters


def main():
    evaluate_keystrokes('evaluation_data/test 3.txt', ngram)



if __name__ == "__main__":
    main()
                    
        
        