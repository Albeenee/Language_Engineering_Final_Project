import nltk
from torch._C import *
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs

class TrigramTrainer(object):
    """
    This class constructs a trigram language model from a corpus.
    """

    def process_files(self, f):
        """
        Processes the file @code{f}.
        """
        char = ['!', '"', '', '$', '%', '&', "'", '(', ')', ',', '-', '.','/', ':', ';', '', '=','+', '', '?', '','.']+[chr(letter) for letter in range(65, 91)]

        if os.path.isdir( f ) :
            for root,dirs,files in os.walk( f) :
                for file in files :
                    self.process_files( os.path.join(root, file ))
        else :
            print( f )
            stream = open( f, mode='r', encoding='utf-8', errors='ignore' )
            text = stream.read()
            try :
                # self.tokens = nltk.word_tokenize(text) 
                self.tokens = [token.lower() for token in nltk.word_tokenize(text) if any(c in token for c in char)]
            except LookupError :
                nltk.download('punkt')
                self.tokens = nltk.word_tokenize(text)
            for i, token in enumerate(self.tokens) :
                self.process_token(token)


    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram, bigram and
        trigram counts.

        :param token: The current word to be processed.
        """
        # YOUR CODE HERE
        
        
        self.total_words+=1
        if token not in self.index:
        
            self.index[token]=len(self.index)
            self.word[len(self.index)-1]=token
        
        if self.last_index==-1:
            self.unigram_count[token] = self.unigram_count[token]+1
            
        elif self.last_index<1:
       
            self.unigram_count[token] = self.unigram_count[token]+1
           
            self.bigram_count[self.word[self.last_index]][token] = self.bigram_count[self.word[self.last_index]][token]+1
        
        elif self.last_index>0:
            self.unigram_count[token] = self.unigram_count[token]+1
           
            self.bigram_count[self.word[self.last_index]][token] = self.bigram_count[self.word[self.last_index]][token]+1
            self.trigram_count[self.word[self.last_index-1]][self.word[self.last_index]][token] = self.trigram_count[self.word[self.last_index-1]][self.word[self.last_index]][token]+1
        self.last_index=self.index[token]   

        
         


    def stats(self):
        """
        Creates a list of rows to print of the language model.

        """
        
        rows_to_print = []

        # YOUR CODE HERE
        rows_to_print.append(str(len(self.unigram_count))+" "+str(self.total_words))
        for index,token in enumerate(self.unigram_count):
            rows_to_print.append(str(self.index[token])+" "+str(token)+" "+str(self.unigram_count[token]))
        
        for index,token in enumerate(self.bigram_count):
            
            for token2 in self.bigram_count[token]:
                
                
                stat=math.log(self.bigram_count[token][token2]/self.unigram_count[token])

                rows_to_print.append(str(token)+" "+str(token2)+" "+"%.15f" % stat)

        
        for index,token in enumerate(self.trigram_count):

            for token2 in self.trigram_count[token]:

                for token3 in self.trigram_count[token][token2]:
                    if self.bigram_count[token][token2]!=0:
                        stat = math.log(self.trigram_count[token][token2][token3]/self.bigram_count[token][token2])
                        rows_to_print.append(str(token)+" "+str(token2)+" "+str(token3)+" "+ "%.15f" % stat)
            
        rows_to_print.append(str(-1))
        return rows_to_print
    

    def predict(self, tokens):
        if tokens[0] not in self.index:
            print("Word not in vocabulary")
            
            return 'error'
        if len(tokens)==1:
            max_prob=max(self.bigram_count[tokens[0]],key=self.bigram_count[tokens[0]].get)
            
            return max_prob
        if len(tokens)>1:
            if tokens[1] not in self.index:
                print("Word not in vocabulary")
                return 'error'
           
            if len(tokens)>=2:
                try:

                    max_prob=max(self.trigram_count[tokens[-2]][tokens[-1]],key=self.trigram_count[tokens[-2]][tokens[-1]].get)
                except ValueError:
                    try:
                        max_prob=max(self.bigram_count[tokens[-1]],key=self.bigram_count[tokens[-1]].get)
                    except ValueError:
                        max_prob=max(self.unigram_count,key=self.unigram_count.get)
                    
                return max_prob
        
    def __init__(self):
        """
        <p>Constructor. Processes the file <code>f</code> and builds a language model
        from it.</p>

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        # Bigram end trigram counts
        self.bigram_count = defaultdict(lambda: defaultdict(int))
        self.trigram_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # The identifier of the previous word processed.
        self.last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        self.laplace_smoothing = True
        


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTrainer')
    parser.add_argument('--file', '-f', type=str,  default='trigram/data', help='The files used in the training.')
    parser.add_argument('--destination','-d', default='ngram.txt', type=str, help='file in which to store the language model')
    parser.add_argument('--load','-l', default='ngram.txt', type=str, help='which file to load')

    arguments = parser.parse_args()
    """  file = 'project_language_engineering/harry1.txt'
    destination = 'ngram1.txt' """
    bigram_trainer = TrigramTrainer()

    bigram_trainer.process_files(arguments.file)

    stats = bigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
            for row in stats: f.write(row + '\n')
    else:
        for row in stats: print(row)
        
        
  # ==================== User interaction ==================== #       
    while( True ) :
        text = input( "> " )
        if text == "" :
            continue
    
        tokens = nltk.word_tokenize(text)
        pred=bigram_trainer.predict(tokens)

        text = text + " " + pred
        print( text )
        while True:



            inp=input("next word or correct word(enter): ")
   
            if inp=="":
                tokens.append(pred)
                
                pred=bigram_trainer.predict(tokens)
                text = " ".join(tokens) + " " + pred
                print( text )
            else :
                tokens.append(inp)
                pred=bigram_trainer.predict(tokens)
                text =" ".join(tokens) + " " + pred
                print( text )
            
            
            
            


if __name__ == "__main__":
    main()
    