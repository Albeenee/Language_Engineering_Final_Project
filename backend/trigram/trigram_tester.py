import argparse
import codecs
from collections import defaultdict
import nltk
import nltk
import codecs
import heapq
import numpy as np


# TRIGRAM WITH WORDS FOR KEYS
class NgramTester(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):
    
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_count = defaultdict(dict)

        # The trigram log-probabilities.
        self.trigram_count = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0

        


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        print('start reading model')

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                ind=0
                for line in f:
                    line = line.strip()
                    if line=='-1':
                        break

                    if ind<self.unique_words:
                        
                        if line:
                            index,word, count= line.split(' ')
                            index = int(index)
                            self.index[word] = index
                            self.word[index] = word
                            self.unigram_count[word] = int(count)

                    elif len(line.split())==3:
                        
                        if line:
                            indone, indtwo, prob = line.split(' ')
                            # indone = int(indone)
                            # indtwo = int(indtwo)
                            self.bigram_count[indone][indtwo]= float(prob)
                    
                    else:
                        if line:
                            indone, indtwo, indthree, prob= line.split(' ')
                            # indone = int(indone)
                            # indtwo = int(indtwo)
                            # indthree = int(indthree)
                            self.trigram_count[indone][indtwo] = defaultdict(dict)
                            self.trigram_count[indone][indtwo][indthree]= float(prob)

                    ind+=1      
                
                print('model read')
                return True
            
        except IOError:
            print("Couldn't find ngram probabilities file {}".format(filename))
            return False
    

    def predict_end_word(self, tokens):

        if len(tokens)==1:
            unigram_keys = np.array(list(self.unigram_count.keys()))
            words = unigram_keys[np.char.startswith(unigram_keys, tokens[-1].lower())].tolist()
            pred = heapq.nlargest(3,words,key=self.unigram_count.get)
            return pred

        if len(tokens)==2:
            try:
                token = tokens[-2]
                bigram_keys = np.array(list(self.bigram_count[token].keys()))
                try:
                    words = np.char.startswith(bigram_keys, tokens[-1])
                    words_ind = np.array(self.bigram_count[token])[words]
                    pred = heapq.nlargest(3,words_ind,key=self.bigram_count[token].get)
                    return pred
                except TypeError:
                    pass

            except IndexError:
                unigram_keys = np.array(list(self.unigram_count.keys()))
                words = unigram_keys[np.char.startswith(unigram_keys, tokens[-1].lower())].tolist()
                pred = heapq.nlargest(3,words,key=self.unigram_count.get)
                return pred
        
        if len(tokens)>2:

            try :
                token1=tokens[-3]
                token2=tokens[-2]
                trigram_keys = np.array(list(self.trigram_count[token1][token2].keys()))
                words = np.char.startswith(trigram_keys, tokens[-1])
                words_ind = np.array(self.trigram_count[token1][token2])[words]
                pred = heapq.nlargest(3,words_ind,key=self.trigram_count[token1][token2].get)
                return pred
            
            except (KeyError, IndexError):

                try:
                    bigram_keys = np.array(list(self.bigram_count[tokens[-2]].keys()))
                    words = np.array([str(key).startswith(str(tokens[-1])) for key in bigram_keys], dtype=bool)
                    words_ind = np.array(self.bigram_count[tokens[-2]])[words]
                    pred = heapq.nlargest(3,words_ind,key=self.bigram_count[tokens[-2]].get)
                    return pred
                
                except IndexError:
                    unigram_keys = np.array(list(self.unigram_count.keys()))
                    words = unigram_keys[np.char.startswith(unigram_keys, tokens[-1].lower())].tolist()
                    pred = heapq.nlargest(3,words,key=self.unigram_count.get)
                    return pred

        

    

    def predict(self, tokens):
        # if tokens[-1] not in self.index:
        #     pass

        three_best = []

        if len(tokens) == 0:
            max_prob = heapq.nlargest(3, self.unigram_count, key=self.unigram_count.get)
            three_best.extend(max_prob)

        if len(tokens) == 1:

            try:
                word_ind_0 = tokens[-1]
                max_prob = heapq.nlargest(3,self.bigram_count[word_ind_0], key=self.bigram_count[word_ind_0].get)
                three_best.extend(max_prob)
                if len(three_best) < 3:
                    max_prob = heapq.nlargest(3-len(three_best), self.unigram_count, key=self.unigram_count.get)
                    three_best.extend(max_prob)
            
            except KeyError:
                # print('keyerr')
                three_best = heapq.nlargest(3, self.unigram_count, key=self.unigram_count.get)
                pass


        if len(tokens) > 1:
            if tokens[-2] not in self.index:
                pass

            if len(tokens) >= 2:
                try:
                    ind_1 = tokens[-2]
                    ind_2 = tokens[-1]

                    if ind_1 in self.trigram_count and ind_2 in self.trigram_count[ind_1]:
                        max_prob = heapq.nlargest(3, self.trigram_count[ind_1][ind_2], key=self.trigram_count[ind_1][ind_2].get)
                        three_best.extend(max_prob)

                    if len(three_best) < 3 and ind_2 in self.bigram_count:
                        max_prob = heapq.nlargest(3-len(three_best), self.bigram_count[ind_2], key=self.bigram_count[ind_2].get)
                        three_best.extend(max_prob)

                    if len(three_best) < 3:
                        max_prob = heapq.nlargest(3-len(three_best), self.unigram_count, key=self.unigram_count.get)
                        three_best.extend(max_prob)
                
                except KeyError:
                    pass

                # Pad the list with None values if it has fewer than 3 elements
                three_best += [None] * (3 - len(three_best))

        return three_best[:3]


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='NgramTester')
    parser.add_argument('--file', '-f', default='ngram.txt', type=str,  required=False, help='file with language model')

    arguments = parser.parse_args()

    ngram = NgramTester()
    ngram.read_model(arguments.file)



    arguments = parser.parse_args()
        
  # ==================== User interaction ==================== #       
    while( True ) :
        text = input( "> " )
        if text == "" :
            continue
    
        tokens = nltk.word_tokenize(text)
        #pred = ngram.word[ngram.predict(tokens)]
        pred = [ngram.word[i] for i in ngram.predict(tokens)]

        text = text + " " + pred[0]
        print( text, pred )

        while True:

            inp=input("next word or correct word(enter): ")
   
            if inp == "":
                tokens.append(pred[0])
                
                pred = [ngram.word[i] for i in ngram.predict(tokens)]
                text = " ".join(tokens) + " " + pred[0]
                print(text, pred)
            else :
                tokens.append(inp)
                pred = [ngram.word[i] for i in ngram.predict(tokens)]
                text = " ".join(tokens) + " " + pred[0]
                print( text, pred )
            

if __name__ == "__main__":
    main()
