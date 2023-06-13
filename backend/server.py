from flask import Flask, request, jsonify
import json
from collections import defaultdict
import nltk
import pickle
import os
import torch
import codecs

#from trigram.trigram import NgramTester
from trigram.trigram_tester import NgramTester
from RNN.LSTM_with_glove import LSTMModel
from RNN.GRU_with_glove import GRUModel

nltk.download('punkt')

#load trigram
ngram = NgramTester()
ngram.read_model('trigram/ngrams/ngram 2.txt')


#load gru
device = torch.device("cpu")
path = 'RNN/GRU_model'
source_w2i_gru = pickle.load(open(os.path.join(path, "source_w2i"), 'rb'))
source_i2w_gru = pickle.load(open(os.path.join(path, "source_i2w"), 'rb'))
settings = json.load(open(os.path.join(path, "settings.json")))
gru= GRUModel(

    no_of_outputs=len(source_i2w_gru),
    embedding_size=settings['embedding_size'],
    hidden_size=settings['hidden_size'],
    device=device
    )


gru.load_state_dict(torch.load(os.path.join(path, "encoder.model"),map_location=device))


#load lstm
device = torch.device("cpu")
path = 'RNN/LSTM_model'
source_w2i_lstm = pickle.load(open(os.path.join(path, "source_w2i"), 'rb'))
source_i2w_lstm = pickle.load(open(os.path.join(path, "source_i2w"), 'rb'))
settings = json.load(open(os.path.join(path, "settings.json")))
lstm= LSTMModel(

    no_of_outputs=len(source_i2w_lstm),
    embedding_size=settings['embedding_size'],
    hidden_size=settings['hidden_size'],
    device=device
    )


lstm.load_state_dict(torch.load(os.path.join(path, "encoder.model"),map_location=device))


app = Flask(__name__)

############## TRIGRAM ROUTE ############## 
@app.route("/trigram", methods = ['POST'])

def trigram():
    #get entered text
    data = request.get_json()
    enteredText = data['input']
    guesstype = data['guesstype']


    #process the text
    
    pred = ""
    res = {'tokens':"", "pred":""}

    if len(enteredText)>0:
        tokens = nltk.word_tokenize(enteredText)

        if guesstype =='nextword':
            print(pred)
            pred = ngram.predict(tokens)
            if type(pred[0])!=str:
                pred = [ngram.word[i] for i in ngram.predict(tokens)]
            res = {'tokens':tokens, 'pred':pred}
            
        elif guesstype=='endword':
            #pred = [ngram.word[i] for i in ngram.predict_end_word(tokens)]
            pred = ngram.predict_end_word(tokens)
            if type(pred[0])!=str:
                pred = [ngram.word[i] for i in ngram.predict_end_word(tokens)]
            res = {'tokens':tokens, 'pred':pred}
    

    return jsonify(res)



############## GRU ROUTE ############## 
@app.route("/gru", methods = ['POST'])

def gru_route():

    #get entered text
    data = request.get_json()
    enteredText = data['input']
    guesstype = data['guesstype']


    #process the text
    
    pred = ['test','test','test']
    res = {'tokens':"", "pred":""}
    if len(enteredText)>0:
        print(enteredText)

        if guesstype =='nextword':
            # guess next word
            print('yes')
            pred=gru.predict(enteredText,source_w2i_gru, source_i2w_gru)
            res = {'pred':pred}
            
        elif guesstype=='endword':
            # guess end of word
            pred=gru.predict_endword(enteredText,source_w2i_gru, source_i2w_gru)
            res = {'pred':pred}
    

    return jsonify(res)


############## LSTM ROUTE ############## 
@app.route("/lstm", methods = ['POST'])

def lstm_route():

    #get entered text
    data = request.get_json()
    enteredText = data['input']
    guesstype = data['guesstype']


    #process the text
    
    pred = ['test','test','test']
    res = {'tokens':"", "pred":""}
    if len(enteredText)>0:
        print(enteredText)

        if guesstype =='nextword':
            # guess next word
            print('yes')
            pred=lstm.predict(enteredText,source_w2i_lstm, source_i2w_lstm)
            res = {'pred':pred}
            
        elif guesstype=='endword':
            # guess end of word
            pred=lstm.predict_endword(enteredText,source_w2i_lstm, source_i2w_lstm)
            res = {'pred':pred}
    

    return jsonify(res)


if __name__=="__main__":
    app.run(debug=True)