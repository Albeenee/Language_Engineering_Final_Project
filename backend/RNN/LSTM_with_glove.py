from datetime import datetime
import argparse
import random
import pickle
import codecs
import json
import os
import nltk
import torch
import numpy as np
from pprint import pprint
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader




# ==================== Datasets ==================== #

# Mappings between symbols and integers, and vice versa.
# They are global for all datasets.
source_w2i = {}
source_i2w = []
target_w2i = {}
target_i2w = []

# The padding symbol will be used to ensure that all tensors in a batch
# have equal length.
PADDING_SYMBOL = ' '
source_w2i[PADDING_SYMBOL] = 0
source_i2w.append( PADDING_SYMBOL )
target_w2i[PADDING_SYMBOL] = 0
target_i2w.append( PADDING_SYMBOL )

# The special symbols to be added at the end of strings
START_SYMBOL = '<START>'
END_SYMBOL = '<END>'
UNK_SYMBOL = '<UNK>'
source_w2i[START_SYMBOL] = 1
source_i2w.append( START_SYMBOL )
target_w2i[START_SYMBOL] = 1
target_i2w.append( START_SYMBOL )
source_w2i[END_SYMBOL] = 2
source_i2w.append( END_SYMBOL )
target_w2i[END_SYMBOL] = 2
target_i2w.append( END_SYMBOL )
source_w2i[UNK_SYMBOL] = 3
source_i2w.append( UNK_SYMBOL )
target_w2i[UNK_SYMBOL] = 3
target_i2w.append( UNK_SYMBOL )

# Max number of words to be predicted if <END> symbol is not reached
MAX_PREDICTIONS = 20


def load_glove_embeddings(embedding_file) :
    """
    Reads pre-made embeddings from a file
    """
    N = len(source_w2i)
    embeddings = [0]*N
    with codecs.open(embedding_file, 'r', 'utf-8') as f:
        for line in f:
            data = line.split()
            word = data[0].lower()
            if word not in source_w2i:
                source_w2i[word] = N
                source_i2w.append(word)
                N += 1
                embeddings.append(0)
            vec = [float(x) for x in data[1:]]
            D = len(vec)
            embeddings[source_w2i[word]] = vec
    # Add a '0' embedding for the padding symbol
    embeddings[0] = [0]*D
    # Check if there are words that did not have a ready-made Glove embedding
    # For these words, add a random vector
    for word in source_w2i :
        index = source_w2i[word]
        if embeddings[index] == 0 :
            embeddings[index] = (np.random.random(D)-0.5).tolist()
    return D, embeddings




class create_dataset(Dataset):
    def __init__(self, filename,num_layers,record_symbols):
        self.source_list = []
        self.target_list = []
        with codecs.open(filename, 'r', 'utf-8',errors='ignore') as f:
            lines = f.read().split('\n')
            for line in lines :
                
                source_sentence=[0 for i in range(num_layers-1)]
                for w in list(nltk.word_tokenize(line)):
                    if (w not in source_i2w) and record_symbols :
                        source_w2i[w] = len(source_i2w)
                        source_i2w.append( w )
                    source_sentence.append( source_w2i.get(w, source_w2i[UNK_SYMBOL]) )
                   
                for index in range(len(list(source_sentence))-num_layers):
                    self.source_list.append(source_sentence[index:index+num_layers])
                    self.target_list.append(source_sentence[index+num_layers])
                    
    def __len__(self) :
        return len(self.source_list)

    def __getitem__(self, idx) :
        return self.source_list[idx], self.target_list[idx]                
                        

                    
                    
                    
        
    
class LSTMModel(nn.Module):
    def __init__(self,embedding_size,no_of_outputs, hidden_size,device,embeddings=None,):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.gru = nn.LSTM(
            input_size=embedding_size, 
            hidden_size=hidden_size, 
            
            batch_first=True,
            
        )
        self.embedding = nn.Embedding(no_of_outputs,embedding_size)
        if embeddings !=  None :
            self.embedding.weight = nn.Parameter( torch.tensor(embeddings, dtype=torch.float), requires_grad=True )
        self.fc = nn.Linear(hidden_size, no_of_outputs)
        self.device=device
        self.to(device)
 
    def forward(self, x):
        embedded_x=self.get_embedding(x)
        output, hidden_state = self.gru(embedded_x)
        output = self.fc(hidden_state[0])
        return output
    
    def get_embedding(self,word):
        embed=self.embedding.weight[word]
        return embed
    
    def predict(self,sentence, source_w2i, source_i2w):
        sentence = sentence.split()
        sentence = [source_w2i.get(w, source_w2i[UNK_SYMBOL]) for w in sentence]
        if len(sentence) >= 10 :
            sentence=sentence[-10:]
        else:
            sentence=[0 for i in range(10-len(sentence))]+sentence
        sentence = torch.tensor(sentence).unsqueeze(0)
        output = self(sentence)
        output = torch.topk(output,3)

        output=output[1].squeeze(0).squeeze(0).tolist()
        output=[source_i2w[i] for i in output]
        return output
    
    def predict_endword(self,sentence,source_w2i,source_i2w):
        sentence = sentence.split()
        last_char=sentence[-1]
        sentence=sentence[:-1]
        sentence = [source_w2i.get(w, source_w2i[UNK_SYMBOL]) for w in sentence]

        if len(sentence) >= 10 :
            sentence=sentence[-10:]
        else:
            sentence=[0 for i in range(10-len(sentence))]+sentence
        sentence = torch.tensor(sentence).unsqueeze(0)
        output = self(sentence)
        matching_keys = []
        for key in source_w2i.keys():
            if key.startswith(last_char):
                matching_keys.append(source_w2i[key])
        intersection=output.squeeze(0).squeeze(0)[matching_keys]
        if len(intersection)==0:
            return [None]
        
        output = torch.topk(intersection,min(3,len(intersection)))
    
        
        output=output[1].squeeze(0).squeeze(0).tolist()
        if type(output)==int:
            output=[output]
            
        indexes=[matching_keys[i] for i in output]
        output=[source_i2w[i] for i in indexes]
        return output
    
def evaluate(devtest,grumodel):
    
    correct_pred, incorrect_pred,counter= 0,0,0
    
    for x,y in devtest:
        
        output=grumodel(torch.tensor(x).unsqueeze(0))
        pred=torch.argmax(output)
        counter+=1
        if pred==y:
            correct_pred+=1
        else:
            incorrect_pred+=1
    return correct_pred/counter
            

    
def load_model(path,source_w2i,source_i2w,device):
    source_w2i = pickle.load(open(os.path.join(path, "source_w2i"), 'rb'))
    source_i2w = pickle.load(open(os.path.join(path, "source_i2w"), 'rb'))
    settings = json.load(open(os.path.join(path, "settings.json")))
    model= LSTMModel(

        no_of_outputs=len(source_i2w),
        embedding_size=settings['embedding_size'],
        hidden_size=settings['hidden_size'],
        device=device,
        )
    model.load_state_dict(torch.load(os.path.join(path, "encoder.model")))
    return model,source_w2i,source_i2w

def evaluate_keystrokes(filename,grumodel):
    keystrokes=0
    
    with codecs.open(filename, 'r', 'utf-8',errors='ignore') as f:
        lines = f.read().split('\n')

        characters=0
        sentence=''
        for line in tqdm(lines):
            for word in nltk.word_tokenize(line):
                characters+=len(word)
                word=word.lower()
                if grumodel.predict(sentence)[0]==word:
                    sentence+=word+' '
                else:

                    new_char=''
                    for char in word.lower():
                        keystrokes+=1
                        new_char+=char

                        pred=grumodel.predict_endword(sentence+new_char)[0]
                        pred=str(pred).lower()
                        if pred==None:
                            keystrokes+=len(word)-len(new_char)
                            break
                        if pred==word or len(new_char)==len(word):
                            sentence+=word+' '
                            break
                     
    saved_keystrokes=characters-keystrokes
    return saved_keystrokes/characters,saved_keystrokes,characters
                        


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-tr', '--train', default='project_language_engineering/rnn/train.txt', help='A training file')
    parser.add_argument('-de', '--dev', default='project_language_engineering/rnn/eval.txt', help='A test file')
    parser.add_argument('-te', '--test', default='project_language_engineering/rnn/test.txt', help='A test file')
    parser.add_argument('-ef', '--embeddings', default='project_language_engineering/rnn/glove.6B.50d.txt', help='A file with word embeddings')
    parser.add_argument('-et', '--tune-embeddings', action='store_true',default=True, help='Fine-tune GloVe embeddings')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-hs', '--hidden_size', type=int, default=100, help='Size of hidden state')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('-sl', '--sentence_length', type=int,default=10, help='Number of epochs')
    parser.add_argument('-ev', '--eval',default=False, help='evaluate the model ')
    parser.add_argument('-k', '--key',default=True, help='calculate the keystrokes saved ')


    parser.add_argument( '-s', '--save', action='store_true', default=True,help='Save model' )
    parser.add_argument( '-l', '--load', type=str,default='project_language_engineering/LSTMmodel',help="The directory with encoder and decoder models to load")
#default= 'project_language_engineering/model_2023-05-24_19_30_34_147091',
    args = parser.parse_args()
    device = torch.device("mps")
    if args.load:

        model,source_w2i,source_i2w=load_model(args.load,source_w2i,source_i2w,device)
    else:
            # ==================== Training ==================== #
        # Reproducibility
        # Read a bit more here -- https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(5719)
        np.random.seed(5719)
        torch.manual_seed(5719)
        torch.use_deterministic_algorithms(True)
        training_dataset = create_dataset( args.train,args.sentence_length,record_symbols=True)
        dev_dataset = create_dataset( args.dev, args.sentence_length,record_symbols=True )

        if args.embeddings :
            embedding_size, embeddings = load_glove_embeddings( args.embeddings )



        # Read datasets


        print( "Number of source words: ", len(source_i2w) )
        print( "Number of target words: ", len(target_i2w) )
        print( "Number of training sentences: ", len(training_dataset) )
        print()

        # If we have pre-computed word embeddings, then make sure these are used


        training_loader = DataLoader(training_dataset, batch_size=args.batch_size)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
        
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.NLLLoss()

        model= LSTMModel(
            embeddings=embeddings,
            embedding_size=embedding_size,
            device=device,
            no_of_outputs=len(source_i2w),
        
            hidden_size=args.hidden_size,


        )


        encoder_optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


        model.train()

        print( datetime.now().strftime("%H:%M:%S"), "Starting training." )

        for epoch in range( args.epochs ) :
            total_loss = 0
            for source, target in tqdm(training_loader): #tqdm(training_loader, desc="Epoch {}".format(epoch + 1)):
                encoder_optimizer.zero_grad()

                loss = 0
                # hidden is (D * num_layers, B, H)
                outputs =model( torch.stack(source).T )


                loss+=criterion(outputs.squeeze(),torch.tensor(target).to(device))
                

                loss.backward()
                encoder_optimizer.step()

            total_loss += loss/args.batch_size
            print( datetime.now().strftime("%H:%M:%S"), "Epoch", epoch, "loss:", total_loss.detach().item() )
            
        if ( args.save ) :
                dt = str(datetime.now()).replace(' ','_').replace(':','_').replace('.','_')
                newdir = 'model_' + dt
                os.mkdir( newdir )
                torch.save( model.state_dict(), os.path.join(newdir, 'encoder.model') )
                
                with open( os.path.join(newdir, 'source_w2i'), 'wb' ) as f :
                    pickle.dump( source_w2i, f )
                    f.close()
                with open( os.path.join(newdir, 'source_i2w'), 'wb' ) as f :
                    pickle.dump( source_i2w, f )
                    f.close()


                settings = {
                    'training_set': args.train,
                    'test_set': args.test,
                    'epochs': args.epochs,
                    'learning_rate': args.learning_rate,
                    'batch_size': args.batch_size,
                    'hidden_size': args.hidden_size,
                    'sentence_length': args.sentence_length,

                    'embedding_size': embedding_size,

                    'tune_embeddings': args.tune_embeddings
                }
                with open( os.path.join(newdir, 'settings.json'), 'w' ) as f:
                    json.dump(settings, f)
    if args.eval:
        model.eval()
        test_dataset=create_dataset(args.test,settings['sentence_length'],record_symbols=False)
    
        print('prediction accuracy:',evaluate(test_dataset,model))
    if args.key:
        model.eval()
        percentage,saved_keystrokes,full_characters=evaluate_keystrokes(args.test,model)
        print('percentage of keystrokes saved:',percentage)
        print('saved keystrokes:',saved_keystrokes)
        print('amount of characters:',full_characters)

    # ==================== User interaction ==================== #       
    while( True ) :
        text = input( "> " )
        if text == "" :
            continue
    
        tokens = nltk.word_tokenize(text)
        pred=model.predict(tokens)

        text = text + " " + pred
        print( text )
        while True:



            inp=input("next word or correct word(enter): ")
   
            if inp=="":
                tokens.append(pred)
                
                pred=model.predict(tokens)
                text = " ".join(tokens) + " " + pred
                print( text )
            else :
                tokens.append(inp)
                pred=model.predict(tokens)
                text =" ".join(tokens) + " " + pred
                print( text )
    