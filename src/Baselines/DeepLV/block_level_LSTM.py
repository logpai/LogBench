import yaml
import os
import sys
import re as re
import string

import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.porter import PorterStemmer

import random as rn
seed_value = 17020 
seed_window = 1500
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import tensorflow as tf
import Metrics
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Embedding, LSTM, Bidirectional, Activation, LeakyReLU
from keras.models import model_from_yaml
from keras.utils import np_utils
from keras_self_attention import SeqSelfAttention


import Helper


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 16} ) 
sess = tf.Session(config=config) 
K.set_session(sess)


csv.field_size_limit(100000000)
sys.setrecursionlimit(1000000)
# set parameters:
n_iterations = 1
embedding_iterations = 1
n_epoch = 50

vocab_dim = 100
maxlen = 100
n_exposures = 10
window_size = 7
batch_size = 24
input_length = 100
cpu_count = multiprocessing.cpu_count()

test_list = []
neg_full = []
pos_full = []
syntactic_list = []



model_location = 'model_block'  +'/lstm_'+ sys.argv[1]
embedding_location = 'embedding_block' + '/Word2vec_model_' + sys.argv[1] + '.pkl'


def loadfile():

    data_full=pd.read_csv('block_processing/blocks/logged_syn'  + '_' + sys.argv[1] + '.csv', usecols=[1,2,3,4], engine='python')

    dataset = data_full.values
    classes = dataset[:, 2]
    data=data_full['Values'].values.tolist()
    combined = data
    combined_full = data_full.values.tolist()

    encoder = LabelEncoder()
    encoder.fit(classes)
    encoded_Y = encoder.transform(classes)
    y = Helper.ordinal_encoder(classes)



    x_train, x_test, y_train, y_test = train_test_split(combined_full, y, test_size=0.2, train_size=0.8, random_state=seed_value, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, train_size=0.75, random_state=seed_value, stratify=y_train)
    test_block_list = []
    train_block_list = []
    for x in x_test:
        test_list.append(x[0])
        test_block_list.append(x[1])
    x_test = np.array(test_block_list)
    for x in x_train:
        train_block_list.append(x[1])
    x_train = train_block_list

    return combined,y, x_train, x_val, x_test, y_train, y_val,  y_test



def word_splitter(word, docText):
    splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', word)).split()
    for word in splitted:
        docText.append(word.lower())




def tokenizer(text):
    newText = []
    for doc in text:
        docText = []
        #for word in str(doc).replace("['", "").replace("']", "").replace(",", "").replace("'", "").split(' '):
        for word in str(doc).replace("'", "").replace("[", "").replace("]", "").replace(",", "").replace('"', "").split(' '):
            docText.append(word)
            
        newText.append(docText)
    #print (newText)
    return newText
    


def input_transform(words):
    model=Word2Vec.load(embedding_location)
    _, _,dictionaries=create_dictionaries(model,words)
    return dictionaries






def create_dictionaries(model=None,
                        combined=None):

    from keras.preprocessing import sequence
    
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model.wv[word] for word in w2indx.keys()}

        def parse_dataset(combined):
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec,combined


def word2vec_train(combined):
    model = Word2Vec(size=vocab_dim, #dimension of word embedding vectors
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count, sg=1,
                     iter=embedding_iterations)
    model.build_vocab(combined)
    model.save(embedding_location)
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined


def get_data(index_dict,word_vectors,combined):

    n_symbols = len(index_dict) + 1  
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]


    return n_symbols,embedding_weights


def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test, x_val, y_val):
    
    tf.set_random_seed(seed_value)

    


    model = Sequential()  
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length)) 
    model.add(Bidirectional(LSTM(output_dim=128,activation='sigmoid')))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='sigmoid')) 


    print ('Compiling the Model..')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print ("Train...")
    metrics = Metrics.Metrics()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1, validation_data=(x_val, y_val))

    base_min = optimal_epoch(history)
    print ("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    yaml_string = model.to_yaml()
    with open(model_location +'.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights(model_location + sys.argv[1] + '.h5')
    np.set_printoptions(threshold=sys.maxsize)

    prob_predicted = model.predict(x_test,  verbose=1)
    label_predicted = Helper.predict_prob_encoder(prob_predicted)  
    num_y_test = Helper.pd_encoder(y_test)
    num_y_predicted = Helper.pd_encoder(label_predicted)

    val_accuracy = accuracy_score(y_test, label_predicted)
    print ('Accuracy: ', val_accuracy)
    Helper.class_accuracy(y_test, label_predicted)

    with open(model_location + '_target.txt', 'wt') as f:
        for y in y_test:
            f.write(str(y)+ '\n')
    with open(model_location + '_predicted.txt', 'wt') as f:
        for y in label_predicted:
            f.write(str(y)+ '\n')
    return [val_accuracy]



        

def get_FP_FN(label_predicted, label_target):
    FP_id_list = []
    FN_id_list = []
    for i in range(0, len(label_predicted)):
        if int(label_predicted[i]) == 1 and int(label_target[i]) == 0:
            FP_id_list.append(i)
        elif int(label_predicted[i]) == 0 and int(label_target[i]) == 1:
            FN_id_list.append(i)
    #print (FP_id_list)
    #print (FN_id_list)
    with open('model_block'  +'/labels/list/lstm_FP_' + sys.argv[1] + '.txt', 'wt') as f:
        for fp in FP_id_list:
            f.write(str(test_list[int(fp)])+ '\n')
    with open('model_block' +'/labels/list/lstm_FN_' + sys.argv[1] + '.txt', 'wt') as f:
        for fn in FN_id_list:
            f.write(str(test_list[int(fn)])+ '\n')
        

def train():
    os.environ['PYTHONHASHSEED']=str(seed_value)
    np.random.seed(seed_value)  
    rn.seed(seed_value)
    print ('Loading Data...')
    combined,y,x_train, x_val, x_test, y_train, y_val,  y_test=loadfile()
    print ('Tokenizing...')
    combined = tokenizer(combined)
    x_train = tokenizer (x_train)
    x_test = tokenizer (x_test)
    x_val = tokenizer (x_val)
    print ('Training a Word2vec model...')
    index_dict, word_vectors,combined=word2vec_train(combined)
    x_train = input_transform(x_train)
    x_test = input_transform(x_test)
    x_val = input_transform(x_val)
    print ('Setting up Arrays for Keras Embedding Layer...')
    n_symbols,embedding_weights=get_data(index_dict, word_vectors,combined)
    #print (x_train.shape,y_train.shape)
    result = train_lstm(n_symbols,embedding_weights,x_train,y_train, x_val , y_val , x_test,y_test)
    return result


def pipeline_train(iterations):
    seed_and_result = {}
    if iterations == 1:
        train()
    else:
        for i in range(0, iterations):
            print('Iteration: ', i)
            global seed_value
            result = train()
            seed_and_result[seed_value] = result
            seed_value = seed_value + seed_window
            i = i + 1
    return seed_and_result

def eval_metric(model, history, metric_name):
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]
    e = range(1, n_epoch + 1)
    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.xlabel('Epoch number')
    plt.ylabel(metric_name)
    plt.title('Comparing training and validation ' + metric_name + ' for ' + model.name)
    plt.legend()
    plt.show()


def optimal_epoch(model_hist):
    min_epoch = np.argmin(model_hist.history['val_loss']) + 1
    print("Minimum validation loss reached in epoch {}".format(min_epoch))
    return min_epoch




if __name__=='__main__':
    result_dict = pipeline_train(n_iterations)
    print (sys.argv[1])
