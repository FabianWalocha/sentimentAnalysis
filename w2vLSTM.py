from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Activation, Flatten, Conv1D, SpatialDropout1D, MaxPooling1D, AveragePooling1D, Bidirectional, merge, concatenate, Input, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.client import device_lib
from datetime import datetime
import numpy as np
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.preprocessing.text import one_hot
from collections import defaultdict
import re
from gensim.models import Word2Vec
from keras.models import load_model
import string 
    

def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''
    
    text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+','<url>',text)
    text = re.sub(r'[:;xX8][\^-]?[)DPp]+','<happy>',text)
    text = re.sub(r'\^[\.\_-]?\^','<happy>',text)
    text = re.sub(r':\'?[-\^]?[\/(C]+','<sad>',text)
    text = re.sub(r'T[\._-]?T','<sad>',text)
    text = re.sub(r'-[\._-]?-','<sad>',text)
    text = re.sub(r'\b[oO][\.-_,]?[oO]\b','<surpsised>',text)
    text = re.sub(r'-[\._-]?-','<surprised>',text)
    text = re.sub(r':[\^-]?[oO]','<surprised>',text)
    text = re.sub(r'D:','<surprised>',text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"u002c", "", text)
    text = re.sub(r"u2019", "'", text)
    text = re.sub(r"\n", "",  text)
    text = re.sub(r"[-()]", "", text)
    text = re.sub(r"\.", " .", text)
    text = re.sub(r"\!", " !", text)
    text = re.sub(r"\?", " ?", text)
    text = re.sub(r"\,", " ,", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"oh+", "oh", text)
    text = re.sub(r"ah+", "ah", text)
    text = re.sub(r"aa+","a", text)
    text = re.sub(r"soo+\b",'soo', text)
    text = re.sub(r"aint", "is not", text)
    text = re.sub(r"gonna", "going to", text)
    text = re.sub(r"ima", "i am going to", text)
    text = re.sub(r"\/\w+", "", text)
    return text

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def w2vmodel(embedding_size=200, max_words=200, y_dim=1, vocabulary_size=50,
          num_filters=200, filter_sizes=[3,4,5], pool_padding='valid', dropout=0.5):
    embed_input = Input(shape=(max_words,embedding_size))
    pooled_outputs = []
    for i in range(len(filter_sizes)):
        conv = Conv1D(num_filters, kernel_size=filter_sizes[i], padding='valid', activation='relu')(embed_input)
        conv = MaxPooling1D(pool_size=max_words-filter_sizes[i]+1)(conv)           
        pooled_outputs.append(conv)
    merge = concatenate(pooled_outputs)
    
    x = Dense(30, activation='relu')(merge)
    x = Dropout(dropout)(x)
    x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.5, recurrent_dropout=0.1))(x)
    x = Dense(30, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(y_dim, activation='softmax')(x)

    model = Model(inputs=embed_input,outputs=x)

    # model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])
    # print(model.summary())
    
    from keras.utils import plot_model
    plot_model(model, to_file='shared_input_layer.png')
    
    return model


def get_onehot(X, min_count=1):
    # inspired by https://github.com/saurabhrathor/InceptionModel_SentimentAnalysis/
    vocabulary = dict()
    count = defaultdict(int)
    inverse_vocabulary = ['PADDING']
    for text in X:
        text = text.split()
        for word in text:
            count[word]+=1
            if word not in vocabulary:
                vocabulary[word] = len(inverse_vocabulary)
                inverse_vocabulary.append(word)

    max_len = 0
    sequences = []
    for text in X:
        text = text.split()
        text_sequence = []
        for word in text:
          # constraint given term frequency
            if count[word] >= min_count:
                text_sequence.append(vocabulary[word])
                inverse_vocabulary.append(word)
        # save longest tweet length
        if len(text_sequence)>max_len:
            max_len=len(text_sequence)
        sequences.append(text_sequence)
    return sequences, vocabulary, max_len


def score_CNN_LSTM(X_train, y_train, X_val, y_val, X_test, y_test, min_count=3, 
                   epochs = 50, batch_size=32, embedding_size=200, dropout=0.5, verbose=1):
    # inspired by https://github.com/mihirahlawat/Sentiment-Analysis
    # BB_twtr at SemEval-2017 Task 4: Twitter Sentiment Analysis with CNNs and LSTMs
    
    y_train = to_categorical(one_hot(" ".join(y_train),n=3),3)
    y_val = to_categorical(one_hot(" ".join(y_val),n=3),3)
    y_test = to_categorical(one_hot(" ".join(y_test),n=3),3)
    
    X = X_train+X_val+X_test
    Xc = []
    tweet_lengths = []
    for line in X:
        cleaned = clean_text(line)
        Xc.append(cleaned)
        tweet_lengths.append(len(cleaned.split(" ")))
    X = Xc
    
    # Return original indices
    X_train = X[:len(y_train)]
    X_val = X[len(y_train):(len(y_train)+len(y_val))]
    X_test = X[(len(y_train)+len(y_val)):]
    
    # Get vocabulary and one_hot of training data
    sequences, vocabulary, MAX_SEQUENCE_LENGTH = get_onehot(X_train)
    
    window_size = 2
    w2v = Word2Vec([" ".join(X_train).split(" ")+['<$>']],size=200, window=2, min_count=1, workers=4)
    
    bodies_seq = np.zeros([len(X),max(tweet_lengths),embedding_size])
    for idx,tweet in enumerate(X_train):
        for inner,word in enumerate(tweet.split(" ")):
            print(word)
            bodies_seq[idx,inner,:] = w2v[word]
    for idx2, tweet in enumerate(X_val):
        for inner, word in enumerate(tweet.split(" ")):
            try:
                bodies_seq[idx+idx2+1,inner,:] = w2v[word]
            except KeyError:
                bodies_seq[idx+idx2+1,inner,:] = w2v['<$>']
    for idx3, tweet in enumerate(X_test):
        for inner, word in enumerate(tweet.split(" ")):
            try:
                bodies_seq[idx+idx2+idx3+1,inner,:] = w2v[word]
            except:
                bodies_seq[idx+idx2+idx3+1,inner,:] = w2v['<$>']
    
                      
    K.tensorflow_backend._get_available_gpus()

    mdl = w2vmodel(embedding_size=bodies_seq.shape[2], max_words=MAX_SEQUENCE_LENGTH, vocabulary_size=len(vocabulary)+1,
            y_dim=y_train.shape[1],filter_sizes = [3,4,5],dropout=0.1)
    mdl.compile(loss=f1_loss,
                optimizer='adam', 
                metrics=['acc',f1_m,precision_m, recall_m])

    mcp_save = ModelCheckpoint('w2v_saved_model.h5', verbose=0, monitor='val_loss',save_best_only=True, mode='min')

    history = mdl.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, 
              callbacks=[mcp_save])
    
    
    mdl = load_model('cnn_lstm.h5', custom_objects={
                                                      'f1_loss':f1_loss, 
                                                      'f1_m':f1_m,
                                                      'precision_m':precision_m,
                                                      'recall_m':recall_m
                                                       })

    loss, acc, f1, prec, rec = mdl.evaluate(X_test, y_test)
    return loss, acc, f1, prec, rec