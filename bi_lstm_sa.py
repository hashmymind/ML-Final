from keras.models import Sequential
from keras.layers import Dense, LSTM,Embedding, Bidirectional
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from keras_self_attention import SeqSelfAttention
from data import *
from preprocessing import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle

def get_model(classes=5, word_index=None):
    # pretrain embedding
    try:
        with open('embed.mat','rb') as fp:
            embedding_matrix=pickle.load(fp)    
    
    except:
        embeddings_index = {}
        f = open('glove.6B.300d.txt',encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        
        with open('embed.mat','wb') as fp:
            pickle.dump(embedding_matrix,fp)
            
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,300,weights=[embedding_matrix],input_length=40,trainable=False))
    model.add(Bidirectional(LSTM(300,return_sequences=True, dropout=0.5, recurrent_dropout=0.5)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.5)))
    model.add(Dense(classes,activation='softmax'))
    return model
    
if __name__ == '__main__':
    
    # data
    
    (train_X, train_y), (test_X, test_y) = get_sst('5')
    try:
        with open('train_X.token','rb') as fp:
            train_X=pickle.load(fp)
        with open('test_X.token','rb') as fp:
            test_X=pickle.load(fp)
        with open('word_index','rb') as fp:
            word_index=pickle.load(fp)
    except:
        train_X = preprocess_batch(train_X)
        test_X = preprocess_batch(test_X)
        text = train_X+ test_X
        
        tokenizer = Tokenizer(nb_words=10000)
        tokenizer.fit_on_texts(text)
        train_seq = tokenizer.texts_to_sequences(train_X)
        train_X = pad_sequences(train_seq, maxlen=40)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        test_seq = tokenizer.texts_to_sequences(test_X)
        test_X = pad_sequences(test_seq, maxlen=40)
    
        with open('train_X.token','wb') as fp:
            pickle.dump(train_X,fp)
        with open('test_X.token','wb') as fp:
            pickle.dump(test_X,fp)
        with open('word_index','wb') as fp:
            pickle.dump(word_index,fp)
            
            
    filepath="lstmsa.h5"
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5)
    ck = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model = get_model(5,word_index)
    model.summary()
    

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['sparse_categorical_accuracy'])
    model.fit(train_X,train_y,epochs=20,validation_split=0.2,callbacks=[ck,es])
    
    
    # testing
    model.load_weights(filepath)
    pred_y = []
    for embed in test_X:
        pred_y.append(np.argmax(model.predict(np.expand_dims(embed, axis=0))[0]))
    
    target_names = ['very negative','negative','neutral', 'positive', 'very positive']
    print(classification_report(test_y, pred_y, target_names=target_names))
    print(confusion_matrix(test_y, pred_y))
    print(accuracy_score(test_y, pred_y)) 
        
        
        
        
        
        
        
        
        
        
        