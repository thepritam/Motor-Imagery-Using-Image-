from keras.models import Sequential
from keras.layers import Dense,Dropout, concatenate
from keras.layers import LSTM,CuDNNLSTM
from keras.layers import TimeDistributed

import numpy as np

import boto3
import os
import pickle
from sklearn.model_selection import TimeSeriesSplit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
bucketname = 'motorimageryusingeeg'
s3 = boto3.resource('s3')
'''
fname1 = 'No_Preprocess_Dataset/X_Train'
fname2 = 'No_Preprocess_Dataset/Y_Train'
dname1 = 'X_Train'
dname2 = 'Y_Train'
s3.Bucket(bucketname).download_file(fname1, dname1)
s3.Bucket(bucketname).download_file(fname2, dname2)
'''
f1 = open('X_Train1312_0', 'rb')
example_dict = pickle.load(f1)
f1.close()

#print(example_dict.shape)
x = example_dict

f2 = open('Y_Train1312_0', 'rb')
example_dict1= pickle.load(f2)
f2.close()
#pickle_in = open(r"/home/krishna/eeg/eeg_data/X_Train","rb")
#example_dict = pickle.load(f1)
#pickle_out = open(r"/home/krishna/eeg/eeg_data/Y_Train","rb")
#example_dict1= pickle.load(f2)

#x=example_dict.reshape(18581,645,64)

y=example_dict1
y = np.argmax(y, axis = 1)

print(x.shape)
print(y.shape)

x = x[0:8000,:,:]
y = y[0:8000,]

print(y.shape)

#count = 18581
'''
maxi=0
for i in range(0,17000):
    if ((i%500) == 0):
        print(i)
    for j in range(0,645):
        for k  in range(0,64):
            maxi=max(maxi,x[i][j][k])
x=x/maxi
'''
'''
f = open('Normalized', 'wb')
pickle.dump(x, f)
f.close()
'''

tscv=TimeSeriesSplit(n_splits=4)
print(tscv)
for train_index,test_index in tscv.split(x):
    #print("Train:" , train_index,"Test: ", test_index)
    x_train,x_val=x[train_index],x[test_index]
    y_train,y_val= y[train_index],y[test_index]



def save_history(history,file):
    f = open(file, 'wb')
    pickle.dump(history, f)


from keras import layers
from keras.layers import Input, Add, Dense,Multiply, Reshape, SpatialDropout1D, Dropout, Concatenate, Activation, merge , ZeroPadding1D, BatchNormalization, Flatten, Conv1D, AveragePooling1D, GlobalAveragePooling2D, MaxPooling1D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.initializers import glorot_uniform
from keras.optimizers import *
import keras.backend as K
from skimage import io,color
from skimage.transform import resize
import scipy.misc
from matplotlib import pyplot as plt



def Model1D(classes = 4, input_shape = (1312, 64)):

        X_input = Input(input_shape)

	#CNN

        X = ZeroPadding1D(3)(X_input)
        X = BatchNormalization(axis = 1)(X)
        X = Conv1D(filters = 64, kernel_size = 5, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
        X = Activation('relu')(X)
        X = Conv1D(filters = 32, kernel_size = 5, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
        X = Activation('relu')(X)
        X = MaxPooling1D(2)(X)
        X = SpatialDropout1D(0.3)(X)
        print(X.shape)

        X = Conv1D(filters = 32, kernel_size = 3, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
        X = Activation('relu')(X)
        X = Conv1D(filters = 32, kernel_size = 3, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
        X = Activation('relu')(X)
        X = MaxPooling1D(2)(X)
        X = SpatialDropout1D(0.3)(X)
        print(X.shape)

        X = Conv1D(filters = 32, kernel_size = 3, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
        X = Activation('relu')(X)
        #X = Conv1D(filters = 32, kernel_size = 3, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
        #X = Activation('relu')(X)
        X = MaxPooling1D(2)(X)
        X = SpatialDropout1D(0.2)(X)
        print(X.shape)

        X = Conv1D(filters =16, kernel_size = 3, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
        X = Activation('relu')(X)
        #X = Conv1D(filters = 32, kernel_size = 3, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
        #X = Activation('relu')(X)
        #X = MaxPooling1D(3)(X)
        X = Dropout(0.1)(X)
        X = AveragePooling1D(3)(X)
        print(X.shape)
        cnn = Flatten()(X)
        print(cnn.shape)
        
        #LSTM

        X = TimeDistributed(Dense(512),input_shape=(1312, 64))(X_input)
        X = LSTM(64, return_sequences=True)(X)
        X = Dropout(0.2)(X)
        #X = LSTM(416, return_sequences=True)(X)
        #X = Dropout(0.3)(X)
        X = LSTM(32, return_sequences=False)(X)
        lstm = Dropout(0.2)(X)

        cnn_lstm_output = concatenate([cnn, lstm])
        predictions = Reshape((288,2))(cnn_lstm_output)

        #predictions = Dense(64, activation='relu')(cnn_lstm_output)
        #predictions=Dropout(0.2)(predictions)
        #predictions = Dense(128, activation='relu')(predictions)
        #predictions=Dropout(0.3)(predictions)
        #predictions = Dense(64, activation='relu')(predictions)
        #predictions=Dropout(0.2)(predictions)

        predictions = Dense(4, activation='softmax')(cnn)

	
        model = Model(inputs = X_input, outputs = predictions)

        return model





model = Model1D(classes = 4,input_shape = (1312, 64))
model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, batch_size=128, epochs=35, verbose=2, validation_data=(x_val, y_val))

model.save('cnn_model1312_0.h5')

#print(history.history['accuracy'])

save_history(history.history,'history_cnn_model1312_0')

