import numpy as np

import boto3
import os
import pickle
import numpy as np

bucketname = 'motorimageryusingeeg'
s3 = boto3.resource('s3')

fname1 = 'mod_data_set/X_Train'
fname2 = 'mod_data_set/Y_Train'
dname1 = 'X_Train'
dname2 = 'Y_Train'
s3.Bucket(bucketname).download_file(fname1, dname1)
s3.Bucket(bucketname).download_file(fname2, dname2)

f1 = open('X_Train', 'rb')
X_train = pickle.load(f1)
f1.close()

f2 = open('Y_Train', 'rb')
Y_train = pickle.load(f2)
f2.close()

print(X_train.shape)
print(Y_train.shape)
print(Y_train[78])

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
Y_train = np.argmax(Y_train, axis = 1)
print(X_train.shape)
print(Y_train.shape)
print(Y_train[78])

XTrain = X_train[0:12600, :, :]
YTrain = Y_train[0:12600]
XValidation = X_train[12601:18000, :, :]
YValidation = Y_train[12601:18000]
XTest = X_train[18001: , :, :]
YTest = Y_train[18001:]
print(XTrain.shape)
print(XValidation.shape)
print(XTest.shape)
 


os.remove(dname1)
os.remove(dname2)


from keras import layers

from keras.layers import Input, Add, Dense, SpatialDropout1D, Dropout, Concatenate, Activation,merge , ZeroPadding1D, BatchNormalization, Flatten, Conv1D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling1D, GlobalMaxPooling2D

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


def Model1D(classes = 4, input_shape = (645, 64)):
	X_input = Input(input_shape)
	X = ZeroPadding1D(3)(X_input)
	X = BatchNormalization(axis = 1)(X)
	X = Conv1D(filters = 64, kernel_size = 5, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
	X = Activation('relu')(X)
	X = Conv1D(filters = 64, kernel_size = 5, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
	X = Activation('relu')(X)    
	X = MaxPooling1D(2)(X)
	X = SpatialDropout1D(0.1)(X)
	print(X.shape)
    
	X = Conv1D(filters = 32, kernel_size = 3, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
	X = Activation('relu')(X)
	X = Conv1D(filters = 32, kernel_size = 3, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X) 
	X = Activation('relu')(X)
	X = MaxPooling1D(2)(X)
	X = SpatialDropout1D(0.1)(X)
	print(X.shape)
    
	X = Conv1D(filters = 32, kernel_size = 3, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
	X = Activation('relu')(X)
	X = Conv1D(filters = 32, kernel_size = 3, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
	X = Activation('relu')(X)
	X = MaxPooling1D(2)(X)
	X = SpatialDropout1D(0.1)(X)
	print(X.shape)
	
	X = Conv1D(filters = 32, kernel_size = 3, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
	X = Activation('relu')(X)
	X = Conv1D(filters = 32, kernel_size = 3, padding = 'same', kernel_initializer = glorot_uniform(seed=0), data_format = 'channels_last')(X)
	X = Activation('relu')(X)
	X = MaxPooling1D(2)(X)
	X = Dropout(0.1)(X)
	print(X.shape)
	X = Flatten()(X)
	print(X.shape)
	X = Dense(4, activation = 'softmax')(X)
	print(X.shape)
	model = Model(inputs = X_input, outputs = X)
    
	return model





model = Model1D(classes = 4,input_shape = (1320, 64))
model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(XTrain, YTrain, batch_size = 50, verbose = 2, epochs = 30, validation_data = (XValidation, YValidation))

'''
preds = model.evaluate(X_train, Y_train, batch_size = 50, verbose = 2)


print ("Validation Loss = " + str(preds[0]))

print ("Validation Accuracy = " + str(preds[1]))
'''

preds = model.evaluate(XTrain, YTrain, batch_size = 50, verbose = 2)

print ("Train Loss = " + str(preds[0]))

print ("Train Accuracy = " + str(preds[1]))

preds = model.evaluate(XValidation, YValidation, batch_size = 50, verbose = 2)

#preds = model.evaluate_generator(validation_generator, steps=175)

print ("Validation Loss = " + str(preds[0]))

print ("Validation Accuracy = " + str(preds[1]))

#preds = model.evaluate_generator(test_generator, steps=175)

preds = model.evaluate(XTest, YTest, batch_size = 50, verbose = 2)

print ("Test Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

print("DONE!")
