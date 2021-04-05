import numpy as np
import pickle
pickle_in = open(r"/home/krishna/eeg/eeg_data/X_Train_Min","rb")
example_dict = pickle.load(pickle_in)
pickle_out = open(r"/home/krishna/eeg/eeg_data/Y_Train_Min","rb")
example_dict1= pickle.load(pickle_out)
x_train=example_dict.reshape(840,1320,64)
y_train=example_dict1
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM 
from keras.layers import TimeDistributed
n_feature=64
time_stamp=1320
samples=840
n_epoch=50
model = Sequential()
model.add(TimeDistributed(Dense(512), input_shape=(1320, 64)))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs=10, batch_size=40,verbose=1)
