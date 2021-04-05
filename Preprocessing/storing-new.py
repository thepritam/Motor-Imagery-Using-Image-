import boto3
import os
import pickle
import numpy as np
from keras.utils import np_utils
from sklearn.utils import shuffle
import progressbar

bucketname = 'motorimageryusingeeg'

name = ["Rest", "LeftFist", "RightFist", "BothFists", "BothFeet"]


def uploadFile(file, fname):
	#fname = "X_Train"
	# np.save(name, sample)
	f = open(fname, 'wb')
	pickle.dump(file, f, protocol = 4)
	f.close()
	statinfo = os.stat(fname)
	upload_file_name = 'No_Preprocess_Dataset1/' + fname
	s3 = boto3.resource('s3')
	up_progress = progressbar.progressbar.ProgressBar(maxval=statinfo.st_size)
	up_progress.start()
	def upload_progress(chunk):
		up_progress.update(up_progress.currval + chunk)

	s3.Bucket(bucketname).upload_file(fname, upload_file_name, Callback=upload_progress)
	up_progress.finish()	
	os.remove(fname)


feature_x = []
feature_y = []
Y_train = []
for n in range(1,110):
	for j in range(1,5):
		fname = 'EEG_DATA/Mod_Dataset/S'+ str(n).zfill(3) + '/' +  name[j] + '.pickle'
		
		#fname = 'EEG_DATA/No_Preprocess_Final/' + name[j-1] + '.pickle'
		dname = 'sample.pickle'
		print(fname)
		s3 = boto3.resource('s3')
		s3.Bucket(bucketname).download_file(fname, dname)
		f = open(dname, 'rb')
		data = pickle.load(f)
		f.close()

		v = len(data)
		#print(str(data[0].shape) + ' ' +str(v))
		

		for t in range(1, v-1):
			X_train = np.asarray(data[t][:, 0:645])
			X_train = np.transpose(X_train)
			X_train = X_train.reshape((1,X_train.shape[0], X_train.shape[1],1))
			#Y_train.append(j-1)
			#Y_train = np_utils.to_categorical(Y_train, len(name))
			#Y_train = Y_train.reshape((1,1,len(name)))
			if(X_train.shape == (1, 645, 64, 1)):
				feature_x.append(X_train[0])
				print(X_train[0].shape)
				Y_train.append(j-1)
			#feature_y.append(Y_train[0])	
		os.remove(dname)
stacked_array = np.stack(feature_x, axis = 0)
stacked_labels = np.stack(Y_train, axis = 0)
#stacked_labels = np_utils.to_categorical(stacked_labels, 4)

print("Compressing Data...")
stacked_array, stacked_labels = shuffle(stacked_array, stacked_labels)
print("X_Train : " + str(stacked_array.shape) + " Y_Train : " + str(stacked_labels.shape))
print("Data Compression Over, preparing file for Upload")
uploadFile(stacked_array, "X_Train")
uploadFile(stacked_labels, "Y_Train")
