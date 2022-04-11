# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:57:13 2019

@author: HOME
"""
## The training data used in this code is extracted from AHA database
## and segmented to fixed length of 500 timesteps.
## The work can be reproduced just by downloading the .mat files prepared in Matlab
##Importing X and y from Matlab mat files
from numpy import array
from keras import layers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, LeakyReLU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import RMSprop, Adam
import keras
#import scipy.io
from sklearn.model_selection import train_test_split
import time

import hdf5storage

segment_py = hdf5storage.loadmat('seg_samples_STRIDED.mat')
segment = segment_py['seg_samples']
label_py = hdf5storage.loadmat('label_samples_STRIDED.mat')
label = label_py['label_samples']
label_arr = label#[:,1:5]
segment=segment.reshape((segment.shape[0], segment.shape[1],1))
label=label.reshape((label.shape[0], label.shape[1]))

#Randomely reshuffle data
from sklearn.utils import shuffle


X = segment
y = label_arr #Change to label_arr to avoid fiting to the normal
X, y = shuffle(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from scipy.io import savemat
savemat('X_train_STRIDED.mat', {'X_train': X_train})
savemat('X_test_STRIDED.mat', {'X_test': X_test})
savemat('y_train_STRIDED.mat', {'y_train': y_train})
savemat('y_test_STRIDED.mat', {'y_test': y_test})
# The train and test split is run from these files to compare results
## between different configurations
X_train = hdf5storage.loadmat('X_train_STRIDED.mat')
X_train = X_train['X_train']
X_test = hdf5storage.loadmat('X_test_STRIDED.mat')
X_test = X_test['X_test']
y_train = hdf5storage.loadmat('y_train_STRIDED.mat')
y_train = y_train['y_train']
y_test = hdf5storage.loadmat('y_test_STRIDED.mat')
y_test = y_test['y_test']
# define model

# Removing samples with N only label to minimize the imbalance of classes
b = np.array([1, 0, 0, 0, 0])
indexx_train = np.where(np.sum(y_train!=b,axis=1))
X_train_no_N = X_train[indexx_train[0]]
y_train_no_N = y_train[indexx_train[0]]

indexx_test = np.where(np.sum(y_test!=b,axis=1))
X_test_no_N = X_test[indexx_test[0]]
y_test_no_N = y_test[indexx_test[0]]
###################
# ArrhyNet
# from keras.layers import GlobalMaxPooling1D
# model1 = Sequential()
# model1.add(Conv1D(filters=16, kernel_size=30, strides = 1, padding='same', activation='relu', input_shape=(200,1)))
# model1.add(MaxPooling1D(pool_size=2))
# model1.add(Conv1D(filters=32, kernel_size=21, strides = 1, padding='same', activation='relu'))
# model1.add(MaxPooling1D(pool_size=2))
# model1.add(Conv1D(filters=64, kernel_size=15, strides = 1, padding='same', activation='relu'))
# model1.add(Conv1D(filters=128, kernel_size=7, strides = 1, padding='same', activation='relu'))
# model1.add(MaxPooling1D(pool_size=2))
# model1.add(Conv1D(filters=256, kernel_size=3, strides = 1, padding='same', activation='relu'))
# model1.add(Conv1D(filters=512, kernel_size=3, strides = 1, padding='same', activation='relu'))
# model1.add(GlobalMaxPooling1D(data_format="channels_last"))
# model1.add(Flatten())
# model1.add(Dense(256, activation='relu'))
# model1.add(Dense(64, activation='relu'))
# model1.add(Dense(5, activation='softmax'))
# model1.summary()


# Conventional 1D CNN
# model = Sequential()
# model.add(Conv1D(filters=16, kernel_size=3, use_bias=False, padding='same', activation='relu', input_shape=(500,1)))
# model.add(Conv1D(filters=16, kernel_size=3, use_bias=False, padding='same', activation='relu'))
# model.add(Conv1D(filters=16, kernel_size=3, use_bias=False, padding='same', activation='relu'))
# model.add(Conv1D(filters=16, kernel_size=3, use_bias=False, padding='same', activation='relu'))
# model.add(Conv1D(filters=16, kernel_size=3, use_bias=False, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=16, kernel_size=3, use_bias=False, padding='same', activation='relu'))
# model.add(Conv1D(filters=16, kernel_size=3, use_bias=False, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=16, kernel_size=3, use_bias=False, padding='same', activation='relu'))
# model.add(Conv1D(filters=16, kernel_size=3, use_bias=False, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=16, kernel_size=3, use_bias=False, padding='same', activation='relu'))
# model.add(Conv1D(filters=16, kernel_size=3, use_bias=False, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=16, kernel_size=3, use_bias=False, padding='same', activation='relu'))
# model.add(Conv1D(filters=16, kernel_size=3, use_bias=False, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(256, use_bias=False, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(Dense(32, use_bias=False, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(Dense(4, use_bias=False, activation='sigmoid'))
#model.compile(optimizer=Adam(lr=1e-6), loss='binary_crossentropy', metrics=['acc'])

# Insert shufflenet model and comment out the traditional CNN above
from shufflenet_1D import ShuffleNet1D_V2
model = ShuffleNet1D_V2(classes=5,num_shuffle_units=[1]) #,include_top=False
model.compile(optimizer=Adam(lr=1e-2), loss='binary_crossentropy', metrics=['acc','mae'])
model.summary()


# from focal_loss import focal_loss
# model.compile(optimizer=Adam(lr=1e-2),  loss=[focal_loss(alpha=.3, gamma=3)], metrics=['acc','mae'])
# model.compile(optimizer=Adam(lr=1e-2), loss='binary_crossentropy', metrics=['acc','mae'])

##Varying learning rate slowly during training
def lr_scheduler(epoch, lr):
    if epoch > 55:
        lr = 1e-6
    elif epoch > 60:
        lr = 1e-2
        return lr
    return lr
callbacks = [keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)]



#from keras_flops import get_flops
#flops = get_flops(model, batch_size=1)
#print(f"FLOPS: {flops / 10 ** 9:.03} G")


t_training = time.time()
# fit model


history = model.fit(X_train_no_N, y_train_no_N, 
                    epochs=40, 
                    batch_size=1024 , 
                    verbose=1,
                    validation_split=0.2, 
                    callbacks=callbacks)
elapsed_training = time.time() - t_training
print(elapsed_training)

test_acc = model.evaluate(X_test_no_N, y_test_no_N)


#y_predict=model.predict(X_test)
t = time.time()
y_predict=model.predict(X_test_no_N)
elapsed = time.time() - t
print(elapsed)


#from keras.utils import plot_model
#plot_model(model, to_file='ShuffleNet1D_V2_new.svg', show_layer_names=True, show_shapes=True)
epoch=range(1, 41)
#Plotting the acc and val of the training
val = history.history['val_acc']
acc = history.history['acc']

import matplotlib.pyplot as plt
plt.plot(epoch, val, 'bo', label='validation')
# "bo" is for "blue dot"
plt.plot(epoch, acc, label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('acc')
plt.legend()
plt.show()
#Confution Matrix and Classification Report
from sklearn.metrics import confusion_matrix
y_true = y_test_no_N
y_predict[y_predict < 0.5] = 0
y_predict[y_predict >= 0.5] = 1

#y_pred = y_predict
Cc = confusion_matrix(y_true.argmax(axis=1), y_predict.argmax(axis=1))
Conf_normallized = Cc / Cc.astype(np.float).sum(axis=1)
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.metrics import f1_score
print(precision_score(y_true,y_predict,average=None))
print(recall_score(y_true,y_predict,average=None))
print(f1_score(y_true,y_predict,average='weighted'))
print(classification_report(y_true,y_predict,target_names=None))

#model.save('ECG_1D_CNN.h5')

## Visualizing intermediate activations
from keras import models

# Extracts the outputs of the top 8 layers:
start_layer = 26
end_layer = 28
layer_outputs = [layer.output for layer in model.layers[start_layer:end_layer]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

input_tensor=X_test_no_N[11]
input_tensor=input_tensor.reshape(1,500, 1)
#samples=range(1,501)
#import matplotlib.pyplot as plt
#plt.plot(samples, input_tensor[0,:, 0], label='imput_ECG')

activations = activation_model.predict(input_tensor)

first_layer_activation = activations[1]
print(first_layer_activation.shape)
import matplotlib.pyplot as plt

#samples_act=range(1,first_layer_activation.shape[2]+1)
#plt.plot(samples_act, first_layer_activation[0,0,:, 6]) #10th activation filter


samples_act=range(1,input_tensor.shape[1]+1)
plt.title(y_test_no_N[11])
plt.plot(samples_act, input_tensor[0, :, 0]) #10th activation filter
plt.show()
Col_size = 4
row_size = 4#first_layer_activation.shape[2] // Col_size
fig, axs = plt.subplots(row_size,Col_size)
for rr in range(0,row_size):
    for cc in range(0,Col_size):
        samples_act=range(1,first_layer_activation.shape[2]+1)
        axs[rr,cc].plot(samples_act, first_layer_activation[0, 0, :, rr+cc]) #10th activation filter
    

  
model.save('shufflenet_70epoc_32hNew.h5')
