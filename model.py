#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:03:14 2017

@author: KarimM
"""

"""
Start of model training
"""
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D ,AveragePooling2D
from Preprocess import convert_lfw,cropImage
import matplotlib.pyplot as plt

batch_size = 128
nb_classes = 2
nb_epoch = 6

train_features,train_targets,test_features,test_targets = convert_lfw('lfw_Preprocessed')

# convert class vectors to binary class matrices
#train_targets = np_utils.to_categorical(train_targets, nb_classes)
#test_targets = np_utils.to_categorical(test_targets, nb_classes)

#train_features = np.asarray(map(cropImage, train_features))
#test_features = np.asarray(map(cropImage, test_features))

train_features = np.transpose(train_features,[0,2,3,1])
test_features = np.transpose(test_features,[0,2,3,1])

train_features = train_features/255.0
test_features = test_features/255.0

modelA = Sequential()
modelA.add(Conv2D(32, (5,5),
                  input_shape=train_features.shape[1:],
                  padding='same',
                  data_format="channels_last",
                  activation='relu'))

modelA.add(Conv2D(32, (5,5),
                  padding='same',
                  data_format="channels_last",
                  activation='relu'))

modelA.add(AveragePooling2D(pool_size=(2,2),data_format="channels_last"))
modelA.add(Dropout(0.2))

modelA.add(Conv2D(32, (5,5),
                  padding='same',
                  data_format="channels_last",
                  activation='relu'))

modelA.add(AveragePooling2D(pool_size=(2,2),data_format="channels_last"))
modelA.add(Dropout(0.2))

modelA.add(Flatten())
modelA.add(Dense(512, activation='relu'))
modelA.add(Dense(128, activation='relu'))
modelA.add(Dropout(0.5))
modelA.add(Dense(1, activation='sigmoid'))

print(modelA.summary())

opt = keras.optimizers.Adam(lr=0.00001)
modelA.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
history = modelA.fit(train_features, train_targets, 
           batch_size=batch_size, 
           epochs=nb_epoch, 
           verbose=1,
           validation_data=(test_features, test_targets),shuffle=True)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
