# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 07:24:06 2020

@author: femiogundare
"""

#import the required libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.regularizers import l2
from keras import backend as K


class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        #for channels_last
        inputShape = (height, width, depth)
        chanDim = -1
        
        #if the backend is channels_first
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanDim = 1
            
        #building the architecture
        model = Sequential()
        model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), input_shape=inputShape, padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation(activation='relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(rate=0.25))
        
        model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))
        
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))
        
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation('softmax'))
        
        return model