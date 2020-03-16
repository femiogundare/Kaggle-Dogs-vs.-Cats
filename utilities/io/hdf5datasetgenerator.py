# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:48:10 2020

@author: femiogundare
"""

#import the required packages
import h5py
import numpy as np
from keras.utils import np_utils

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, classes=2):
        
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.binarize = binarize
        self.aug = aug
        self.classes = classes
        
        #open the HDF5 database and determine the total no of entries in the database
        self.db = h5py.File(dbPath)
        self.numImages = self.db['labels'].shape[0]
    
    def generator(self, passes = np.inf):
        epochs = 0
        
        #keep looping infinitely....this will stop when the model reaches the desired epoch
        while epochs < passes:
            #loop over the HDF5 dataset
            for i in np.arange(0, self.numImages, self.batchSize):
                images = self.db['images'][i: i + self.batchSize]
                labels = self.db['labels'][i: i + self.batchSize]
                
                #check whether or not the labels should be binarized
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)
                    
                #check to see if the preprocessors are not None
                if self.preprocessors is not None:
                    #initialize the list of processed images
                    procImages = []
                    
                    #loop over the images
                    for image in images:
                        #loop over the preprocessors and apply them to the images
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                            
                        procImages.append(image)
                        
                    #create an array of the preprocessed images
                    images = np.array(procImages)
                    
                
                #check to see if data augmentation for image generation/augmentation is set
                #here, self.aug is just acting in place of a keras image generator
                #if self.aug is instantiated to be a keras generator, then augmented images
                #are generated based on batches
                
                if self.aug is not None:
                    images, labels = next(
                            self.aug.flow(images, labels, batch_size=self.batchSize)
                            )
                #yield a tuple of images and labels
                yield(images, labels)
            
            epochs +=1
        
    def close(self):
        self.db.close()