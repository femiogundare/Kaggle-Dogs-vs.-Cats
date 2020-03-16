# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:12:58 2020

@author: femiogundare
"""


#Import the required packages
import numpy as np
import os
import cv2
import json
import progressbar
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utilities.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from utilities.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from utilities.io.hdf5datasetwriter import HDF5DatasetWriter
from config import dogs_vs_cats_config as config


#grab the paths to the images
trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[1].split('.')[0] for p in trainPaths]

#convert the labels to integers
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

#split the dataset into training and testing sets
trainPaths, testPaths, trainLabels, testLabels = train_test_split(
        trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES, stratify=trainLabels, 
        random_state=42
        )

#split the training set in order to allow for a validation set
trainPaths, valPaths, trainLabels, valLabels = train_test_split(
        trainPaths, trainLabels, test_size=config.NUM_VAL_IMAGES, stratify=trainLabels, 
        random_state=42
        )

# construct a list pairing the training, validation, and testing image paths 
#along with their corresponding labels and output HDF5 files
datasets = [
        ('train', trainPaths, trainLabels, config.TRAIN_HDF5),
        ('val', valPaths, valLabels, config.VAL_HDF5),
        ('test', testPaths, testLabels, config.TEST_HDF5)
        ]

#initialize the image preprocessors and the list of RGB channel averages
aap = AspectAwarePreprocessor(width=256, height=256)
itap  = ImageToArrayPreprocessor(data_format=None)

R, G, B = [], [], []

#loop over the datasets tuples
for dType, paths, labels, outputPath in datasets:
    #create HDF5 writer
    print('Building {}...'.format(outputPath))
    writer = HDF5DatasetWriter(dims=(len(paths), 256, 256, 3), outputPath=outputPath)
    
    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),
                                   widgets=widgets).start()
    
    #loop over the image paths
    for i, (path, label) in enumerate(zip(paths, labels)):
        #load the image and preprocess it
        image = cv2.imread(path)
        image = aap.preprocess(image)
        image = itap.preprocess(image)
        
        #compute the mean of each channel in each image in the training set and update the lists
        if dType=='train':
            g, b, r = cv2.mean(image)[:3]
            B.append(b)
            G.append(g)
            R.append(r)
        #add the image and label to the hdf5 writer
        writer.add([image], [label])
        pbar.update(i)
    
    pbar.finish()
    writer.close()
    
    
#calculate the average RGB values over all images in the dataset, and then serialize
print('Serializing...')
avgs = {'R' : np.mean(R), 'G' : np.mean(G), 'B' : np.mean(B)}
file = config.DATASET_MEAN
open_file = open(file, 'w')
open_file.write(json.dumps(avgs))
open_file.close()