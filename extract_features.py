# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:27:34 2020

@author: femiogundare
"""

#import the required packages
import os
import numpy as np
import random
import progressbar
import argparse
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from utilities.io.hdf5datasetwriter import HDF5DatasetWriter
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import ResNet50, imagenet_utils

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to the dataset')
ap.add_argument('-o', '--output', required=True, help='path to output HDF5 file')
ap.add_argument(
        '-b', '--batch-size', default=16, type=int, 
        help='batch size of the images to be passed into the network'
        )
ap.add_argument('-s', '--buffer-size', default=1000, 
                type=int, help='size of feature extraction buffer'
                )
args = vars(ap.parse_args())


print('Loading images...')
imagePaths = list(paths.list_images(args[['dataset']]))
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[1].split('.')[0] for p in imagePaths]

#convert the labels to integers
le = LabelEncoder()
labels = le.fit_transform(labels)


print('Loading network...')
model = ResNet50(weights='image_net', include_top=False)

#initialize the HDF5 writer and store the class label names in the dataset
dataset = HDF5DatasetWriter(
        dims=(len(imagePaths), 2048), outputPath=args['output'], 
        dataKey='features', buffSize=args['buffer-size']
        )
dataset.storeClassLabels(le.classes_)

#initialize the progress bar
widgets = [
        "Extracting Features: ", progressbar.Percentage(), " ",progressbar.Bar(), " ", 
        progressbar.ETA()
        ]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),widgets=widgets).start()


#loop over the images in batches, preprocess them and store in HDF5
for i in np.arange(0, len(imagePaths), args['batch-size']):
    batchPaths = imagePaths[i:i + args['batch-size']]
    batchLabels = labels[i:i + args['batch-size']]
    batchImages = []
    
    for j, imagePath in enumerate(batchPaths):
        #load the image and convert it to 224*224
        image = load_img(imagePath, target_size=(224, 224))
        #convert the image to array
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        
        batchImages.append(image)
        
    #stack the batch images vertically so each assumes a shape of (N, 224, 224, 3)
    batchImages = np.vstack(batchImages)
    #make the model perform prediction
    features = model.predict(batchImages, batch_size=args['batch-size']) #shape becomes (N, 2048)
    #flatten the image vectors
    features = features.reshape((features.shape[0], 2048))
        
    #add the features and labels to the hdf5 dataset
    dataset.add(features, batchLabels)
    pbar.update(i)
        
dataset.close()
pbar.finish()