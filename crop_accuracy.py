# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:07:12 2020

@author: femiogundare
"""

#import the required libraries
import os
import json
import numpy as np
import progressbar
from config import dogs_vs_cats_config as config
from utilities.io.hdf5datasetgenerator import HDF5DatasetGenerator
from utilities.preprocessing.simplepreprocessor import SimplePreprocessor
from utilities.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from utilities.preprocessing.meanpreprocessor import MeanPreprocessor
from utilities.preprocessing.croppreprocessor import CropPreprocessor
from utilities.utils.ranked import rank5_accuracy
from keras.models import load_model


#load the RGB means from the training set
means = open(config.DATASET_MEAN)
means = means.read()
means = json.loads(means)

#initialize the preprocessors
sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(rMean=means['R'], gMean=means['G'], bMean=means['B'])
cp = CropPreprocessor(227, 227)
itap = ImageToArrayPreprocessor()


print('Loading the model...')
model = load_model(config.MODEL_PATH)



#----------------------------BASELINE MODEL-----------------------------
#here, I will use sp, mp and itap as preprocessors, and then calculate the score
print('Predicting on test data with no crops...')
testGen = HDF5DatasetGenerator(
        dbPath=config.TEST_HDF5, batchSize=64, preprocessors=[sp, mp, itap], 
        classes=2)
predictions = model.predict_generator(
        testGen.generator(), steps=testGen.numImages // 64, max_que_size=64*2
        )
rank1, _ = rank5_accuracy(predictions, testGen.db['labels'])
print('Rank 1 accuracy: {:.2f}%'.format(rank_1*100))
testGen.close()



#-----------------------------------OVERSAMPLING---------------------------------
#here, I will be using the CropPreprocessor
testGen = HDF5DatasetGenerator(
        dbPath=config.TEST_HDF5, batchSize=64, preprocessors=[mp], 
        classes=2)
predictions = []

#initialize the progress bar
widgets = ["Evaluating: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGen.numImages // 64,
                                   widgets=widgets).start()

#loop over a single pass of the test data i.e set pass=1 cos I'm to evaluate
#this loop runs only once
for i, (images, labels) in enumerate(testGen.generator(passes=1)):
    #since pass=1, the generator produces only a single batch of the images in the
    #test HDF5 file...from this, I can for sure say that this single pass produces
    #64 images which I am going to loop over each one of them and perform cropping!!!
    
    #now loop over the 64 images
    for image in images:
        #apply the CropPreprocessor; it produces 10 different crops;
        #then apply the ImageTo ArrayPreprocessor
        crops = cp.preprocess(image)
        crops = [itap.preprocess(crop) for crop in crops]
        crops = np.array(crops, dtype='float32')
        
        #make predictions on the crops and then average them in order to get the final
        #prediction for the image
        pred = model.predict(crops)
        predictions.append(np.mean(pred, axis=0))
        
    #update the progress bar
    pbar.update(i)
    
pbar.finish()

print('Predicting on test data with crops...')
rank1, _ = rank5_accuracy(predictions, testGen.db['labels'])
print('Rank 1 accuracy: {:.2f}%'.format(rank_1*100))
testGen.close()