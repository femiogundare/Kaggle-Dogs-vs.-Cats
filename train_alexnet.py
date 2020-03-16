# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:27:12 2020

@author: femiogundare
"""

#import the required packages
import matplotlib
matplotlib.use("Agg")


import os
import json
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from config import dogs_vs_cats_config as config
from utilities.callbacks.trainingmonitor import TrainingMonitor
from utilities.io.hdf5datasetgenerator import HDF5DatasetGenerator
from utilities.preprocessing.simplepreprocessor import SimplePreprocessor
from utilities.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from utilities.preprocessing.meanpreprocessor import MeanPreprocessor
from utilities.preprocessing.patchpreprocessor import PatchPreprocessor
from utilities.nn.cnn.alexnet import AlexNet

#construct the training image generator for data augmentation
aug = ImageDataGenerator(
        rotation_range=20, zoom_range=0.15, width_shift_range=0.2, 
        height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, 
        fill_mode="nearest"
        )

#load the RGB means for the training set
means = open(config.DATASET_MEAN)
means = means.read()
means = json.loads(means)

#initialize the preprocessors
sp = SimplePreprocessor(width=227, height=227)
pp = PatchPreprocessor(height=227, width=227)
mp = MeanPreprocessor(rMean=means['R'], gMean=means['G'], bMean=means['B'])
itap = ImageToArrayPreprocessor()

#initialize the training and validation generators
trainGen = HDF5DatasetGenerator(
        dbPath=config.TRAIN_HDF5, batchSize=128, preprocessors=[pp, mp, itap], 
        aug=aug, classes=2
        )
valGen = HDF5DatasetGenerator(
        dbPath=config.VAL_HDF5, batchSize=128, preprocessors=[sp, mp, itap], 
        classes=2
        )

#compile the model
print('Compiling model...')
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(
        optimizer=optimizer, 
        loss='binary_crossentropy', 
        metrics=['accuracy']
        )

#construct the set of callbacks
print('[INFO]: Process ID: {}'.format(os.getpid()))
fig_path = os.path.sep.join([config.OUTPUT_PATH, '{}.png'.format(os.getpid())])
json_path = os.path.sep.join([config.OUTPUT_PATH, '{}.json'.format(os.getpid())])
callbacks = [TrainingMonitor(fig_path, json_path)]

#train the model
H = model.fit_generator(
        trainGen.generator(), validation_data=valGen.generator(),
        steps_per_epoch=trainGen.numImages // 128,
        validation_steps=valGen.numImages // 128,
        epochs=75, max_queue_size=128*2, callbacks=callbacks, verbose=1
        )

print('Serializing model...')
model.save(config.MODEL_PATH, overwrite=True)

#close the HDF5 datasets
trainGen.close()
valGen.close()


#plots
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 75), H.history["loss"], label = "Training loss")
plt.plot(np.arange(0, 75), H.history["val_loss"], label = "Validation loss")
plt.plot(np.arange(0, 75), H.history["accuracy"], label = "Training accuracy")
plt.plot(np.arange(0, 75), H.history["val_accuracy"], label = "Validation accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()