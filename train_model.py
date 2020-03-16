# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:13:24 2020

@author: femiogundare
"""


#import the required packages
import h5py
import argparse
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

ap = argparse.ArgumentParser()
ap.add_argument('-db', '--database', required=True, help='path HDF5 database')
ap.add_argument('-m', '--model', required=True, help='path to output model')
ap.add_argument('-j', '--jobs', type=int, default=-1, 
                help='no. of jobs when tuning parameters'
                )
args = vars(ap.parse_args())

#open the HDF5 database and determine the index of the training and test split
db = h5py.File(args['database'], 'r')
i = (db['labels'].shape[0] * 0.75)


print('Tuning the hyperparameters of the model...')
params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
grid = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args['jobs'])
grid.fit(db['features'][:i], db['labels'][:i])
print('Best hyperparameters: {}'.format(grid.best_params_))

predictions = grid.best_estimator_.predict(db['features'][i:])
print(
      classification_report(predictions, db['labels'][i:], target_names=db['label_names'])
      )


#compute the accuracy score
acc = accuracy_score(db['labels'][i:], predictions)
print('Accuracy score: {:.2f}'.format(acc))


print('Saving model...')
file = open(args['model'], 'wb')
file.write(pickle.dumps(grid.best_estimator_))
file.close()

db.close()