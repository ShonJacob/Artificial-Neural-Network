#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 05:35:02 2018

@author: shonjacob
"""
#Data PreProcessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('diabetic_data.csv')
#print(dataset.describe())

#Handling missing values
#from sklearn.preprocessing import Imputer
dataset = dataset.replace('Unknown/Invalid', np.NaN)
dataset.dropna(inplace=True)
X = dataset.iloc[:, [3,4,9,12,13,14,16,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]].values#.values
#imputer = Imputer(missing_values = 'Unknown/Invalid', strategy = 'most_frequent', axis = 0)
#imputer = imputer.fit(X[:, 0])
#X[:, 0] = imputer.transform(X[:, 0])
df_X = pd.DataFrame(X)
y = dataset.iloc[:, [49]].values
df_Y = pd.DataFrame(y)

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
#gender,age,maxglu,a1C
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0]) 
X[:, 1] = labelEncoder_X.fit_transform(X[:, 1])
X[:, 7] = labelEncoder_X.fit_transform(X[:, 7]) 
X[:, 8] = labelEncoder_X.fit_transform(X[:, 8]) 
#medicines
X[:, 9] = labelEncoder_X.fit_transform(X[:, 9]) 
X[:, 10] = labelEncoder_X.fit_transform(X[:, 10]) 
X[:, 11] = labelEncoder_X.fit_transform(X[:, 11]) 
X[:, 12] = labelEncoder_X.fit_transform(X[:, 12]) 
X[:, 13] = labelEncoder_X.fit_transform(X[:, 13]) 
X[:, 14] = labelEncoder_X.fit_transform(X[:, 14]) 
X[:, 15] = labelEncoder_X.fit_transform(X[:, 15]) 
X[:, 16] = labelEncoder_X.fit_transform(X[:, 16]) 
X[:, 17] = labelEncoder_X.fit_transform(X[:, 17]) 
X[:, 18] = labelEncoder_X.fit_transform(X[:, 18]) 
X[:, 19] = labelEncoder_X.fit_transform(X[:, 19]) 
X[:, 20] = labelEncoder_X.fit_transform(X[:, 20]) 
X[:, 21] = labelEncoder_X.fit_transform(X[:, 21]) 
X[:, 22] = labelEncoder_X.fit_transform(X[:, 22]) 
X[:, 23] = labelEncoder_X.fit_transform(X[:, 23]) 
X[:, 24] = labelEncoder_X.fit_transform(X[:, 24]) 
X[:, 25] = labelEncoder_X.fit_transform(X[:, 25]) 
X[:, 26] = labelEncoder_X.fit_transform(X[:, 26]) 
X[:, 27] = labelEncoder_X.fit_transform(X[:, 27]) 
X[:, 28] = labelEncoder_X.fit_transform(X[:, 28]) 
X[:, 29] = labelEncoder_X.fit_transform(X[:, 29]) 
X[:, 30] = labelEncoder_X.fit_transform(X[:, 30]) 
X[:, 31] = labelEncoder_X.fit_transform(X[:, 31])
#changemed
X[:, 32] = labelEncoder_X.fit_transform(X[:, 32]) 
X[:, 33] = labelEncoder_X.fit_transform(X[:, 33]) 
#y transformation
y[:, 0] = labelEncoder_X.fit_transform(y[:, 0]) 

onehotencoder = OneHotEncoder(categorical_features = [1,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
onehotencoderY = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
y = onehotencoderY.fit_transform(y).toarray()

#adding a baseline variable to remove dummy variable trap
#X = X[:, 1:]#chooses all rows and all columns from 1-end

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
#The idea behind StandardScaler is that it will transform your data such that its distribution
# will have a mean value 0 and standard deviation of 1. Given the distribution of the data, 
#each value in the dataset will have the sample mean value subtracted, 
#and then divided by the standard deviation of the whole dataset.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#feature selection algorithm


#PART 2 - CREATING THE ANN
import keras
from keras.models import  Sequential
from keras.layers import Dense

classifier = Sequential()
#rectifier activation function for hidden layer and sigmoid activation fn. for output layer
#try with softmax to see output for output layer
#later on we need to do crossvalidation to determine the number of nodes in hidden layer through experimentation

#adding the input layer and first hidden layer
classifier.add(Dense(output_dim = 50,init = 'uniform',activation = 'relu', input_dim = 97))

#adding the second hidden layer
classifier.add(Dense(output_dim = 50,init = 'uniform',activation = 'relu', input_dim = 50))

#adding the output layer, softmax activation fn. for more than 2 categories output
classifier.add(Dense(output_dim = 3,init = 'uniform',activation = 'sigmoid', input_dim = 50))

#compiling the ANN, optimizer is stochastic gradient descend(garrett mclaws function) logarithmic error ,binary_crossentropy for binary output,categorical_crossentropy
#metrics is how to improve performance
#adaptive moment estimation
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the Training set
#batchsize is number of observations after which we update weights
#find an optimal method to find batchsize and number of epochs
classifier.fit(X_train, y_train, batch_size = 200, nb_epoch = 50)
#Apply Evolutionary Algorithm

# Predicting the Test set results
#choosing a threshold value 
y_pred = classifier.predict(X_test)
y_pred=(y_pred > 0.8)

## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#from mlxtend.evaluate import confusion_matrix
#cm = confusion_matrix(y_target=y_test, 
 #                     y_predicted=y_pred, 
  #                    binary=False)
#cm
#from mlxtend.plotting import plot_confusion_matrix
#fig, ax = plot_confusion_matrix(conf_mat=cm)
#plt.show()

#Accuracy
#(1515+199)/2000

# the following hyperparameters could be optimized

#batch size and training epochs
#optimization algorithm
#learning rate and momentum
#network weight initialization
#activation function in hidden layers
#dropout regularization
#the number of neurons in the hidden layer.
