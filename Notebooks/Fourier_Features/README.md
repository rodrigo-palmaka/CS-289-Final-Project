# Transformed Fourier Feature Prediction of Right Ascension and Declination

This folder contains all the code relevant to the sections: 
-Transformed Fourier Features
-Transformed Fourier Feature Regression
-Transformed Fourier Feature with Neural Networks

### RA/DECL_REG.ipynb
Running through this file will demonstrate the following:
1. Cyclical nature of right ascension and declination
2. Fitting linear regression on right ascension and declination training data with fourier features
3. Predicting test data using optimal models found via hyperparameter tuning on a static validation set (80-20; train-val split)

### RA/DECL_NN.py
Running through this file will demonstrate the following:
1. Conversion of regression model into neural network based model with 2 layers of 5 neurons each with relu activation.
2. Fitting neural network on right ascension and declination training data with fourier features
3. Predicting test data using optimal models found via hyperparameter tuning on a static validation set (80-20; train-val split)
