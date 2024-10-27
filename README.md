# Advanced Machine Learning Project 2: Heart rhythm classification from raw ECG signals

## Introduction
This project was conducted for the course "Advanced Machine Learning" that I followed in Fall 2022 at ETHZ: https://ml2.inf.ethz.ch/courses/aml2022/


## Goal
The goal was to classify heart rhythm 

## Solution
1. Feature Extraction
   - The biospy library was used to extract the raw features like rpeaks, heart rate, heartbeats and the cleaned signal from the ECG
   - Extracted the P, Q, R, S, and T points from the ECG signal, as well as the calcuéated intervals (PR, QRS, ST)
   - Computed the entropy of both the raw and filtered ECG signal
   - Calculated wavelet energy to capture frequency-domain features
   - Measured the difference between consecutive R-peak indices to compute the variability of heartbeat intervals
   - Measure the minimum, maximum, mean, median and standard deviation of heart rate across the signal
    
2. Preprocessing
     - Applied RobustScaler and PowerTransformer on training and test sets to handle outliers and transform features to a standardized scale
     - Used IsolationForest to identify and remove outliers
       
4. Classification
   - Used XGBoost Classifier
   - Feature selection based on feature importances from XGBoost, and retrain the model with a subset of the selected features
   - Cross-validation to validate the feature selection
   - Hyperparameter tuning with hyperopt to perform Bayesian optimization on XGBoost
   - Final model prediction after fine-tuning and model retraining.
