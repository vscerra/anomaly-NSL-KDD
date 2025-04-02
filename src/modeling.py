# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 12:34:14 2025

@author: vscerra
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd

# Baseline model: Logistic Regression

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model for binary classification
    """
    model = LogisticRegression(max_iter = 1000)
    model.fit(X_train, y_train)
    return model
  

def evaluate_binary_model(model, X_test, y_test):
    """
    Evaluate the binary classifier using standard metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
        }
    
    return metrics
  

# Unsupervised Learning: Isolation Forest

def train_isolation_forest(X_train, contamination = 0.1, random_state = 42):
    """
    Train Isolation Forest model for anomaly detection
    """
    model = IsolationForest(contamination = contamination, random_state = random_state)
    model.fit(X_train)
    return model
  

def evaluate_unsupervised_model(model, X_test, y_test):
    """ 
    Evaluate an unsupervised model by mapping its predictions to binary labels
    """
    y_pred_raw = model.predict(X_test)
    
    # convert from [-1, 1] to [1, 0] (1 = anomaly, 0 = normal)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
        }
    
    return metrics
  

# Deep learning: Autoencoder architecture in Keras

def build_autoencoder(input_dim, encoding_dim=16):
    """
    Define an autoencoder architecture with customizable bottleneck
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
    from tensorflow.keras.optimizers import Adam
    
    input_layer = Input(shape = (input_dim,))
    #encoder
    x = Dense(64, activation = 'relu')(input_layer)
    x = BatchNormalization()(x)
    #x = Dropout(0.1)(x)
    x = Dense(encoding_dim, activation = 'relu')(x)
    
    #decoder
    x = Dense(64, activation = 'relu')(x)
    x = BatchNormalization()(x)
    output_layer = Dense(input_dim, activation = 'linear')(x)
    
    autoencoder = Model(inputs = input_layer, outputs = output_layer)
    autoencoder.compile(optimizer = 'adam', loss = 'mse')
    
    return autoencoder
    
    
def evaluate_autoencoder_model(y_test, y_pred_ae):
    """ 
    Evaluate an autoencoder model by mapping its predictions to binary labels
    """
       
    metrics = {
      'accuracy': accuracy_score(y_test, y_pred_ae),
      'precision': precision_score(y_test, y_pred_ae),
      'recall': recall_score(y_test, y_pred_ae),
      'f1': f1_score(y_test, y_pred_ae)
      }
        
    return metrics