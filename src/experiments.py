# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 11:23:30 2025

@author: vscerra
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.modeling import build_autoencoder

def run_autoencoder_experiment(
    X_train_normal,
    X_test,
    y_test,
    encoding_dim = 16,
    dropout_rate = 0.1,
    learning_rate = 1e-4,
    epochs = 100,
    batch_size = 128, 
    threshold_percentile = 99,
    verbose = 0
    ):
    """ 
    Trains and evaluates an autoencoder anomaly detector and returns metrics
    """
    input_dim = X_train_normal.shape[1]
    
    # build model
    autoencoder = build_autoencoder(input_dim, encoding_dim = encoding_dim)
    
    # compile with custom learning rate
    autoencoder.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'mse')
    
    # early stopping
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
    
    # train
    history = autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs = epochs,
        batch_size = batch_size,
        shuffle = True,
        validation_split = 0.1,
        verbose = verbose,
        callbacks = [early_stop]
        )
    
    # get reconstruction error on test set
    X_test_pred = autoencoder.predict(X_test, verbose = 0)
    recon_error = np.mean(np.square(X_test - X_test_pred), axis = 1)
    
    # use threshold from training reconstruction error
    train_recon = autoencoder.predict(X_train_normal, verbose = 0)
    train_error = np.mean(np.square(X_train_normal - train_recon), axis = 1)
    threshold = np.percentile(train_error,  threshold_percentile)
    
    # predict
    y_pred = (recon_error > threshold).astype(int)
    
    # metrics
    metrics= {
        "encoding_dim": encoding_dim,
        "dropout": dropout_rate,
        "lr": learning_rate,
        "epochs_run": len(history.history["loss"]),
        "f1": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "threshold": threshold
        }
    
    return metrics