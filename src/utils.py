# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:39:54 2025

@author: vscerra
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve

def simplify_labels(df: pd.DataFrame, label_col: str = 'label') -> pd.DataFrame:
    """ 
    Convert multi-class attack types into binary labels (normal vs. anomaly)
    Paramters:
      df (pd.DataFrame): DataFrame with original 'label' column
    Returns:
      pd.DataFrame with binary 'binary_label' column
    """
    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df['binary_label'] = df[label_col].apply(lambda x: 0 if x == 'normal' else 1)
    return df
  

def plot_confusion(y_true, y_pred, title = "Confusion Matrix", labels = [0, 1]):
    """ 
    Plot a confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels = labels)
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = labels, yticklabels = labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()
    
    
def plot_precision_recall(y_true, y_score, title = 'Precision-Recall Curve'):
    """
    Plot a precision-recall curve using predicted probabilites or anomaly scores
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.plot(recall, precision, color = 'darkorange', lw =2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)
    plt.show()
