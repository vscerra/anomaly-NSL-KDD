# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:39:54 2025

@author: vscerra
"""
import pandas as pd

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