# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:39:56 2025

@author: vscerra

"""
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import numpy as np

def load_nsl_kdd(path: str) -> pd.DataFrame:
    """ 
    Load NSL-KDD dataset from .csv or .txt
    Parameters: 
      path (str): file path to dataset
    Returns: 
      df (pd.DataFrame): Loaded dataset
    """
    col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"
      ]
    
    df = pd.read_csv(path, names = col_names)
    return df
  
  
def clean_nsl_kdd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unneeded columns and fix any obvious data issues
    Paramters:
      df (pd.DataFrame): Raw dataset
    Returns:
      pd.DataFrame: Cleaned dataset
    """
    
    df = df.copy()
    if 'num_outbound_cmds' in df.columns:
        df.drop(columns = ['num_outbound_cmds'], inplace = True)
    if 'difficulty_level' in df.columns:
        df.drop(columns = ['difficulty_level'], inplace = True)
    return df
    

def encode_nsl_kdd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features with LabelEncoder
    Parameters: 
      df (pd.DataFrame): Cleaned dataset
    Returns:
      df (pd.DataFrame): Encoded dataset
    """
    df = df.copy()
    categorical_columns = df.select_dtypes(include = 'object').columns.tolist()
    for col in categorical_columns:
        if col != 'label':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    return df


def normalize_features(df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize numerical features using MinMaxScaler or StandardScaling
    Parameters:
      df (pd.DataFrame): Input dataframe with numeric features
      method (str): "minmax" or "standard"
    Returns: 
      pd.DataFrame: Scaled dataset
    
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include = np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'binary_label'] # to skip the target column
    
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
  
  
def select_features_variance(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """ 
    Remove featuers with variance below a threshold
    Parameters:
      df (pd.Dataframe): Input dataframe
      threshold (float): Variance threshold (default = 0.01)
    Returns: 
      pd.DataFrame: Reduced dataset
    """
    df = df.copy()
    selector = VarianceThreshold(threshold = threshold)
    features = df.select_dtypes(include = np.number).drop(columns = ['binary_label'], errors = 'ignore')
    reduced_array = selector.fit_transform(features)
    selected_cols = features.columns[selector.get_support()]
    
    #build new DataFrame from selected features
    df_reduced = pd.DataFrame(reduced_array, columns = selected_cols, index = df.index)
    
    #reattach the target column, if present
    if 'binary_label' in df.columns:
        df_reduced['binary_label'] = df['binary_label']
    return df_reduced
  

def select_features_corr(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """ 
    Drop one of any pair of features with correlation > threshold
    Parameters:
      df (pd.DataFrame): input dataframe
    Returns:
      pd.DataFrame: reduced dataset
    """
    df = df.copy()
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df.drop(columns = to_drop, inplace = True)
    return df
  

def preprocess_nsl_kdd(
    path: str,
    normalize: bool = True,
    scaler: str = 'standard',
    apply_variance_filter: bool = True,
    variance_threshold: float = 0.01,
    apply_corr_filter: bool = True,
    corr_threshold: float = 0.9) -> pd.DataFrame:
    """
    Full preprocessing pipeline for NSL-KDD dataset
    Parameters: 
      path (str): Path to raw dataset
       normalize (bool): Whether to normalize numeric features
       scaler (str): 'standard' or 'minmax'
       apply_variance_filter (bool): Whether to filter low-variance features
       variance_threshold (float): Variance threshold
       apply_corr_filter (bool): Whether to drop highly correlated features
       corr_threshold (float): Correlation threshold
    Returns:
       pd.DataFrame: Fully preprocessed dataset
    """
    from src.utils import simplify_labels
    
    df = load_nsl_kdd(path)
    df = clean_nsl_kdd(df)
    df = encode_nsl_kdd(df)
    df = simplify_labels(df)
    
    if normalize:
        df = normalize_features(df, method = 'standard')
    if apply_variance_filter:
        df = select_features_variance(df, threshold = variance_threshold)
    if apply_corr_filter:
        df = select_features_corr(df, threshold = corr_threshold)
        
    return df
  