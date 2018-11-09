# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 21:21:10 2018

@author: slsefe
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)
import lightgbm as lgb
import xgboost as xgb
from sklearn import linear_model
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error,median_absolute_error, explained_variance_score, r2_score
print(__doc__)

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_seq_items', None)

import warnings
import time
import os
warnings.filterwarnings("ignore")
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def read_data(train_file, test_file):
    # read data
    train_df = pd.read_csv(train_file,sep='\t')
    test_df = pd.read_csv(test_file,sep='\t')
    return train_df,test_df

# drop abnormal value in train data
def drop_except(data, whi):
    for col in data.columns:
        if col != 'target':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            delta_Q = Q3 - Q1
            upper = Q3 + whi*delta_Q
            lower = Q1 - whi*delta_Q
            data[col] = data[col].apply(lambda x: np.nan if (x>upper or x<lower) else x)
    return data

# V9,V17,V22,V23,V24,V28,V33,V34,V35
def cut_and_dummy(data,cols,bins_list):
    for col,bins in zip(cols,bins_list):
        # data[col+'_'] = train_df[col].apply(lambda x:x/scale)
        data[col+'_group'] = pd.cut(x=data[col],bins=bins)
        dummy = pd.get_dummies(data[[col+'_group']],prefix=col)
        del data[col+'_group']
        data = pd.concat([data,dummy],axis=1)
    return data

# V10,18,V23,V30,V32,V36
def add_min_fea(data,cols,thresholds):
    for col,threshold in zip(cols,thresholds):  
        data[col+'_min'] = data[col].apply(lambda x:1 if x<threshold else 0)
    return data

# minus max,times -1,np.log
def log_transform(data,cols):
    for col in cols:
        data[col+'_log'] = np.log1p(data[col].max()-data[col])
    return data

# exp1m towards numericial features
def exp_transform(data,cols):
    for col in cols:
        data[col+'_exp'] = np.expm1(data[col])
    return data

# plus max, sqrt
def sqrt_transform(data,cols):
    for col in cols:
        data[col+'_sqrt'] = np.sqrt(data[col]-data[col].min())
    return data

def square_transform(data,cols):
    for col in cols:
        data[col+'_square'] = np.square(data[col])
    return data

# caeate poly feature
def poly_transform(X_train, X_test, degree):
    poly = PolynomialFeatures(degree=degree)
    poly.fit(X_train)
    X_train = poly.transform(X_train)
    X_test = poly.transform(X_test)
    return X_train,X_test

# use f-regression select features
def features_select(X_train, y_train, X_test, n):
    freg = SelectKBest(f_regression, k=n).fit(X_train, y_train)
    X_train = freg.transform(X_train)
    X_test = freg.transform(X_test)
    return X_train,X_test

# PCA
def pca_decomposition(X_train, X_test, n):
    pca = PCA(n_components=n)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return X_train,X_test


def feature_engineering(train_df,test_df,drop_abnormal=True,whi=3,one_hot=True,add_min=True,log=True,
                        exp=True,sqrt=True,square=True,poly=False,poly_degree=2,selectKBest=True,K=100,
                        pca=False,n_components=30):
    if drop_abnormal:
        # drop abnormal data
        train_df = drop_except(train_df,whi)
        train_df = train_df.dropna()
        train_df = train_df.reset_index(drop=True)
        
    # combine train data and test data
    test_df['target']=np.nan
    data = pd.concat([train_df,test_df])
    
    if one_hot:
        # 39-183
        cate_cols = ['V9','V17','V22','V23','V28','V33','V34','V35']
        bins_list=[17,17,17,17,17,17,17,17]
        data = cut_and_dummy(data, cate_cols, bins_list)
        
    if add_min:
        # 183-190
        min_cols = ['V10','V18','V23','V28','V30','V32','V36']
        thresholds = [-2.4,-3.4,-5,-2,-4,-3.8,-2.5]
        data = add_min_fea(data,min_cols,thresholds)
        
    # 190-310
    num_cols = ['V0','V1','V2','V3','V4','V5','V6','V7','V8','V10','V11','V12','V13','V14','V15','V16',
                'V18','V19','V20','V21','V24','V25','V26','V27','V29','V30','V31','V32','V36','V37']
    if log:
        data = log_transform(data,num_cols)
    if exp:
        data = exp_transform(data,num_cols)
    if sqrt:
        data = sqrt_transform(data,num_cols)
    if square:
        data = square_transform(data,num_cols)

    # split train data and test data
    train_df = data[data.target.notnull()]
    X_train = np.array(train_df.drop(['target'],axis=1))
    y_train = np.array(train_df['target']).reshape((-1,))
    test_df = data[data.target.isnull()]
    X_test = np.array(test_df.drop(['target'],axis=1))
    
    if poly:
        X_train,X_test = poly_transform(X_train,X_test,poly_degree)
    if selectKBest:
        X_train,X_test = features_select(X_train,y_train,X_test,K)
    if pca:
        X_train,X_test = pca_decomposition(X_train,X_test,n_components)
    
    print(X_train.shape,y_train.shape,X_test.shape)
    return X_train,y_train,X_test


def train_model(X_train, y_train, X_test, submit_dir, test_ratio=0.2, seed=2019, cv=5):
    # split train data and validation data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_ratio, random_state=seed)
    print(X_train.shape,y_train.shape,X_val.shape,y_val.shape,X_test.shape)


    params={'alpha':[1e-5,1e-4,1e-3,0.01,0.1,1,10]}
    lasso = linear_model.Lasso()
    grid = GridSearchCV(lasso, params, cv=cv, scoring=['r2','mean_squared_error'], refit='mean_squared_error')
    grid.fit(X_train, y_train)
    lasso_model=grid.best_estimator_

    y_train_pred=lasso_model.predict(X_train)
    print('MSE:' + str(mean_squared_error(y_train, y_train_pred)))
    y_val_pred=lasso_model.predict(X_val)
    print('MSE:' + str(mean_squared_error(y_val, y_val_pred)))
    
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    submit_file = submit_dir + now + str(np.round(mean_squared_error(y_val, y_val_pred),4)) + '.txt'
    pd.Series(lasso_model.predict(X_test)).to_csv(submit_file,index = False)


train_file = 'd:/zhengqi/zhengqi_train.txt'
test_file = 'd:/zhengqi/zhengqi_test.txt'
submit_dir = 'd:/zhengqi/'
train_df,test_df = read_data(train_file,test_file)
X_train,y_train,X_test = feature_engineering(train_df,test_df)
train_model(X_train,y_train,X_test,submit_dir=submit_dir)


