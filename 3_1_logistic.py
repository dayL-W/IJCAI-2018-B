# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:40:38 2018

@author: Liaowei
"""

import pandas as pd
import numpy as np
import time
import datetime
import os
from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,load_splited_df
from utils import _load_splited_df,_save_splited_df,splited_apply,split_and_save
from utils import cal_log_loss,submmit_result
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import minmax_scale
import xgboost as xgb
import operator 
from sklearn.cross_validation import KFold,train_test_split
import matplotlib.pyplot as plt
import multiprocessing
import random
rate = 1
def _LR_train(df_all):
    
    train_df,cv_df,test_df = df_all
    train_Y = train_df['is_trade'].values
    
    drop_cols = ['is_trade']
    train_now = train_df.drop(drop_cols,axis=1)
    cv_now = cv_df.drop(drop_cols,axis=1)
    test_now = test_df.drop(drop_cols,axis=1)
    
    clf = LogisticRegression(C=1.2, fit_intercept=True, max_iter=100, solver='sag',verbose=1,random_state=7)
    clf.fit(X=train_now.values, y=train_Y)
    
    predict_train = clf.predict_proba(train_now.values)[:,1]
    predict_cv = clf.predict_proba(cv_now.values)[:,1]
    predict_test = clf.predict_proba(test_now.values)[:,1]
    predict_train = np.float16(predict_train)
    predict_cv = np.float16(predict_cv)
    predict_test = np.float16(predict_test)
    return (predict_train,predict_cv,predict_test)

def LR_offline(train, cv):
    print('off line')
    train_data = train.copy()
    cv_data = cv.copy()
    train_Y = train_data['is_trade']
    cv_Y = cv_data['is_trade']
    
    fold = 5
    kf = KFold(len(train_data), n_folds = fold, shuffle=True, random_state=520)
    train_preds = np.zeros(train_data.shape[0])
    cv_preds = np.zeros(train_data.shape[0])
    test_preds = np.zeros((cv_data.shape[0], fold))
    
    data_in_process = [(train_data.loc[train_index],train_data.loc[cv_index],cv_data) for i,(train_index,cv_index) in enumerate(kf)]
    index_all = [(train_index,cv_index) for i,(train_index,cv_index) in enumerate(kf)]
    with multiprocessing.Pool(fold) as p:
        k_val_list = p.map(_LR_train,data_in_process)
    for i,index,val in zip(range(fold), index_all, k_val_list):
        print('no %d train'%(i))
        train_index,cv_index = index
        predict_train,predict_cv,predict_test = val
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        test_preds[:,i] = predict_test
        print('  训练损失:',cal_log_loss(predict_train, train_Y[train_index]))
        print('  测试损失:',cal_log_loss(predict_cv, train_Y[cv_index]))
    '''
    
    for i, (train_index, cv_index) in enumerate(kf):
        print('no %d train:'%(i))
        train_feat = train_data.loc[train_index]
        cv_feat = train_data.loc[cv_index]
        
        predict_train,predict_cv,predict_test = _LR_train((train_feat,cv_feat,cv_data))
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        test_preds[:,i] = predict_test
        
        print('  训练损失:',cal_log_loss(predict_train, train_Y[train_index]))
        print('  测试损失:',cal_log_loss(predict_cv, train_Y[cv_index]))
    '''
    predict_test = np.median(test_preds,axis=1)
    print('mean:',np.mean(predict_test))
    print('训练损失:',cal_log_loss(train_preds/(fold-1), train_Y))
    print('测试损失:',cal_log_loss(cv_preds, train_Y))
    print('验证损失:',cal_log_loss(predict_test, cv_Y))
    
    
def LR_online(train_data, cv_data, test_data):
    print('on line')
    train_data = pd.concat([train_data, cv_data],axis=0)
    train_data.reset_index(inplace=True,drop=True)
    train_Y = train_data['is_trade']
    
    fold = 5
    kf = KFold(len(train_data), n_folds = fold, shuffle=True, random_state=520)
    train_preds = np.zeros(train_data.shape[0])
    cv_preds = np.zeros(train_data.shape[0])
    test_preds = np.zeros((test_data.shape[0], fold))
    
    data_in_process = [(train_data.loc[train_index],train_data.loc[cv_index],test_data) for i,(train_index,cv_index) in enumerate(kf)]
    index_all = [(train_index,cv_index) for i,(train_index,cv_index) in enumerate(kf)]
    with multiprocessing.Pool(fold) as p:
        k_val_list = p.map(_LR_train,data_in_process)
    for i,index,val in zip(range(fold), index_all, k_val_list):
        print('no %d train'%(i))
        train_index,cv_index = index
        predict_train,predict_cv,predict_test = val
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        test_preds[:,i] = predict_test
        print('  训练损失:',cal_log_loss(predict_train, train_Y[train_index]))
        print('  测试损失:',cal_log_loss(predict_cv, train_Y[cv_index]))
    '''
    for i, (train_index, cv_index) in enumerate(kf):
        print('no %d train:'%(i))
        train_feat = train_data.loc[train_index]
        cv_feat = train_data.loc[cv_index]
        
        predict_train,predict_cv,predict_test = _LR_train((train_feat,cv_feat,test_data))
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        test_preds[:,i] = predict_test
        
        print('  训练损失:',cal_log_loss(predict_train, train_Y[train_index]))
        print('  测试损失:',cal_log_loss(predict_cv, train_Y[cv_index]))
    '''
    predict_test = np.median(test_preds,axis=1)
    print('mean:',np.mean(predict_test))
    print('训练损失:',cal_log_loss(train_preds/(fold-1), train_Y))
    print('测试损失:',cal_log_loss(cv_preds, train_Y))
    submmit_result(predict_test, 'LR')
    
if __name__ == '__main__':
    
    t0 = time.time()
#    train_data,train_list = load_splited_df(cache_pkl_path+'train_LR/')
    train_data = _load_splited_df(path=cache_pkl_path +'cv_LR')
    test_data = _load_splited_df(path=cache_pkl_path +'test_LR')
    
    cv_index = random.sample(list(train_data.index),200000)
    cv_data = train_data.loc[cv_index,:]
    train_data.drop(cv_index,axis=0,inplace=True)
    
    train_data.reset_index(inplace=True,drop=True)
    cv_data.reset_index(inplace=True,drop=True)
    LR_offline(train_data, cv_data)
    LR_online(train_data, cv_data, test_data)