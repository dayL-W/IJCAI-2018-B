#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 19:25:57 2017

@author: boweiy
"""

import pandas as pd
import numpy as np
import time
import os
from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,load_splited_df
from utils import _load_splited_df,_save_splited_df,splited_apply,split_and_save
from utils import cal_log_loss,submmit_result
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.cross_validation import KFold,train_test_split
import matplotlib.pyplot as plt
import multiprocessing
import random
import xgboost as xgb
import operator

def ceate_feature_map(features):  
    outfile = open('xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1
#        print(i)
    outfile.close() 
    
rate = 1

params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric':'logloss',
    'max_depth':6,
    'colsample_bytree': 0.85,
    'nthread':15,
#    'gamma':0.6,
    'lambda':1,
    'eta':0.4,
#    'silent':0,
#    'alpha':0.01,
#    'subsample':1,
}

n_round=500
def xgb_offline(train,cv):
    train_data = train.copy()
    cv_data = cv.copy()
    
    train_Y = train_data['is_trade']
    cv_Y = cv_data['is_trade']
    
    drop_cols = ['is_trade']
    train_data.drop(drop_cols,axis=1,inplace=True)
    cv_data.drop(drop_cols,axis=1,inplace=True)
    
    folds = 5
    kf = KFold(len(train_data), n_folds = folds, shuffle=True, random_state=7)
    train_preds = np.zeros(train_data.shape[0])
    cv_preds = np.zeros(train_data.shape[0])
    test_preds = np.zeros((cv_data.shape[0], 5))
    for i, (train_index, cv_index) in enumerate(kf):
        
        train_feat = train_data.loc[train_index]
        cv_feat = train_data.loc[cv_index]
        
        print('第{}次训练...'.format(i))
        train_feat = xgb.DMatrix(train_feat.values, label=train_Y[train_index])
        cv_feat = xgb.DMatrix(cv_feat.values,label=train_Y[cv_index])
        test_feat = xgb.DMatrix(cv_data.values)
        watchlist = [(train_feat, 'train'),(cv_feat, 'val')]

        
        clf = xgb.train(params=params, dtrain=train_feat,num_boost_round=n_round,\
            evals=watchlist,early_stopping_rounds=7,verbose_eval=False)
    
        predict_train = clf.predict(train_feat)
        predict_cv = clf.predict(cv_feat)
        predict_test = clf.predict(test_feat)
        
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        test_preds[:,i] = predict_test
        #特征重要度
        features = train_data.columns
        ceate_feature_map(features)
        importance = clf.get_fscore(fmap='xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        feat_imp = pd.DataFrame(importance, columns=['feature', 'fscore'])
        feat_imp['fscore'] = feat_imp['fscore'] / feat_imp['fscore'].sum()
        print(clf.best_iteration)
        print(clf.best_score)
        print('   训练损失:',cal_log_loss(predict_train, train_Y.loc[train_index]))
        print('   测试损失:',cal_log_loss(predict_cv, train_Y.loc[cv_index]))
    predict_test = np.median(test_preds,axis=1)
    predict_test = predict_test/(predict_test+(1-predict_test)/rate)
    print(params)
    print('训练损失:',cal_log_loss(train_preds/(folds-1), train_Y))
    print('测试损失:',cal_log_loss(cv_preds, train_Y))
    print('验证损失:',cal_log_loss(predict_test, cv_Y))
    print('cv mean:',np.mean(predict_test))
    return feat_imp, predict_test

def xgb_online(train, cv, test):
    
    train_data = train.copy()
    cv_data = cv.copy()
    test_data = test.copy()
    
    train_data = pd.concat([train_data, cv_data],axis=0)
    train_data.reset_index(inplace=True,drop=True)
    train_Y = train_data['is_trade']
    
    drop_cols = ['is_trade']
    train_data.drop(drop_cols,axis=1,inplace=True)
    test_data.drop(drop_cols,axis=1,inplace=True)
    
    folds = 5
    kf = KFold(len(train_data), n_folds = folds, shuffle=True, random_state=7)
    train_preds = np.zeros(train_data.shape[0])
    cv_preds = np.zeros(train_data.shape[0])
    test_preds = np.zeros((test_data.shape[0], 5))
    for i, (train_index, cv_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        train_feat = train_data.loc[train_index]
        cv_feat = train_data.loc[cv_index]
        
        train_feat = xgb.DMatrix(train_feat.values, label=train_Y[train_index])
        cv_feat = xgb.DMatrix(cv_feat.values,label=train_Y[cv_index])
        test_feat = xgb.DMatrix(test_data.values)
        watchlist = [(train_feat, 'train'),(cv_feat, 'val')]

        
        clf = xgb.train(params=params, dtrain=train_feat,num_boost_round=n_round,\
            evals=watchlist,early_stopping_rounds=7,verbose_eval=False)
    
        predict_train = clf.predict(train_feat)
        predict_cv = clf.predict(cv_feat)
        predict_test = clf.predict(test_feat)
        
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        test_preds[:,i] = predict_test
        #特征重要度
        features = train_data.columns
        ceate_feature_map(features)
        importance = clf.get_fscore(fmap='xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        feat_imp = pd.DataFrame(importance, columns=['feature', 'fscore'])
        feat_imp['fscore'] = feat_imp['fscore'] / feat_imp['fscore'].sum()
        print(clf.best_iteration)
        print(clf.best_score)
        print('   训练损失:',cal_log_loss(predict_train, train_Y.loc[train_index]))
        print('   测试损失:',cal_log_loss(predict_cv, train_Y.loc[cv_index]))
    predict_test = np.median(test_preds,axis=1)
    predict_test = predict_test/(predict_test+(1-predict_test)/rate)
    print(params)
    print('训练损失:',cal_log_loss(train_preds/(folds-1), train_Y))
    print('测试损失:',cal_log_loss(cv_preds, train_Y))
    print('test mean:',np.mean(predict_test))
    submmit_result(predict_test, 'XGB')
    return feat_imp,predict_test
if __name__ == '__main__':
    
    t0 = time.time()
    
    train_data = _load_splited_df(path=cache_pkl_path +'Tree_day7/train')
    cv_data = _load_splited_df(path=cache_pkl_path +'Tree_day7/cv')
    test_data = _load_splited_df(path=cache_pkl_path +'Tree_day7/test')
    
    drop_cols = ['is_afternoon','is_evening', 'moring_12hour','after_12hour','day']
    train_data.drop(drop_cols, axis=1, inplace=True)
    cv_data.drop(drop_cols, axis=1, inplace=True)
    test_data.drop(drop_cols, axis=1, inplace=True)
    print(train_data.shape)
    print(cv_data.shape)
    print(test_data.shape)
    
    feat_imp,predict_test = xgb_offline(train_data, cv_data)
    feat_imp,predict_test = xgb_online(train_data, cv_data, test_data)
    t1 = time.time()
    print('cost:',t1-t0)