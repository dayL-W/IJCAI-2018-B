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
import lightgbm as lgb
rate = 1

params = {
    'max_depth': 6,                 #4
#    'min_data_in_leaf': 40,-
    'feature_fraction': 0.7,       #0.55
    'learning_rate': 0.04,          #0.04
    'boosting_type': 'gbdt',
    'objective': 'binary',
#    'verbose': -1,
    'metric': 'binary_logloss',
#    'max_bin':240,
    'bagging_seed':3,
    'feature_fraction_seed':3,
    'num_threads':40,
    'num_leaves':31,
#    'lambda_l2': 0.02
#    'lambda_l1':0.05
}

def lgb_offline(train,cv):
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
        lgb_train = lgb.Dataset(train_feat.values, train_Y.loc[train_index])
        lgb_cv = lgb.Dataset(cv_feat.values, train_Y.loc[cv_index])
        gbm = lgb.train(params=params,
                        train_set=lgb_train,
                        num_boost_round=6000,
                        valid_sets=lgb_cv,
                        verbose_eval=False,
                        early_stopping_rounds=50)
        #评价特征的重要性
        feat_imp = pd.Series(gbm.feature_importance(), index=train_data.columns).sort_values(ascending=False)
        
        predict_train = gbm.predict(train_feat.values)
        predict_cv = gbm.predict(cv_feat.values)
        test_preds[:,i] = gbm.predict(cv_data.values)
        
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        
        feat_imp = pd.Series(gbm.feature_importance(), index=train_data.columns).sort_values(ascending=False)
        print(gbm.best_iteration)
        print(gbm.best_score)
        print('   训练损失:',cal_log_loss(predict_train, train_Y.loc[train_index]))
        print('   测试损失:',cal_log_loss(predict_cv, train_Y.loc[cv_index]))
    predict_test = np.median(test_preds,axis=1)
    predict_test = predict_test/(predict_test+(1-predict_test)/rate)
    print(params)
    print('训练损失:',cal_log_loss(train_preds/(folds-1), train_Y))
    print('测试损失:',cal_log_loss(cv_preds, train_Y))
    print('验证损失:',cal_log_loss(predict_test, cv_Y))
    print('cv mean:',np.mean(predict_test))
    return feat_imp, gbm

def lgb_online(train, cv, test):
    
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
        lgb_train = lgb.Dataset(train_feat.values, train_Y.loc[train_index])
        lgb_cv = lgb.Dataset(cv_feat.values, train_Y.loc[cv_index])
        gbm = lgb.train(params=params,
                        train_set=lgb_train,
                        num_boost_round=6000,
                        valid_sets=lgb_cv,
                        verbose_eval=False,
                        early_stopping_rounds=50)
        #评价特征的重要性
        feat_imp = pd.Series(gbm.feature_importance(), index=train_data.columns).sort_values(ascending=False)
        
        predict_train = gbm.predict(train_feat.values)
        predict_cv = gbm.predict(cv_feat.values)
        test_preds[:,i] = gbm.predict(test_data.values)
        
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        
        feat_imp = pd.Series(gbm.feature_importance(), index=train_data.columns).sort_values(ascending=False)
        print(gbm.best_iteration)
        print(gbm.best_score)
        print('   训练损失:',cal_log_loss(predict_train, train_Y.loc[train_index]))
        print('   测试损失:',cal_log_loss(predict_cv, train_Y.loc[cv_index]))
    predict_test = np.median(test_preds,axis=1)
    predict_test = predict_test/(predict_test+(1-predict_test)/rate)
    print(params)
    print('训练损失:',cal_log_loss(train_preds/(folds-1), train_Y))
    print('测试损失:',cal_log_loss(cv_preds, train_Y))
    print('test mean:',np.mean(predict_test))
    submmit_result(predict_test, 'LGB')
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
    
    feat_imp,predict_test = lgb_offline(train_data, cv_data)
    feat_imp,predict_test = lgb_online(train_data, cv_data, test_data)
    t1 = time.time()
    print('cost:',t1-t0)