#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:57:07 2017

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
rate = 1


params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric':'logloss',
    'max_depth':6,
    'colsample_bytree': 0.85,
    'nthread':20,
#    'gamma':0.6,
    'lambda':1,
    'eta':0.4,
#    'silent':0,
#    'alpha':0.01,
#    'subsample':1,
}
n_round = 500
def ceate_feature_map(features):  
    outfile = open('xgb.fmap', 'w')  
    i = 0  
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1
#        print(i)
    outfile.close() 
    
def _XGB_train(df_all):
    
    train_name,day7_data,cv_data = df_all
    train_data = _load_splited_df(path=train_name)
    train_Y = train_data['is_trade'].values
    day7_Y = day7_data['is_trade'].values
    
    drop_cols = ['is_trade']
    train_now = train_data.drop(drop_cols,axis=1)
    day7_now = day7_data.drop(drop_cols, axis=1)
    cv_now = cv_data.drop(drop_cols,axis=1)
    
    train_feat = xgb.DMatrix(train_now.values, label=train_Y)
    day7_feat = xgb.DMatrix(day7_now.values,label=day7_Y)
    test_feat = xgb.DMatrix(cv_now.values)
    watchlist = [(train_feat, 'train'),(day7_feat, 'val')]

    
    clf = xgb.train(params=params, dtrain=train_feat,num_boost_round=n_round,\
        evals=watchlist,early_stopping_rounds=20,verbose_eval=False)
    
    
    predict_train = clf.predict(train_feat)
    predict_day7 = clf.predict(day7_feat)
    predict_cv = clf.predict(test_feat)
    print('train:',cal_log_loss(predict_train, train_Y))
    
    #特征重要度
    features = train_data.columns
    ceate_feature_map(features)
    importance = clf.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    feat_imp = pd.DataFrame(importance, columns=['feature', 'fscore'])
    feat_imp['fscore'] = feat_imp['fscore'] / feat_imp['fscore'].sum()
    
    predict_day7 = np.float16(predict_day7)
    predict_cv = np.float16(predict_cv)
    
    
    return (predict_day7,predict_cv)

def XGB_model_first(day7, cv):
    
    day7_data = day7.copy()
    cv_data = cv.copy()
    
    
    name_list = [cache_pkl_path +'Tree_day/train_'+str(i) for i in range(7)]
    data_in_process = [(train_name,day7_data,cv_data) for train_name in name_list]
    len_name = len(name_list)
    
    predict_day7 = np.zeros((day7_data.shape[0], len_name))
    predict_cv = np.zeros((cv_data.shape[0], len_name))
    
    with multiprocessing.Pool(len_name) as p:
        k_val_list = p.map(_XGB_train,data_in_process)
    
    for i,val in zip(range(len_name),k_val_list):
        print('no %d train'%(i))
        day7_i, cv_i = val
        print('  测试损失:',cal_log_loss(day7_i, day7_data['is_trade'].values))
        print('  验证损失:',cal_log_loss(cv_i, cv_data['is_trade'].values))
        predict_day7[:,i] = day7_i
        predict_cv[:,i] = cv_i
    return (predict_day7, predict_cv)

def XGB_model_second(day7_data, cv_data):
    
    predict_day7, predict_cv = XGB_model_first(day7_data, cv_data)
    
    
    
    #model 2
    train_Y = day7_data['is_trade'].values
    cv_Y = cv_data['is_trade'].values
    
    
    clf = LinearRegression(fit_intercept =True,normalize=True,n_jobs=-1)
    clf.fit(X=predict_day7, y=train_Y)
    
    predict_train_2 = clf.predict(predict_day7)
    predict_cv_2 = clf.predict(predict_cv)
    
    print('train:',cal_log_loss(predict_train_2, train_Y))
    print('test:',cal_log_loss(predict_cv_2, cv_Y))
    print('train mean:',np.mean(predict_train_2))
    print('cv mean:',np.mean(predict_cv_2))
    
    return predict_cv_2
if __name__ == '__main__':
    
    t0 = time.time()
    
    day7_data = _load_splited_df(path=cache_pkl_path +'Tree_day/train_7')
    cv_data = _load_splited_df(path=cache_pkl_path +'Tree_day/cv')
    test_data = _load_splited_df(path=cache_pkl_path +'Tree_day/test')
    
    print('off line')
    _ = XGB_model_second(day7_data, cv_data)
    
    print('on line')
#    day7_data = pd.concat([day7_data, cv_data],axis=0)
    predict_cv_2 = XGB_model_second(day7_data, test_data)
    submmit_result(predict_cv_2,'XGB_LR')