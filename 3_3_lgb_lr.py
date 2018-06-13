#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 13:55:21 2017

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
    'learning_rate': 0.1,          #0.04
    'boosting_type': 'gbdt',
    'objective': 'binary',
#    'verbose': -1,
    'metric': 'binary_logloss',
#    'max_bin':240,
    'bagging_seed':3,
    'feature_fraction_seed':3,
    'num_threads':20
#    'num_leaves':64
#    'lambda_l2': 0.02
#    'lambda_l1':0.05
}
def _LGB_train(df_all):
    
    train_name,day7_data,cv_data = df_all
    train_data = _load_splited_df(path=train_name)
    train_Y = train_data['is_trade'].values
    day7_Y = day7_data['is_trade'].values
    
    drop_cols = ['is_trade']
    train_now = train_data.drop(drop_cols,axis=1)
    day7_now = day7_data.drop(drop_cols, axis=1)
    cv_now = cv_data.drop(drop_cols,axis=1)
    
    
    lgb_train = lgb.Dataset(train_now.values, train_Y)
    lgb_day7 = lgb.Dataset(day7_now.values, day7_Y)
    
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=300,
                    valid_sets=lgb_day7,
                    verbose_eval=False,
                    early_stopping_rounds=50)
    
    predict_train = gbm.predict(train_now.values)
    predict_day7 = gbm.predict(day7_now.values)
    predict_cv = gbm.predict(cv_now.values)
    feat_imp = pd.Series(gbm.feature_importance(), index=train_now.columns).sort_values(ascending=False)
    
    print(feat_imp)
    print('train:',cal_log_loss(predict_train, train_Y))
    predict_day7 = np.float16(predict_day7)
    predict_cv = np.float16(predict_cv)
    
    
    return (predict_day7,predict_cv)

def LGB_model_first(day7, cv):
    
    day7_data = day7.copy()
    cv_data = cv.copy()
    
    
    name_list = [cache_pkl_path +'Tree_day/train_'+str(i) for i in range(7)]
    data_in_process = [(train_name,day7_data,cv_data) for train_name in name_list]
    len_name = len(name_list)
    
    predict_day7 = np.zeros((day7_data.shape[0], len_name))
    predict_cv = np.zeros((cv_data.shape[0], len_name))
    
    with multiprocessing.Pool(len_name) as p:
        k_val_list = p.map(_LGB_train,data_in_process)
    
    for i,val in zip(range(len_name),k_val_list):
        print('no %d train'%(i))
        day7_i, cv_i = val
        print('  测试损失:',cal_log_loss(day7_i, day7_data['is_trade'].values))
        print('  验证损失:',cal_log_loss(cv_i, cv_data['is_trade'].values))
        predict_day7[:,i] = day7_i
        predict_cv[:,i] = cv_i
    return (predict_day7, predict_cv)

def LGB_model_second(day7_data, cv_data):
    
    predict_day7, predict_cv = LGB_model_first(day7_data, cv_data)
    
    
    
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
    _ = LGB_model_second(day7_data, cv_data)
    
    print('on line')
#    day7_data = pd.concat([day7_data, cv_data],axis=0)
    predict_cv_2 = LGB_model_second(day7_data, test_data)
    submmit_result(predict_cv_2,'LGB_LR')