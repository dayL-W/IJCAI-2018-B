# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:40:38 2018

@author: Liaowei
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
rate = 1
def _LR_train(df_all):
    
    train_name,day7_data,cv_data,test_data = df_all
    train_data = _load_splited_df(path=train_name)
    train_Y = train_data['is_trade'].values
    
    drop_cols = ['is_trade']
    train_now = train_data.drop(drop_cols,axis=1)
    day7_now = day7_data.drop(drop_cols, axis=1)
    cv_now = cv_data.drop(drop_cols,axis=1)
    test_now = test_data.drop(drop_cols,axis=1)
    
    clf = LogisticRegression(C=1.2, fit_intercept=True, max_iter=300, solver='sag',verbose=1,random_state=7)
    clf.fit(X=train_now.values, y=train_Y)
    
    predict_train = clf.predict_proba(train_now.values)[:,1]
    predict_day7 = clf.predict_proba(day7_now.values)[:,1]
    predict_cv = clf.predict_proba(cv_now.values)[:,1]
    predict_test = clf.predict_proba(test_now.values)[:,1]
    
    print('train:',cal_log_loss(predict_train, train_Y))
#    predict_train = np.float16(predict_train)
    predict_day7 = np.float16(predict_day7)
    predict_cv = np.float16(predict_cv)
    predict_test = np.float16(predict_test)
    return (predict_day7,predict_cv,predict_test)

def LR_model_first(day7, cv, test):
    
    day7_data = day7.copy()
    cv_data = cv.copy()
    test_data = test.copy()
    
    
    name_list = [cache_pkl_path +'LR_day/train_'+str(i) for i in range(7)]
    data_in_process = [(train_name,day7_data,cv_data,test_data) for train_name in name_list]
    len_name = len(name_list)
    
    predict_day7 = np.zeros((day7_data.shape[0], len_name))
    predict_cv = np.zeros((cv_data.shape[0], len_name))
    predict_test = np.zeros((test_data.shape[0], len_name))
    
    with multiprocessing.Pool(len_name) as p:
        k_val_list = p.map(_LR_train,data_in_process)
    
    for i,val in zip(range(len_name),k_val_list):
        print('no %d train'%(i))
        day7_i, cv_i, test_i = val
        print('  测试损失:',cal_log_loss(day7_i, day7_data['is_trade'].values))
        print('  验证损失:',cal_log_loss(cv_i, cv_data['is_trade'].values))
        predict_day7[:,i] = day7_i
        predict_cv[:,i] = cv_i
        predict_test[:,i] = test_i
    return (predict_day7, predict_cv, predict_test)
if __name__ == '__main__':
    
    t0 = time.time()
    
    day7_data = _load_splited_df(path=cache_pkl_path +'LR_day/train_7')
    cv_data = _load_splited_df(path=cache_pkl_path +'LR_day/cv')
    test_data = _load_splited_df(path=cache_pkl_path +'LR_day/test')
    
    predict_day7, predict_cv, predict_test = LR_model_first(day7_data, cv_data, test_data)
    
    #model 2
    train_Y = day7_data['is_trade'].values
    cv_Y = cv_data['is_trade'].values
     
    
    clf = LinearRegression(fit_intercept =True,normalize=True,n_jobs=-1)
    clf.fit(X=predict_day7, y=train_Y)
    
    predict_train_2 = clf.predict(predict_day7)
    predict_cv_2 = clf.predict(predict_cv)
    predict_test_2 = clf.predict(predict_test)
    
    print('train:',cal_log_loss(predict_train_2, train_Y))
    print('test:',cal_log_loss(predict_cv_2, cv_Y))
    print('train mean:',np.mean(predict_train_2))
    print('cv mean:',np.mean(predict_cv_2))
    print('test mean:',np.mean(predict_test_2))
    submmit_result(predict_test_2, 'LR_LR')