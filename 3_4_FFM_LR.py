#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:28:36 2017

@author: boweiy
"""

import pandas as pd
import numpy as np
import time
import os
from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,load_splited_df
from utils import _load_splited_df,_save_splited_df,splited_apply,split_and_save,load_pickle
from utils import cal_log_loss,submmit_result
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.cross_validation import KFold,train_test_split
import matplotlib.pyplot as plt
import multiprocessing
import random
import xlearn as xl
def _FFM_train(df_all):
    
    train_name,day7_name,cv_name,test_name = df_all
    num = train_name.rsplit('_',1)[-1]
    
    ffm_model = xl.create_ffm()
    ffm_model.setTrain(train_name)
    ffm_model.setValidate(cv_name)
    ffm_model.setTest(day7_name)
    now_result  = cache_pkl_path +'FFM_day/model_day7_'+num
#    ffm_model.disableEarlyStop()
#    
    param = {'task':'binary', 'lr':0.1, 'lambda':0.000005,'epoch':70,'alpha':0.1,'lambda_1':0.01}
    ffm_model.fit(param, "./model.out")
    
    ffm_model.setSigmoid()
    ffm_model.predict("./model.out", now_result)
    predict_day7 = pd.read_csv(now_result,header=None)
    predict_day7 = np.squeeze(predict_day7.values)
    
    ffm_model.setTest(cv_name)
    now_result  = cache_pkl_path +'FFM_day/model_cv__'+num
    ffm_model.predict("./model.out", now_result)
    predict_cv = pd.read_csv(now_result,header=None)
    predict_cv = np.squeeze(predict_cv.values)
    
    ffm_model.setTest(test_name)
    now_result  = cache_pkl_path +'FFM_day/model_test__'+num
    ffm_model.predict("./model.out", now_result)
    predict_test = pd.read_csv(now_result,header=None)
    predict_test = np.squeeze(predict_test.values)
    
    return (predict_day7,predict_cv,predict_test)

def FFM_model_first(day7_data, cv_data, test_data):
    
    day7_name = cache_pkl_path +'FFM_day/train_7'
    cv_name = cache_pkl_path +'FFM_day/cv'
    test_name = cache_pkl_path +'FFM_day/test'
    
    name_list = [cache_pkl_path +'FFM_day/train_'+str(i) for i in range(7)]
    data_in_process = [(train_name,day7_name,cv_name,test_name) for train_name in name_list]
    len_name = len(name_list)
    
    predict_day7 = np.zeros((day7_data.shape[0], len_name))
    predict_cv = np.zeros((cv_data.shape[0], len_name))
    predict_test = np.zeros((test_data.shape[0], len_name))
    
    with multiprocessing.Pool(len_name) as p:
        k_val_list = p.map(_FFM_train,data_in_process)
    
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
    
#    predict_day7, predict_cv, predict_test = FFM_model_first(day7_data,cv_data,test_data)
#    
#    #model 2
#    train_Y = day7_data['is_trade'].values
#    cv_Y = cv_data['is_trade'].values
#    
#    
#    clf = LinearRegression(fit_intercept =True,normalize=True,n_jobs=-1)
#    clf.fit(X=predict_day7, y=train_Y)
#    
#    predict_train_2 = clf.predict(predict_day7)
#    predict_cv_2 = clf.predict(predict_cv)
#    predict_test_2 = clf.predict(predict_test)
#    
#    print('train:',cal_log_loss(predict_train_2, train_Y))
#    print('test:',cal_log_loss(predict_cv_2, cv_Y))
#    print('train mean:',np.mean(predict_train_2))
#    print('cv mean:',np.mean(predict_cv_2))
#    print('test mean:',np.mean(predict_test_2))
#    submmit_result(predict_test_2, 'LR_LR')