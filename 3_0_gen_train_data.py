# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 20:46:04 2018

@author: Liaowei
"""

import pandas as pd
import numpy as np
import time
import datetime
import os
import random
from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,load_splited_df
from utils import _load_splited_df,_save_splited_df,splited_apply,split_and_save,data2libffm,my_data2libffm
#from smooth import BayesianSmoothing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import minmax_scale
import multiprocessing

 # In[]:读取特征并整合成训练数据和测试数据
def gen_tree_data(df):
    
    '''
    1 split and save data
    '''
    
    all_data = df.copy()
    
    #split
    train = all_data.loc[all_data.is_trade!=-1,:]
    test = all_data.loc[all_data.is_trade==-1,:]
    
    day7_index = train.loc[train.day==7].index
    cv_index = random.sample(list(day7_index),520000)
    cv = train.loc[cv_index]
    train.drop(cv_index,axis=0,inplace=True)
    
    cv.reset_index(inplace=True,drop=True)
    test.reset_index(inplace=True,drop=True)
    
    print('train:',train.shape)
    print('cv:',cv.shape)
    print('test:',test.shape)
    
    cv_path = (cv, cache_pkl_path+'Tree_day/cv')
    test_path = (test, cache_pkl_path+'Tree_day/test')
    _save_splited_df(test_path)
    _save_splited_df(cv_path)
    
#    days = list(set(train.day))
    for day in range(8):
        print('save train %d'%(day))
        now_train = train.loc[train.day==day,:]
        now_train.reset_index(inplace=True,drop=True)
        now_train_path = (now_train,cache_pkl_path+'Tree_day/train_%d'%(day))
        _save_splited_df(now_train_path)
    
    
def gen_onehot_data(df_col):
    df,col = df_col
    return pd.get_dummies(df,prefix=col)
def gen_onehot_filled_data(df_col):
    ser,col = df_col
    
    
    x = ser.isnull()
    col_str = col+'#nan'
    ser.fillna(0,inplace=True)
    
    df_fill = ser.to_frame()
    df_fill[col_str]=x.astype(int)
    return df_fill
def gen_LR_data(df):
    '''
    1 read data with pthread
    2 drop some abnormal sample and cols
    3 onehot some cols
    4 onehot cvr and fillna with 0
    5 min_max some cols note: transform as float16
    6 split in train cv test
    7 reindex and save data
    8 gen ffm data
    '''
    all_data = df.copy()
    
    drop_cols = ['user_id','item_id','shop_id','item_brand_id','item_city_id']
    onehot_cols = ['user_gender_id','user_age_level','user_occupation_id','item_price_level',\
                   'item_sales_level','item_pv_level','item_collected_level','context_page_id',\
                   'user_star_level','day','shop_review_num_level','shop_star_level','hour','second_cate']
#    onehot_and_fill_cols = ['max_cp_cvr', 'min_cp_cvr','mean_cp_cvr','item_id_count#buy',\
#                            'user_id_count#cvr','item_id_count#cvr','item_brand_id_count#cvr',\
#                            'second_cate_count#cvr','shop_id_count#cvr','user_id_count#buy',\
#                            'item_brand_id_count#buy','second_cate_count#buy','shop_id_count#buy',\
#                            'shop_id_count#visit','item_brand_id_count#visit','query_item_second_cate_sim',\
#                            'item_id_count#visit','second_cate_count#visit','user_id_count#visit',\
#                            ]
    onehot_and_fill_cols = ['item_id_count#buy',\
                            'user_id_count#cvr','item_id_count#cvr','item_brand_id_count#cvr',\
                            'second_cate_count#cvr','shop_id_count#cvr','user_id_count#buy',\
                            'item_brand_id_count#buy','second_cate_count#buy','shop_id_count#buy',\
                            'shop_id_count#visit','item_brand_id_count#visit','query_item_second_cate_sim',\
                            'item_id_count#visit','second_cate_count#visit','user_id_count#visit',\
                            ]
    cols = ['max_cp_cvr', 'min_cp_cvr','mean_cp_cvr','max_v_cp_cvr', 'top3_v_mean_cp_cvr','uc_cvr','uc_visit','uc_buy']
    for i in cols:
        onehot_and_fill_cols.append(i+'#-1day')
        onehot_and_fill_cols.append(i+'#-2day')
        onehot_and_fill_cols.append(i+'#-4day')
    other_cols = list(set(all_data.columns)-(set(drop_cols)|set(onehot_cols)|set(onehot_and_fill_cols))-set(['is_trade']))
    
    minmax_cols = other_cols+onehot_and_fill_cols
    #drop cols
    all_data.drop(drop_cols, axis=1, inplace=True)
    #    drop_index = all_data.loc[all_data.shop_score_service==-1].index
    #    all_data.drop(drop_index, axis=0, inplace=True)
    
    #onehot
    df_col = [(all_data[col],col) for col in onehot_cols]
    len_cols = len(onehot_cols)
    with multiprocessing.Pool(len_cols) as p:
        now_df = p.map(gen_onehot_data, df_col)
    all_data.drop(onehot_cols,axis=1,inplace=True)
    all_data = pd.concat(now_df+[all_data], axis=1)
    
    #onehot and fill
    df_col = [(all_data[col],col) for col in onehot_and_fill_cols]
    len_cols = len(onehot_and_fill_cols)
    with multiprocessing.Pool(len_cols) as p:
        now_df_fill = p.map(gen_onehot_filled_data, df_col)
    all_data.drop(onehot_and_fill_cols,axis=1,inplace=True)
    all_data = pd.concat(now_df_fill+[all_data], axis=1)
    
    #minmax
    minmax_values = all_data[minmax_cols].values
    X = minmax_scale(minmax_values)
    X = np.float16(X)
    all_data[minmax_cols] = X
    
    #split
    train = all_data.loc[all_data.is_trade!=-1,:]
    test = all_data.loc[all_data.is_trade==-1,:]
    
    day7_index = train.loc[train['day_7']==1].index
    cv_index = random.sample(list(day7_index),520000)
    cv = train.loc[cv_index]
    train.drop(cv_index,axis=0,inplace=True)
    
    cv.reset_index(inplace=True,drop=True)
    test.reset_index(inplace=True,drop=True)
    
    print('train:',train.shape)
    print('cv:',cv.shape)
    print('test:',test.shape)
    cv_path = (cv, cache_pkl_path+'LR_day/cv')
    test_path = (test, cache_pkl_path+'LR_day/test')
    print('save test')
    _save_splited_df(test_path)
    print('save cv')
    _save_splited_df(cv_path)
    
    
#    days = list(set(train.day))
    for day in range(8):
        print('save train %d'%(day))
        day_str = 'day_'+str(day)
        now_train = train.loc[train[day_str]==1,:]
        now_train.reset_index(inplace=True,drop=True)
        now_train_path = (now_train,cache_pkl_path+'LR_day/train_%d'%(day))
        _save_splited_df(now_train_path)
        
def _data2libffm(path):
    print(path)
    data = _load_splited_df(path)
    save_path = cache_pkl_path+'FFM_day/'+path.rsplit('/')[-1]
    my_data2libffm(data, save_path)

def gen_FFM_data():
    '''
    1 read file
    2 tolibffm
    '''
    
    cv_path = cache_pkl_path+'LR_day/cv'
    test_path = cache_pkl_path+'LR_day/test'
    
    data_in_process = [(cv_path),(test_path)]
    data_in_process.extend((cache_pkl_path+'LR_day/train_%d'%(day)) for day in range(8))
    with multiprocessing.Pool(len(data_in_process)) as p:
        p.map(_data2libffm,data_in_process)

def gen_Tree_day7(df):
    
    all_data = df.loc[df.day==7,:]
    train = all_data.loc[all_data.is_trade!=-1,:]
    test = all_data.loc[all_data.is_trade==-1,:]
    
    
    all_index = train.index
    cv_index = random.sample(list(all_index),210000)
    cv = train.loc[cv_index]
    train.drop(cv_index,axis=0,inplace=True)
    
    train.reset_index(inplace=True,drop=True)
    cv.reset_index(inplace=True,drop=True)
    test.reset_index(inplace=True,drop=True)
    
    print('train:',train.shape)
    print('cv:',cv.shape)
    print('test:',test.shape)
    
    train_path = (train, cache_pkl_path+'Tree_day7/train')
    cv_path = (cv, cache_pkl_path+'Tree_day7/cv')
    test_path = (test, cache_pkl_path+'Tree_day7/test')
    _save_splited_df(train_path)
    _save_splited_df(test_path)
    _save_splited_df(cv_path)
    
    
if __name__ == '__main__':
    
    name_list = os.listdir(feature_data_path)
    name_list = [feature_data_path+name+'/' for name in name_list]
    name_len = len(name_list)
    df = pd.DataFrame()
    for name in  name_list:
        print(name)
        feature_all, feature_list = load_splited_df(name)
        df = pd.concat([df, feature_all],axis=1)
#    gen_LR_data(df)
#    gen_tree_data(df)
#    gen_FFM_data()
    gen_Tree_day7(df)