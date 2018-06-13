#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:56:41 2017

@author: boweiy
"""

import pandas as pd
import numpy as np
from   tqdm import tqdm
import time
import datetime
import os

from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,load_splited_df
#from smooth import BayesianSmoothing
from utils import _load_splited_df,_save_splited_df,splited_apply,split_and_save
import multiprocessing

# In[]
def gen_day_visit(gp_col):
    col= gp_col
    
    select_df = pd.DataFrame()
    
    for day in list(set(all_data.day)):
        now_data = all_data.loc[all_data.day==day,:]
        
        before_1day_data = all_data.loc[all_data.day==(day-1),:]
        before_2day_data = all_data.loc[(all_data.day==(day-1))|(all_data.day==(day-2)),:]
        
        now_df = pd.DataFrame()
        gp1 = before_1day_data.groupby(['user_id',col])
        gp1_count = gp1.count().iloc[:,0]
        x = now_data[['user_id',col]].values
        if len(gp1_count) !=0:
            k = gp1_count.loc[[tuple(i) for i in x]]
        else:
            k = pd.Series(np.full(len(x),np.nan))
        print(k.count()/len(x))
        col_str = 'user_visit_'+col+'#-1'
        now_df[col_str] = k.values
     
        gp2 = before_2day_data.groupby(['user_id',col])
        gp2_count = gp2.count().iloc[:,0]
        x = now_data[['user_id',col]].values
        if len(gp2_count) !=0:
            k = gp2_count.loc[[tuple(i) for i in x]]
        else:
            k = pd.Series(np.full(len(x),np.nan))
        print(k.count()/len(x))
        col_str = 'user_visit_'+col+'#-2'
        now_df[col_str] = k.values
        select_df = pd.concat([select_df,now_df],axis=0)

    return select_df
# In[]
def gen_hour_visit(hour):
    print(hour)
    filter_data = all_data.loc[all_data.hour_global==hour,['user_id','second_cate','shop_id','item_id','minute_global']]
    select_df = pd.DataFrame()
    
    start_minute = filter_data.minute_global.min()
    end_minute = filter_data.minute_global.max()
    
    #1hour ago
    step = 20
    hours_df = pd.DataFrame()
    for minute in range(start_minute,end_minute,step):
        start = minute - 60
        end = minute
        before_1hour = all_data.loc[(all_data.minute_global>=start)&(all_data.minute_global<end),['user_id','second_cate','shop_id','item_id']]
        fill_data = filter_data.loc[(filter_data.minute_global>=minute)&(filter_data.minute_global<(minute+step)),['user_id','second_cate','shop_id','item_id']]
        cols = ['second_cate','shop_id','item_id']
        cols_df = pd.DataFrame()
        for col in cols:
            gp = before_1hour.groupby(['user_id',col])
            gp_count = gp.count().iloc[:,0]
            x = fill_data[['user_id',col]].values
            if len(gp_count) !=0:
                k = gp_count.loc[[tuple(i) for i in x]]
            else:
                k = pd.Series(np.full(len(x),np.nan))
            col_str = 'user_visit_'+col+'#-1hour'
            cols_df[col_str] = k.values
        hours_df = pd.concat([hours_df, cols_df],axis=0)
    #30minute ago
    step = 10
    min30_df = pd.DataFrame()
    for minute in range(start_minute,end_minute,step):
        start = minute - 30
        end = minute
        before_30min = all_data.loc[(all_data.minute_global>=start)&(all_data.minute_global<end),['user_id','second_cate','shop_id','item_id']]
        fill_data = filter_data.loc[(filter_data.minute_global>=minute)&(filter_data.minute_global<(minute+step)),['user_id','second_cate','shop_id','item_id']]
        cols = ['second_cate','shop_id','item_id']
        cols_df = pd.DataFrame()
        for col in cols:
            gp = before_30min.groupby(['user_id',col])
            gp_count = gp.count().iloc[:,0]
            x = fill_data[['user_id',col]].values
            if len(gp_count) !=0:
                k = gp_count.loc[[tuple(i) for i in x]]
            else:
                k = pd.Series(np.full(len(x),np.nan))
            col_str = 'user_visit_'+col+'#-30min'
            cols_df[col_str] = k.values
        min30_df = pd.concat([min30_df, cols_df],axis=0)
    #15minute ago
    step = 5
    min15_df = pd.DataFrame()
    for minute in range(start_minute,end_minute,step):
        start = minute - 15
        end = minute
        before_15min = all_data.loc[(all_data.minute_global>=start)&(all_data.minute_global<end),['user_id','second_cate','shop_id','item_id']]
        fill_data = filter_data.loc[(filter_data.minute_global>=minute)&(filter_data.minute_global<(minute+step)),['user_id','second_cate','shop_id','item_id']]
        cols = ['second_cate','shop_id','item_id']
        cols_df = pd.DataFrame()
        for col in cols:
            gp = before_15min.groupby(['user_id',col])
            gp_count = gp.count().iloc[:,0]
            x = fill_data[['user_id',col]].values
            if len(gp_count) !=0:
                k = gp_count.loc[[tuple(i) for i in x]]
            else:
                k = pd.Series(np.full(len(x),np.nan))
            col_str = 'user_visit_'+col+'#-15min'
            cols_df[col_str] = k.values
        min15_df = pd.concat([min15_df, cols_df],axis=0)
    hours_df.reset_index(inplace=True,drop=True)
    min30_df.reset_index(inplace=True,drop=True)
    min15_df.reset_index(inplace=True,drop=True)
    select_df = pd.concat([hours_df,min30_df,min15_df],axis=1)
    return select_df

def gen_2hour_ago_visit(gp_col):
    col = gp_col
    
    filter_data = all_data.loc[:,['user_id','second_cate','shop_id','item_id','minute_global']]
    
    start_minute = filter_data.minute_global.min()
    end_minute = filter_data.minute_global.max()
    
    #1hour ago
    step = 40
    hours_df = pd.DataFrame()
    for minute in range(start_minute,end_minute,step):
        start = minute - 120
        end = minute
        before_1hour = all_data.loc[(all_data.minute_global>=start)&(all_data.minute_global<end),['user_id','second_cate','shop_id','item_id','minute_global']]
        fill_data = filter_data.loc[(filter_data.minute_global>=minute)&(filter_data.minute_global<(minute+step)),['user_id','second_cate','shop_id','item_id','minute_global']]
        cols_df = pd.DataFrame()
        gp = before_1hour.groupby(['user_id',col])
        gp_count = gp.count().iloc[:,0]
        x = fill_data[['user_id',col]].values
        if len(gp_count) !=0:
            k = gp_count.loc[[tuple(i) for i in x]]
        else:
            k = pd.Series(np.full(len(x),np.nan))
        col_str = 'user_visit_'+col+'#-2hour'
        cols_df[col_str] = k.values
        hours_df = pd.concat([hours_df, cols_df],axis=0)
    return hours_df
def gen_1hour_ago_visit(gp_col):
    col = gp_col
    
    filter_data = all_data.loc[:,['user_id','second_cate','shop_id','item_id','minute_global']]
    
    start_minute = filter_data.minute_global.min()
    end_minute = filter_data.minute_global.max()
    
    #1hour ago
    step = 20
    hours_df = pd.DataFrame()
    for minute in range(start_minute,end_minute,step):
        start = minute - 60
        end = minute
        before_1hour = all_data.loc[(all_data.minute_global>=start)&(all_data.minute_global<end),['user_id','second_cate','shop_id','item_id','minute_global']]
        fill_data = filter_data.loc[(filter_data.minute_global>=minute)&(filter_data.minute_global<(minute+step)),['user_id','second_cate','shop_id','item_id','minute_global']]
        cols_df = pd.DataFrame()
        gp = before_1hour.groupby(['user_id',col])
        gp_count = gp.count().iloc[:,0]
        x = fill_data[['user_id',col]].values
        if len(gp_count) !=0:
            k = gp_count.loc[[tuple(i) for i in x]]
        else:
            k = pd.Series(np.full(len(x),np.nan))
        col_str = 'user_visit_'+col+'#-1hour'
        cols_df[col_str] = k.values
        hours_df = pd.concat([hours_df, cols_df],axis=0)
    return hours_df

def gen_30min_ago_visit(gp_col):
    col = gp_col
    
    filter_data = all_data.loc[:,['user_id','second_cate','shop_id','item_id','minute_global']]
    
    start_minute = filter_data.minute_global.min()
    end_minute = filter_data.minute_global.max()
    
    #1hour ago
    step = 10
    hours_df = pd.DataFrame()
    for minute in range(start_minute,end_minute,step):
        start = minute - 30
        end = minute
        before_1hour = all_data.loc[(all_data.minute_global>=start)&(all_data.minute_global<end),['user_id','second_cate','shop_id','item_id','minute_global']]
        fill_data = filter_data.loc[(filter_data.minute_global>=minute)&(filter_data.minute_global<(minute+step)),['user_id','second_cate','shop_id','item_id','minute_global']]
        cols_df = pd.DataFrame()
        gp = before_1hour.groupby(['user_id',col])
        gp_count = gp.count().iloc[:,0]
        x = fill_data[['user_id',col]].values
        if len(gp_count) !=0:
            k = gp_count.loc[[tuple(i) for i in x]]
        else:
            k = pd.Series(np.full(len(x),np.nan))
        col_str = 'user_visit_'+col+'#-30minute'
        cols_df[col_str] = k.values
        hours_df = pd.concat([hours_df, cols_df],axis=0)
    return hours_df
def gen_15min_ago_visit(gp_col):
    col = gp_col
    
    filter_data = all_data.loc[:,['user_id','second_cate','shop_id','item_id','minute_global']]
    
    start_minute = filter_data.minute_global.min()
    end_minute = filter_data.minute_global.max()
    
    #1hour ago
    step = 5
    hours_df = pd.DataFrame()
    for minute in range(start_minute,end_minute,step):
        start = minute - 15
        end = minute
        before_1hour = all_data.loc[(all_data.minute_global>=start)&(all_data.minute_global<end),['user_id','second_cate','shop_id','item_id','minute_global']]
        fill_data = filter_data.loc[(filter_data.minute_global>=minute)&(filter_data.minute_global<(minute+step)),['user_id','second_cate','shop_id','item_id','minute_global']]
        cols_df = pd.DataFrame()
        gp = before_1hour.groupby(['user_id',col])
        gp_count = gp.count().iloc[:,0]
        x = fill_data[['user_id',col]].values
        if len(gp_count) !=0:
            k = gp_count.loc[[tuple(i) for i in x]]
        else:
            k = pd.Series(np.full(len(x),np.nan))
        col_str = 'user_visit_'+col+'#-15minute'
        cols_df[col_str] = k.values
        hours_df = pd.concat([hours_df, cols_df],axis=0)
    return hours_df
# In[]
if __name__ == '__main__':
    all_data,all_data_list = load_splited_df()
    
    
    
    data_in_process = [('second_cate'),('shop_id'),('item_id')]
    with multiprocessing.Pool(3) as p:
        k_val_list = p.map(gen_day_visit,data_in_process)
    day_visit = pd.concat(k_val_list,axis=1)
    
    all_data['minute_global'] = all_data['hour']*60 + all_data['minute_col']
    all_data['hour_global'] = all_data['day']*24 + all_data['hour']
    
    with multiprocessing.Pool(3) as p:
        k_val_list = p.map(gen_2hour_ago_visit,data_in_process)
    hour2_visit = pd.concat(k_val_list,axis=1)
    
    with multiprocessing.Pool(3) as p:
        k_val_list = p.map(gen_1hour_ago_visit,data_in_process)
    hour1_visit = pd.concat(k_val_list,axis=1)
    
    with multiprocessing.Pool(3) as p:
        k_val_list = p.map(gen_30min_ago_visit,data_in_process)
    min30_visit = pd.concat(k_val_list,axis=1)
    
    with multiprocessing.Pool(3) as p:
        k_val_list = p.map(gen_15min_ago_visit,data_in_process)
    min15_visit = pd.concat(k_val_list,axis=1)
    
    day_visit.reset_index(inplace=True,drop=True)
    hour2_visit.reset_index(inplace=True,drop=True)
    hour1_visit.reset_index(inplace=True,drop=True)
    min30_visit.reset_index(inplace=True,drop=True)
    min15_visit.reset_index(inplace=True,drop=True)
    visit_df = pd.concat([day_visit,hour2_visit,hour1_visit,min30_visit,min15_visit],axis=1)
    split_and_save(visit_df,feature_data_path+'/visit/')
#    save_path = (visit_df, cache_pkl_path+'day7_visit')
#    _save_splited_df(save_path)