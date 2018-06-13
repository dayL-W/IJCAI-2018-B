# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:08:54 2018

@author: Liaowei
"""

import pandas as pd
import numpy as np
import time
import datetime
import os

from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,split_and_save,load_splited_df
#from smooth import BayesianSmoothing
from tqdm import tqdm
import multiprocessing
'''
生成时间聚合特征和时间差特征
'''
#%% 
def _calc_user_search(df_cols):
    df,cols = df_cols
    user_day_search = df.groupby(cols).count().iloc[:,0]
    #获取每个样本的,user_id,day组成的索引，以索引聚类后的数据
    x = df.loc[:, tuple(cols)].values
    k = user_day_search.loc[[tuple(i) for i in x]]
    return k.values
# In[ ]: 用户搜索次数的特征，包括：
#当日搜索特征、目前小时搜索次数
def gen_user_search_count(df):
    
    df = df.loc[:,['user_id', 'item_id', 'shop_id','day', 'hour', 'second_cate']]
    
    data_select = pd.DataFrame()
    feature_cols = ['user_day_search',
                    'user_hour_search',
                    'user_day_item_search',
                    'user_hour_item_search',
                    'user_day_shop_search',
                    'user_hour_shop_search',
                    'user_day_cate_search',
                    'user_hour_cate_search']
    cols = [['user_id', 'day'],
            ['user_id', 'day', 'hour'],
            ['user_id', 'day', 'item_id'],
            ['user_id', 'day', 'hour', 'item_id'],
            ['user_id', 'day', 'shop_id'],
            ['user_id', 'day', 'hour', 'shop_id'],
            ['user_id', 'day', 'second_cate'],
            ['user_id', 'day', 'hour', 'second_cate'],
            ]
    df_cols = [(df,col) for col in cols]
    with multiprocessing.Pool(8) as p:
        k_val_list = p.map(_calc_user_search,df_cols)
        
    for feature,val in zip(feature_cols,k_val_list):
        data_select[feature] = val
    return data_select
    #dump_pickle(data_select, feature_data_path +file_name + '_user_search_count')
    
    
# In[]:生成用户的时间差特征：
def gen_user_search_time(df):
    '''
    #用当次搜索距离当天第一次搜索该商品时间差
    #用当次搜索距离当天第最后一次搜索该商品时间差
    #用当次搜索距离当天第一次搜索该店铺时间差
    #用当次搜索距离当天第最后一次搜索该店铺时间差
    #用当次搜索距离当天第一次搜索该品牌时间差
    #用当次搜索距离当天第最后一次搜索该品牌时间差
    #用当次搜索距离当天第一次搜索该类目时间差
    #用当次搜索距离当天第最后一次搜索该类目时间差
    '''
    data_select = pd.DataFrame()
    
    cols = ['item_id','shop_id', 'item_brand_id','second_cate']
    for col in tqdm(cols):
        data_filter = df[['user_id', col,'day','context_timestamp']].groupby(['user_id', col,'day'])
        max_time = data_filter.agg(max)
        min_time = data_filter.agg(min)
        x = df.loc[:, ('user_id', col, 'day')].values
        m = max_time.loc[[tuple(i) for i in x]]
        n = min_time.loc[[tuple(i) for i in x]]
        data_select['sub_maxtime_'+col] = df['context_timestamp'].values - np.squeeze(m.values)
        data_select['sub_mintime_'+col] = df['context_timestamp'].values - np.squeeze(n.values)
        
        data_select['sub_maxtime_'+col] = data_select['sub_maxtime_'+col].apply(lambda x: x.total_seconds())
        data_select['sub_mintime_'+col] = data_select['sub_mintime_'+col].apply(lambda x: x.total_seconds())
    return data_select
    #dump_pickle(data_select, feature_data_path +file_name + '_user_search_time')
    

# In[]
if __name__ == '__main__':
    all_data,all_data_list = load_splited_df()
    user_search_df = gen_user_search_time(all_data)
    split_and_save(user_search_df,feature_data_path+'user_search_time/')
    
    user_count_df = gen_user_search_count(all_data)
    split_and_save(user_count_df,feature_data_path+'user_search_count/')