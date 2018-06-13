# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:36:13 2018

@author: Liaowei
"""

import pandas as pd
import numpy as np
import time
import datetime
import os
import multiprocessing
from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,load_splited_df,split_and_save
#from smooth import BayesianSmoothing


'''
生成一些相对特征
'''

#%%
def _get_relative_feature(df_col1_col2):
    selected_df,col_1,col_2 = df_col1_col2
    mean = selected_df.groupby([col_1])[col_2].mean()
    k = mean.loc[selected_df[col_1].values]
    return selected_df['item_price_level'].values - np.squeeze(k.values)
    

# In[]: 获取商品的相对特征
def gen_item_relative_feature(df):
    '''
    获取商品相对同类目、品牌平均价格差
    获取商品相对同类目、品牌的平均销量差
    获取商品相对同类目、品牌的平均收藏差
    '''
    #df = load_pickle(path=raw_data_path + file_name + '.pkl')
    selected_df = df.loc[:,['second_cate','item_brand_id','item_price_level','item_sales_level','item_collected_level']]
    item_relative_feature = pd.DataFrame()
    feature_cols = ['cate_relative_price',
                    'cate_relative_sales',
                    'cate_relative_collected',
                    'brand_relative_price',
                    'brand_relative_sales',
                    'brand_relative_collected',
                    ]
    args = [(selected_df,col1,col2) for col1 in ['second_cate','item_brand_id'] for col2 in ['item_price_level','item_sales_level','item_collected_level']]
    with multiprocessing.Pool(8) as p:
        k_val_list = p.map(_get_relative_feature,args)
        
    for feature,val in zip(feature_cols,k_val_list):
        item_relative_feature[feature] = val
    return item_relative_feature
    #dump_pickle(item_relative_feature, feature_data_path +file_name + '_item_relative_feature')
# In[]
if __name__ == '__main__':
    all_data,all_data_list = load_splited_df()
    item_relative_df = gen_item_relative_feature(all_data)
    
    split_and_save(item_relative_df,feature_data_path+'item_relative/')