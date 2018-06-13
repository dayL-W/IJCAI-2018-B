# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:02:51 2018

@author: Liaowei
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

'''
生成一些基本特征：
1、用户、商品、店铺的基本信息
2、总值统计：用户、商品、品牌、类目、店铺买入总数
'''
# In[ ]: 获取用户的基本信息，包括：
#id、性别、年龄、职业、星级编号，买入总数，职业是否需要做one-hot?
def gen_user_basic_info(in_out_path_set):
    in_path,out_path = in_out_path_set
    df = _load_splited_df(in_path)
    selected_df = pd.DataFrame()
    selected_df['user_id'] = df['user_id']
    selected_df['user_gender_id'] = df['user_gender_id']
    selected_df['user_age_level'] = df['user_age_level']
    selected_df['user_occupation_id'] = df['user_occupation_id']
    selected_df['user_star_level'] = df['user_star_level']
    
    #用户搜索时间划分，上午/下午/晚上/凌晨
    selected_df['is_morning'] = (df['hour'].values>=8) & (df['hour'].values<=12)
    selected_df['is_afternoon'] = (df['hour'].values>12) & (df['hour'].values<=17)
    selected_df['is_evening'] = (df['hour'].values>17) & (df['hour'].values<=23)
    selected_df['is_before_dawn'] = (df['hour'].values<8)
    selected_df['moring_12hour'] = (df['hour'].values<12)
    selected_df['after_12hour'] = (df['hour'].values>=12)
    selected_df['is_trade'] = df['is_trade']
    
    _save_splited_df((selected_df,out_path))

# In[]:获取商品的基本特征
def gen_item_basic_info(in_out_path_set):
    '''
    商品id, 类目、品牌、属性先不考虑、城市、所在展示的页数
    价格等级、销量等级、展示次数、收藏次数
    商品买入总数、品牌买入总数、类目买入总数
    '''
    in_path,out_path = in_out_path_set
    df = _load_splited_df(in_path)
    
    selected_df = pd.DataFrame()
    
    selected_df['item_id'] = df['item_id']
    selected_df['second_cate'] = df['second_cate']
    selected_df['item_brand_id'] = df['item_brand_id']
    selected_df['item_city_id'] = df['item_city_id']
    selected_df['item_price_level'] = df['item_price_level']
    selected_df['item_sales_level'] = df['item_sales_level']
    selected_df['item_pv_level'] = df['item_pv_level']
    selected_df['item_collected_level'] = df['item_collected_level']
    selected_df['context_page_id'] = df['context_page_id']
    selected_df['day'] = df['day']
    selected_df['hour'] = df['hour']
    
    _save_splited_df((selected_df,out_path))
     
# In[]:生成店铺的基本特征
def gen_shop_basic_features(in_out_path_set):
    '''
    店铺id、评价数量、好评率、星级、服务态度、物流服务、描述相符等级
    店铺买入总数
    '''    
    in_path,out_path = in_out_path_set
    df = _load_splited_df(in_path)
    
    selected_df = pd.DataFrame()
    
    selected_df['shop_id'] = df['shop_id']
    selected_df['shop_review_num_level'] = df['shop_review_num_level']
    selected_df['shop_review_positive_rate'] = df['shop_review_positive_rate']
    selected_df['shop_star_level'] = df['shop_star_level']
    selected_df['shop_score_service'] = df['shop_score_service']
    selected_df['shop_score_delivery'] = df['shop_score_delivery']
    selected_df['shop_score_description'] = df['shop_score_description']
    
    _save_splited_df((selected_df,out_path))
# In[]
def gen_buy_visit_cvr_count(df):
    cols = ['user_id','item_id','item_brand_id','second_cate','shop_id']
    selected_df = pd.DataFrame()
    for col in cols:
        buy_str = col + '_count#buy'
        visit_str = col + '_count#visit'
        cvr_str = col + '_count#cvr'
        buy_all = None
        visit_all = None
        for day in tqdm(df.day.unique()):
            day_filter = df.loc[df.day < day, [col,'is_trade']]
            day_gp = day_filter.groupby([col])
            col_buy_count = day_gp.sum().iloc[:,0]
            col_visit_count = day_gp.count().iloc[:,0]
            
            today_buy = df.loc[df.day == day, [col]]
            today_buy[buy_str] = today_buy.apply(lambda x: \
              col_buy_count[x[col]] if x[col] in col_buy_count.index else -1, axis=1)
            buy_all = pd.concat([buy_all,today_buy], axis=0)
            
            today_visit = df.loc[df.day == day, [col]]
            today_visit[visit_str] = today_visit.apply(lambda x: \
              col_visit_count[x[col]] if x[col] in col_visit_count.index else np.nan, axis=1)
            visit_all = pd.concat([visit_all,today_visit], axis=0)
            
        selected_df[buy_str] = buy_all[buy_str]
        selected_df[visit_str] = visit_all[visit_str]
        selected_df[cvr_str] = selected_df[buy_str]/selected_df[visit_str]
    return selected_df
# In[]
if __name__ == '__main__':
    
    splited_apply(gen_user_basic_info,raw_data_path+'splited/',feature_data_path+'user_basic_info/')
    splited_apply(gen_item_basic_info,raw_data_path+'splited/',feature_data_path+'item_basic_info/')
    splited_apply(gen_shop_basic_features,raw_data_path+'splited/',feature_data_path+'shop_basic_info/')

    all_data,all_data_list = load_splited_df()
    print('test hour:',list(set(all_data.hour)))
    selected_df = gen_buy_visit_cvr_count(all_data)
    split_and_save(selected_df, feature_data_path+'buy_visit_cvr/')