# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:10:36 2018

@author: Liaowei
"""

from _1_preprocess import search_category_explore, gen_sorted_search_cate_property
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,load_splited_df
from utils import _load_splited_df,_save_splited_df,splited_apply,split_and_save
import multiprocessing
# In[]
def get_context_cate_cols(cate_dict, cate_cnt):
    '''
    把根据排序后的类别和属性拼接起来
    '''
    print('generating context cate cols...')
    cols = []
    sorted_cate_items = sorted(cate_dict.items(), key=lambda x: cate_cnt[x[0]], reverse=True)
    for cate in sorted_cate_items:
        cate_cols = list(map(lambda x: str(cate[0])+'_'+str(x), list(cate[1])))
        cols += cate_cols
    return cols
# In[]
def str_to_cate_cols(cate_str):
    '''
    把类别和属性拼接
    '''
    cate_list = cate_str.split(';')
    cate_cols = []
    for cate in cate_list:
        if len(cate.split(':')) < 2:
            continue
        cate_name = cate.split(':')[0]
        cate_value = cate.split(':')[1].split(',')
        cate_cols+=list(map(lambda x: cate_name+'_'+x, cate_value))
    return cate_cols
# In[]
def cal_query_item_second_cate_sim(x):
    '''
    看预测的类别名称是否和商品的类别相匹配
    '''
    second_item_cate = x['second_cate']
    third_item_cate = second_item_cate
    if x['third_cate'] != -1:
        third_item_cate = x['third_cate']
    #获取到预测的类别名称
    query_cate = list(map(lambda cate_str: cate_str.split(':')[0], x['predict_category_property'].split(';')))
    find_index = np.nan
    for idx, cate in enumerate(query_cate):
        if second_item_cate == cate or third_item_cate == cate:
            find_index = idx
            break
    return find_index
# In[]
def cal_query_item_prop_sim(x):
    '''
    预测类别和商品类别不符合时返回-1
    否则返回预测类别属性和商品属性符合的个数
    '''
    if x['query_item_second_cate_sim'] == np.nan:
        return np.nan
    second_item_cate = x['second_cate']
    third_item_cate = second_item_cate
    if x['third_cate'] != -1:
        third_item_cate = x['third_cate']
    querry_pro = x['predict_category_property'].split(';')
    if querry_pro[0] == '-1':
        return 0
    query_cate = list(map(lambda cate_str: cate_str.split(':')[0], querry_pro))    
    query_property = list(map(lambda cate_str: cate_str.split(':')[1], querry_pro)) 
    query_cate_property = dict(zip(query_cate, query_property))
    hit_property = []
    if second_item_cate in query_cate_property.keys():
        hit_property = query_cate_property[second_item_cate].split(',')
    elif third_item_cate in query_cate_property.keys():
        hit_property = query_cate_property[third_item_cate].split(',')
    item_property = x['item_property_list'].split(';')
    return len(set(hit_property).intersection(set(item_property)))

# In[]
def add_query_item_sim(in_out_path_set):
    '''
    给原始数据增加2列：
    1、是否预测对了商品类别
    2、预测对类别情况下，属性吻合的个数
    '''
    in_path,out_path = in_out_path_set
    df = _load_splited_df(in_path)
    
    df['query_item_second_cate_sim'] = df.apply(lambda x: cal_query_item_second_cate_sim(x), axis=1)
    df['query_item_prop_sim'] = df.apply(lambda x: cal_query_item_prop_sim(x), axis=1)
    
    _save_splited_df((df[['query_item_second_cate_sim','query_item_prop_sim']],out_path))
# In[]
def add_context_cate(data):
    
    #得到类别和属性组合后的个数
    context_cate_cols_path = raw_data_path+'context_cate_cols.pkl'
    if os.path.exists(context_cate_cols_path):
        print("found " + context_cate_cols_path)
        cols = load_pickle(context_cate_cols_path)
        cols = list(map(lambda x: x[0], cols))        
    else:
        #cate_dict, cate_cnt, _, _ = search_category_explore(data)
        cols = gen_sorted_search_cate_property(data)
        cols = list(map(lambda x: x[0], cols))
        dump_pickle(cols, context_cate_cols_path)
    
    
    feature_path = feature_data_path + 'context_cate_property_feat.pkl'
    data.cate_cols = data.predict_category_property.apply(lambda x: str_to_cate_cols(x))
    col_index = 0
    #当前商品的类别和属性拼接后是否在前00名
    for col in tqdm(cols[:300]):
        data[col] = data.cate_cols.apply(lambda x: 1 if col in x else 0)
        #if col_index % 200 == 0 and col_index > 100:
        #    dump_pickle(data[['instance_id']+cols[:col_index+1]], feature_path)            
        col_index+=1    
    dump_pickle(data[['instance_id']+cols[:300]], feature_path)
    return data

# In[] 获取类别属性对的转化率
def gen_sorted_cate_property(init_train):
    """
    获取全部的类目-属性对并排序
    """
    cate_col = list(init_train.item_category_list)
    property_col = list(init_train.item_property_list)
    cate_prop_cnt = dict()    
    for cate, properties in zip(cate_col, property_col):
        second_item_cate = cate.split(';')[1]        
        for prop in properties.split(';'):
            cate_prop_col = second_item_cate+'_'+prop
            if cate_prop_col in cate_prop_cnt.keys():
                cate_prop_cnt[cate_prop_col] += 1
            else:
                cate_prop_cnt[cate_prop_col] = 1
    return sorted(cate_prop_cnt.items(), key=lambda x: x[1], reverse=True)

def gen_cate_property_cvr(test_day, data):
    """
    生成test_day之前全部cate-property对的转化率
    """
    cate_prop_cvr = []
#    print(test_day)
    real_data = data
    real_data = real_data[real_data['day']<test_day]
    trade_data = real_data[real_data['is_trade']==1]
    all_cate_prop_cnt = gen_sorted_cate_property(real_data)
    trade_cate_prop_cnt = gen_sorted_cate_property(trade_data)
    cate_prop_cvr = trade_cate_prop_cnt
    
    #不平滑
    all_cate_prop_cnt = dict(all_cate_prop_cnt)
    for i, cate_prop in enumerate(trade_cate_prop_cnt):
        cate_prop_cvr[i] = [cate_prop[0], 1.0*cate_prop[1]/(all_cate_prop_cnt[cate_prop[0]]+1)]
    return cate_prop_cvr

def gen_cate_property_cvr_stats(x, cate_prop_cvr):
    """
    统计cate-property对的转化率，取最大值，最小值，均值
    """
    second_item_cate = x['item_category_list'].split(';')[1]
    properties = x['item_property_list'].split(';')
    cvr_list = []
    for prop in properties:
        cate_prop_col = second_item_cate+'_'+prop
        if cate_prop_col in cate_prop_cvr.keys():
            cvr_list.append(cate_prop_cvr[cate_prop_col])
    if len(cvr_list) == 0:
        return [np.nan,np.nan,np.nan]
    return [max(cvr_list), min(cvr_list), np.mean(cvr_list)]

def gen_today_cate_pro_cvr(df_day):
    df, day = df_day
    print(day)
    cate_prop_cvr = pd.DataFrame()
    cate_prop_cvr_day = gen_cate_property_cvr(day, df)
    cate_prop_cvr_dict = dict(cate_prop_cvr_day)
    cate_prop_feat = df.loc[df.day==day, ['instance_id', 'item_category_list', 'item_property_list']]
    cate_prop_cvr[['max_cp_cvr','min_cp_cvr', 'mean_cp_cvr']] = cate_prop_feat.apply(lambda \
                 x: gen_cate_property_cvr_stats(x, cate_prop_cvr_dict), axis=1)
    return cate_prop_cvr
def add_cate_property_cvr(df):
    df = df.loc[:,['instance_id','item_id','item_category_list','item_property_list','day', 'hour','second_cate','third_cate','is_trade']]
    
    data_select = pd.DataFrame()
    
    days = list(set(df.day))
    df_days = [(df,day) for day in days]
#    len_day = len(days)
#    with multiprocessing.Pool(len_day-1) as p:
#        k_val_list = p.map(gen_today_cate_pro_cvr,df_day)
#    
#    data_select = pd.concat(k_val_list, axis=0)
    for df_day in df_days:
        cvr_today = gen_today_cate_pro_cvr(df_day)
        data_select = pd.concat([data_select, cvr_today],axis=0)
    data_select.sort_index(inplace=True)
    return data_select
# In[]
if __name__ == '__main__':
    
#    splited_apply(add_query_item_sim,raw_data_path+'splited/',feature_data_path+'query_item_sim/')
    all_data,all_data_list = load_splited_df()
    
    cate_pro_cvr = add_cate_property_cvr(all_data)
    split_and_save(cate_pro_cvr,feature_data_path+'cate_pro_cvr/')
    