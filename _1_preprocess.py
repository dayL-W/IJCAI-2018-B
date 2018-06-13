# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:46:11 2018

@author: Liaowei
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:46:11 2018

@author: Liaowei
"""

import pandas as pd
import numpy as np
import time
import datetime
from datetime import datetime
import os
from tqdm import tqdm
import multiprocessing
from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,load_splited_df,split_and_save
from utils import _load_splited_df, _save_splited_df,splited_apply

train_file = 'round2_train.txt'
test_file = 'round2_ijcai_18_test_b_20180510.txt'

# In[ ]:把训练数据和测试数据换成统一的Index

def gen_global_index():
    train = pd.read_table(raw_data_path + train_file,delim_whitespace=True)
    test = pd.read_table(raw_data_path + test_file,delim_whitespace=True)
    test['is_trade'] = -1
    all_data = train.append(test)
    all_data['old_index'] = all_data.index
    all_data.index = np.arange(0,all_data.shape[0])
    return all_data
    
# In[ ]: 对时间戳做处理并提取日期和小时

def gen_day_hour(df):
    df.context_timestamp = df.context_timestamp.apply(datetime.fromtimestamp)
    df['day'] = df.context_timestamp.apply(lambda x:x.date().day)
    df['hour'] = df.context_timestamp.apply(lambda x:x.time().hour)
    df['minute_col'] = df.context_timestamp.apply(lambda x:x.time().minute)
    return df
    #dump_pickle(df, path=raw_data_path + file_name + '.pkl')
    
    
# In[]:获得第二个类别
def gen_category(df):
    item_cate_col = list(df.item_category_list)
    item_cate = list(map(lambda x: x.split(';'), item_cate_col))
    df['second_cate'] = list(map(lambda x: x[1], item_cate))
    return df

def addCate(df):
    def calSecondCate(x):
        return x['item_category_list'].split(';')[1]
    def calThirdCate(x):
        if len(x['item_category_list'].split(';')) < 3:
            return -1
        return x['item_category_list'].split(';')[2]
    df['second_cate'] = df.apply(lambda x: calSecondCate(x), axis=1)
    df['third_cate'] = df.apply(lambda x: calThirdCate(x), axis=1)
    return df
    
# In[]
def gen_sorted_search_property(init_train):
    '''
    统计所有属性的个数并排序
    '''
    category_property_col = list(init_train.predict_category_property)
    prop_cnt = dict()
    for row in tqdm(category_property_col):
        categories = row.split(';')
        for cate in categories:
            if len(cate.split(':')) < 2:
                continue
            cate_name = int(cate.split(':')[0])
            cate_value = list(map(lambda x: int(x), cate.split(':')[1].split(',')))  
            for prop in cate_value:
                prop_col = str(prop)
                if prop_col in prop_cnt.keys():
                    prop_cnt[prop_col] += 1
                else:
                    prop_cnt[prop_col] = 0
    #return cate_prop_cnt
    return sorted(prop_cnt.items(), key=lambda x: x[1], reverse=True)

# In[]:对类别和属性的统计
def search_category_explore(init_train):
    category_property_col = list(init_train.predict_category_property)
    '''
    1、存放类别下的所有属性
    2、统计所有类别的个数
    3、存放属性对应的所有类别
    4、统计所有属性的个数
    '''
    cate_dict = dict()
    cate_cnt = dict()
    property_dict = dict()
    property_cnt = dict()
    for row in tqdm(category_property_col):
        categories = row.split(';')
        for cate in categories:
            try:
                cate_name = int(cate.split(':')[0])
                cate_value = list(map(lambda x: int(x), cate.split(':')[1].split(',')))
                #print('cate_value', list(cate_value))
                for prop in cate_value:
                    if prop in property_dict.keys():
                        property_dict[prop].add(cate_name)
                        property_cnt[prop] += 1
                    else:
                        property_dict[prop] = set([cate_name])                        
                        property_cnt[prop] = 1
                
                if cate_name in cate_dict.keys():
                    cate_dict[cate_name].update(cate_value)
                    cate_cnt[cate_name] += 1
                else:
                    cate_dict[cate_name] = set(cate_value)
                    cate_cnt[cate_name] = 1
            except:
                print("cate", cate)

    return cate_dict, cate_cnt, property_dict, property_cnt

# In[]:
def gen_sorted_search_cate_property(init_train):
    '''
    统计类别和属性组合后的个数并排序
    '''
    category_property_col = list(init_train.predict_category_property)
    cate_prop_cnt = dict()
    for row in tqdm(category_property_col):
        categories = row.split(';')
        for cate in categories:
            if len(cate.split(':')) < 2:
                continue
            cate_name = int(cate.split(':')[0])
            cate_value = list(map(lambda x: int(x), cate.split(':')[1].split(',')))  
            for prop in cate_value:
                cate_prop_col = str(cate_name)+'_'+str(prop)
                if cate_prop_col in cate_prop_cnt.keys():
                    cate_prop_cnt[cate_prop_col] += 1
                else:
                    cate_prop_cnt[cate_prop_col] = 0
    #return cate_prop_cnt
    return sorted(cate_prop_cnt.items(), key=lambda x: x[1], reverse=True)
# %%
def add_all(in_out_path_set):
    in_path,out_path = in_out_path_set
    df = _load_splited_df(in_path)
    df = gen_day_hour(df)
    df = addCate(df)
    df.day[df.day==31] = 0
    _save_splited_df((df,out_path))
# In[]
if __name__ == '__main__':
    all_data = gen_global_index()
    print('test:',len(all_data.loc[all_data.is_trade==-1,:]))
    split_and_save(all_data)
    splited_apply(add_all,raw_data_path+'splited/',raw_data_path+'splited/')
    
    '''
    print('train shape:',train.shape)
    print('test sha[e:',test.shape)
    print('转化率:',sum(train.is_trade.values)/train.shape[0])
    
    print('train day :',list(set(train.day.values)))
    print('test day :',list(set(test.day.values)))
    
    #把31号改成0
    train.loc[train.day==31,'day'] = 0
    
    print('train 7day hour:',list(set(train.loc[train.day==7, 'hour'].values)))
    print('test 7day hour:',list(set(test.loc[test.day==7, 'hour'].values)))
    days = list(set(train.day.values))
    for day in days:
        day_len = len(train.loc[train.day==day])
        day_trade_sum = sum(train.loc[train.day==day]['is_trade'])
        print('训练集第%d天: %d %f' % (day, day_len,day_trade_sum/day_len))
        
    days = list(set(test.day.values))
    for day in days:
        day_len = len(test.loc[test.day==day])
        print('测试集第%d天: %d %f' % (day, day_len))
    train_user = set(train.user_id.values)
    test_user = set(test.user_id.values)
    print('用户数:',len(train_user),len(test_user),len(train_user&test_user)/len(test_user))
    
    train_item = set(train.item_id.values)
    test_item = set(test.item_id .values)
    print('商品数:',len(train_item),len(test_item),len(train_item&test_item)/len(test_item))
    
    train_shop = set(train.shop_id.values)
    test_shop = set(test.shop_id .values)
    print('店铺数:',len(train_shop),len(test_shop),len(train_shop&test_shop)/len(test_shop))
    
    dump_pickle(train,raw_data_path+'train.pkl')
    dump_pickle(test,raw_data_path+'test.pkl')
    
    train.drop(['item_property_list', 'predict_category_property'], axis=1, inplace=True)
    test.drop(['item_property_list', 'predict_category_property'], axis=1, inplace=True)
    
    dump_pickle(train,raw_data_path+'train_drop_pro.pkl')
    dump_pickle(test,raw_data_path+'test_drop_pro.pkl')
    '''
    if not os.path.exists(feature_data_path):
        os.mkdir(feature_data_path)
    if not os.path.exists(cache_pkl_path):
        os.mkdir(cache_pkl_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)