#!/usr/bin/env python
# -*-coding:utf-8-*-

'''

'''

import math
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import csv
# import matplotlib.dates
# from datetime import *
import datetime
from sklearn.preprocessing import *
from sklearn import ensemble
import xgboost as xgb
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score

from sklearn.preprocessing import *
import xgboost as xgb
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.externals import joblib

# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
# import seaborn as sns
# from datetime import datetime
# from pylab import *
# mpl.rcParams['font.sans-serif'] = ['SimHei']

#定义时间处理函数
def dealDateColumns(df, columnName):
    df[columnName] = pd.to_datetime(df[columnName])
    df['year'] = df[columnName].map(lambda x: int(x.year))
    df['month'] = df[columnName].map(lambda x: int(x.month))
    df['day'] = df[columnName].map(lambda x: int(x.day))
    return df

#首先尝试对每个月构造训练集，并为每条样本打上标签
def getTrainDfMonthly(jdata_user_order_data_target, jdata_user_basic_info_data, month, year, start_time, end_time):
    user_set = set(jdata_user_order_data_target['user_id'][((jdata_user_order_data_target.o_date < end_time) & (jdata_user_order_data_target.o_date >= start_time))])
    jdata_train_df_1 = pd.DataFrame(columns=['user_id', 'age', 'sex', 'user_lv_cd'])
    for user_id, age, sex, user_lv_cd in jdata_user_basic_info_data[['user_id', 'age', 'sex', 'user_lv_cd']].values:
        if user_id in user_set:
            insertRow = pd.DataFrame([[user_id, age, sex, user_lv_cd]], columns=['user_id', 'age', 'sex', 'user_lv_cd'])
            jdata_train_df_1 = jdata_train_df_1.append(insertRow)
    jdata_train_df_1['month'] = int(month)
    jdata_train_df_1['year'] = int(year)
    jdata_train_df_1 = pd.merge(jdata_train_df_1, jdata_user_order_data_target[['user_id', 'year', 'month', 'day', 'o_date']], on=['user_id', 'year', 'month'], how='left')
    jdata_train_df_1 = jdata_train_df_1.sort_index(by=['user_id', 'o_date'], ascending=True)
    jdata_train_df_1 = jdata_train_df_1.drop_duplicates(['user_id'])
    jdata_train_df_1['o_date'] = jdata_train_df_1['o_date'].fillna(pd.to_datetime('2000-01-01'))
    jdata_train_df_1['is_order'] = jdata_train_df_1['day'].map(lambda x: 0 if math.isnan(x) else 1)
    jdata_train_df_1['order_day'] = jdata_train_df_1['day'].map(lambda x: -1 if math.isnan(x) else x)
    del jdata_train_df_1['day']
    del jdata_train_df_1['o_date']
    print(year, month, len(jdata_train_df_1[jdata_train_df_1.is_order == 1]))
#     jdata_train_df = pd.concat([jdata_train_df, jdata_train_df_1])
    return jdata_train_df_1

# 统计某用户在滑窗区间内下单某个类目商品的次数，逐个月进行处理
def getUserOrderNumberMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_order_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_order_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_order_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_order_future.drop_duplicates(['user_id', 'o_id'], inplace=True)
    jdata_user_order_data_order_future_pivot_table = pd.pivot_table(jdata_user_order_data_order_future, index=['user_id'], values=['o_id'], aggfunc=len)
    jdata_user_order_data_order_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_order_future_pivot_table.rename(columns={'o_id':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_order_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserOrderNumber(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserOrderNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_orderNumber')
        jdata_df = getUserOrderNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_orderNumber')
        jdata_df = getUserOrderNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_orderNumber')
        jdata_df = getUserOrderNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_orderNumber')
    return jdata_df

# 统计某用户在滑窗区间内购买某个类目商品的次数，逐个月进行处理
def getUserBuyNumberMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['sku_id'], aggfunc=len)
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'sku_id':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserBuyNumber(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyNumber')
        jdata_df = getUserBuyNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyNumber')
        jdata_df = getUserBuyNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyNumber')
        jdata_df = getUserBuyNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyNumber')
    return jdata_df

# 统计某用户在滑窗区间内购买某个类目商品价格的最小值，逐个月进行处理
def getUserBuyPriceMinMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['price'], aggfunc="min")
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'price':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserBuyPriceMin(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyPriceMinMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyPriceMin')
        jdata_df = getUserBuyPriceMinMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyPriceMin')
        jdata_df = getUserBuyPriceMinMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyPriceMin')
        jdata_df = getUserBuyPriceMinMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyPriceMin')
    return jdata_df

# 统计某用户在滑窗区间内购买某个类目商品价格的最大值，逐个月进行处理
def getUserBuyPriceMaxMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['price'], aggfunc="max")
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'price':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserBuyPriceMax(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyPriceMaxMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyPriceMax')
        jdata_df = getUserBuyPriceMaxMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyPriceMax')
        jdata_df = getUserBuyPriceMaxMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyPriceMax')
        jdata_df = getUserBuyPriceMaxMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyPriceMax')
    return jdata_df

# 统计某用户在滑窗区间内购买某个类目商品价格的均值，逐个月进行处理
def getUserBuyPriceMeanMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['price'], aggfunc='mean')
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'price':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserBuyPriceMean(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyPriceMeanMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyPriceMean')
        jdata_df = getUserBuyPriceMeanMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyPriceMean')
        jdata_df = getUserBuyPriceMeanMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyPriceMean')
        jdata_df = getUserBuyPriceMeanMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyPriceMean')
    return jdata_df

# 统计某用户在滑窗区间内购买某个类目商品价格的总值，逐个月进行处理
def getUserBuyPriceSumMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['price'], aggfunc='sum')
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'price':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserBuyPriceSum(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyPriceSumMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyPriceSum')
        jdata_df = getUserBuyPriceSumMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyPriceSum')
        jdata_df = getUserBuyPriceSumMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyPriceSum')
        jdata_df = getUserBuyPriceSumMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyPriceSum')
    return jdata_df

# 统计某用户在滑窗区间内购买某个类目商品参数一的最小值，逐个月进行处理
def getUserBuyPara1MinMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['para_1'], aggfunc='min')
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'para_1':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserBuyPara1Min(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyPara1MinMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyPara1Min')
        jdata_df = getUserBuyPara1MinMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyPara1Min')
        jdata_df = getUserBuyPara1MinMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyPara1Min')
        jdata_df = getUserBuyPara1MinMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyPara1Min')
    return jdata_df

# 统计某用户在滑窗区间内购买某个类目商品参数一的最大值，逐个月进行处理
def getUserBuyPara1MaxMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['para_1'], aggfunc='max')
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'para_1':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserBuyPara1Max(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyPara1MaxMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyPara1Max')
        jdata_df = getUserBuyPara1MaxMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyPara1Max')
        jdata_df = getUserBuyPara1MaxMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyPara1Max')
        jdata_df = getUserBuyPara1MaxMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyPara1Max')
    return jdata_df

# 统计某用户在滑窗区间内购买某个类目商品参数一的均值，逐个月进行处理
def getUserBuyPara1MeanMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['para_1'], aggfunc='mean')
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'para_1':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserBuyPara1Mean(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyPara1MeanMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyPara1Mean')
        jdata_df = getUserBuyPara1MeanMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyPara1Mean')
        jdata_df = getUserBuyPara1MeanMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyPara1Mean')
        jdata_df = getUserBuyPara1MeanMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyPara1Mean')
    return jdata_df

# 统计某用户在滑窗区间内购买某个类目商品的天数，逐个月进行处理
def getUserBuyDayNumberMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future.drop_duplicates(['user_id', 'o_date'], inplace=True)
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['o_id'], aggfunc=len)

    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'o_id':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserBuyDayNumber(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyDayNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyDayNumber')
        jdata_df = getUserBuyDayNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyDayNumber')
        jdata_df = getUserBuyDayNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyDayNumber')
        jdata_df = getUserBuyDayNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyDayNumber')
    return jdata_df

# 统计某用户在滑窗区间内购买某个类目商品的件数，逐个月进行处理
def getUserBuyCountMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['o_sku_num'], aggfunc='sum')
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'o_sku_num':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserBuyCount(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyCountMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyCount')
        jdata_df = getUserBuyCountMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyCount')
        jdata_df = getUserBuyCountMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyCount')
        jdata_df = getUserBuyCountMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyCount')
    return jdata_df

# 统计某用户在滑窗区间内购买某个类目商品在当月第几天的一些统计特征(min,max,mean)，逐个月进行处理
def getUserBuyDayMinMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['day'], aggfunc='min')
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'day':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserBuyDayMin(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyDayMinMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyDayMin')
        jdata_df = getUserBuyDayMinMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyDayMin')
        jdata_df = getUserBuyDayMinMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyDayMin')
        jdata_df = getUserBuyDayMinMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyDayMin')
    return jdata_df

def getUserBuyDayMaxMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['day'], aggfunc='max')
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'day':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserBuyDayMax(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyDayMaxMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyDayMax')
        jdata_df = getUserBuyDayMaxMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyDayMax')
        jdata_df = getUserBuyDayMaxMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyDayMax')
        jdata_df = getUserBuyDayMaxMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyDayMax')
    return jdata_df

def getUserBuyDayMeanMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['day'], aggfunc='mean')
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'day':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserBuyDayMean(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyDayMeanMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyDayMean')
        jdata_df = getUserBuyDayMeanMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyDayMean')
        jdata_df = getUserBuyDayMeanMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyDayMean')
        jdata_df = getUserBuyDayMeanMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyDayMean')
    return jdata_df

# 统计某用户在滑窗区间内购买某个类目商品的月份数，逐个月进行处理
def getUserBuyMonthNumberMonthly(jdata_df, jdata_user_order_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30))]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & ((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30))]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.o_date >= start_time) & (jdata_user_order_data.o_date < end_time) & (jdata_user_order_data.cate == cate)]
    jdata_user_order_data_buy_future.drop_duplicates(['user_id', 'month'], inplace=True)
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['month'], aggfunc=len)
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'month':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserBuyMonthNumber(jdata_df, jdata_user_order_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBuyMonthNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_buyMonthNumber')
        jdata_df = getUserBuyMonthNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_buyMonthNumber')
        jdata_df = getUserBuyMonthNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_buyMonthNumber')
        jdata_df = getUserBuyMonthNumberMonthly(jdata_df, jdata_user_order_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_buyMonthNumber')
    return jdata_df

# 统计某用户在滑窗区间内对某个类目商品的操作个数，逐个月进行处理
def getUserSkuActionNumberMonthly(jdata_df, jdata_user_action_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate == 101) | (jdata_user_action_data.cate == 30))]
        else:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate != 101) & (jdata_user_action_data.cate != 30))]
    else:
        jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & (jdata_user_action_data.cate == cate)]
    jdata_user_action_data_future.drop_duplicates(['user_id', 'sku_id'], inplace=True)
    jdata_user_action_data_future_pivot_table = pd.pivot_table(jdata_user_action_data_future, index=['user_id'], values=['sku_id'], aggfunc=len)
    jdata_user_action_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_action_data_future_pivot_table.rename(columns={'sku_id':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_action_data_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserSkuActionNumber(jdata_df, jdata_user_action_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserSkuActionNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_skuActionNumber')
        jdata_df = getUserSkuActionNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_skuActionNumber')
        jdata_df = getUserSkuActionNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_skuActionNumber')
        jdata_df = getUserSkuActionNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_skuActionNumber')
    return jdata_df

# 统计某用户在滑窗区间内对某个类目商品的操作天数，逐个月进行处理
def getUserSkuActionDayNumberMonthly(jdata_df, jdata_user_action_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate == 101) | (jdata_user_action_data.cate == 30))]
        else:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate != 101) & (jdata_user_action_data.cate != 30))]
    else:
        jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & (jdata_user_action_data.cate == cate)]
    jdata_user_action_data_future.drop_duplicates(['user_id', 'a_date'], inplace=True)
    jdata_user_action_data_future_pivot_table = pd.pivot_table(jdata_user_action_data_future, index=['user_id'], values=['a_date'], aggfunc=len)
    jdata_user_action_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_action_data_future_pivot_table.rename(columns={'a_date':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_action_data_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserSkuActionDayNumber(jdata_df, jdata_user_action_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserSkuActionDayNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_skuActionDayNumber')
        jdata_df = getUserSkuActionDayNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_skuActionDayNumber')
        jdata_df = getUserSkuActionDayNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_skuActionDayNumber')
        jdata_df = getUserSkuActionDayNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_skuActionDayNumber')
    return jdata_df

# 统计某用户在滑窗区间内对某个类目商品的操作总数，逐个月进行处理
def getUserActionNumberMonthly(jdata_df, jdata_user_action_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate == 101) | (jdata_user_action_data.cate == 30))]
        else:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate != 101) & (jdata_user_action_data.cate != 30))]
    else:
        jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & (jdata_user_action_data.cate == cate)]
    jdata_user_action_data_future_pivot_table = pd.pivot_table(jdata_user_action_data_future, index=['user_id'], values=['a_num'], aggfunc='sum')
    jdata_user_action_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_action_data_future_pivot_table.rename(columns={'a_num':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_action_data_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserActionNumber(jdata_df, jdata_user_action_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserActionNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_actionNumber')
        jdata_df = getUserActionNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_actionNumber')
        jdata_df = getUserActionNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_actionNumber')
        jdata_df = getUserActionNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_actionNumber')
    return jdata_df

# 统计某用户在滑窗区间内对某个类目商品的浏览个数，逐个月进行处理
def getUserSkuBrowseNumberMonthly(jdata_df, jdata_user_action_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate == 101) | (jdata_user_action_data.cate == 30)) & (jdata_user_action_data.a_type == 1)]
        else:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate != 101) & (jdata_user_action_data.cate != 30)) & (jdata_user_action_data.a_type == 1)]
    else:
        jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & (jdata_user_action_data.cate == cate) & (jdata_user_action_data.a_type == 1)]
    jdata_user_action_data_future.drop_duplicates(['user_id', 'sku_id'], inplace=True)
    jdata_user_action_data_future_pivot_table = pd.pivot_table(jdata_user_action_data_future, index=['user_id'], values=['sku_id'], aggfunc=len)
    jdata_user_action_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_action_data_future_pivot_table.rename(columns={'sku_id':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_action_data_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserSkuBrowseNumber(jdata_df, jdata_user_action_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserSkuBrowseNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_skuBrowseNumber')
        jdata_df = getUserSkuBrowseNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_skuBrowseNumber')
        jdata_df = getUserSkuBrowseNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_skuBrowseNumber')
        jdata_df = getUserSkuBrowseNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_skuBrowseNumber')
    return jdata_df

# 统计某用户在滑窗区间内对某个类目商品的浏览天数，逐个月进行处理
def getUserSkuBrowseDayNumberMonthly(jdata_df, jdata_user_action_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate == 101) | (jdata_user_action_data.cate == 30)) & (jdata_user_action_data.a_type == 1)]
        else:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate != 101) & (jdata_user_action_data.cate != 30)) & (jdata_user_action_data.a_type == 1)]
    else:
        jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & (jdata_user_action_data.cate == cate) & (jdata_user_action_data.a_type == 1)]
    jdata_user_action_data_future.drop_duplicates(['user_id', 'a_date'], inplace=True)
    jdata_user_action_data_future_pivot_table = pd.pivot_table(jdata_user_action_data_future, index=['user_id'], values=['a_date'], aggfunc=len)
    jdata_user_action_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_action_data_future_pivot_table.rename(columns={'a_date':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_action_data_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserSkuBrowseDayNumber(jdata_df, jdata_user_action_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserSkuBrowseDayNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_skuBrowseDayNumber')
        jdata_df = getUserSkuBrowseDayNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_skuBrowseDayNumber')
        jdata_df = getUserSkuBrowseDayNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_skuBrowseDayNumber')
        jdata_df = getUserSkuBrowseDayNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_skuBrowseDayNumber')
    return jdata_df

# 统计某用户在滑窗区间内对某个类目商品的浏览总数，逐个月进行处理
def getUserBrowseNumberMonthly(jdata_df, jdata_user_action_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate == 101) | (jdata_user_action_data.cate == 30)) & (jdata_user_action_data.a_type == 1)]
        else:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate != 101) & (jdata_user_action_data.cate != 30)) & (jdata_user_action_data.a_type == 1)]
    else:
        jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & (jdata_user_action_data.cate == cate) & (jdata_user_action_data.a_type == 1)]
    jdata_user_action_data_future_pivot_table = pd.pivot_table(jdata_user_action_data_future, index=['user_id'], values=['a_num'], aggfunc="sum")
    jdata_user_action_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_action_data_future_pivot_table.rename(columns={'a_num':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_action_data_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserBrowseNumber(jdata_df, jdata_user_action_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserBrowseNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_browseNumber')
        jdata_df = getUserBrowseNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_browseNumber')
        jdata_df = getUserBrowseNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_browseNumber')
        jdata_df = getUserBrowseNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_browseNumber')
    return jdata_df

# 统计某用户在滑窗区间内对某个类目商品的关注个数，逐个月进行处理
def getUserSkuFocusNumberMonthly(jdata_df, jdata_user_action_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate == 101) | (jdata_user_action_data.cate == 30)) & (jdata_user_action_data.a_type == 2)]
        else:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate != 101) & (jdata_user_action_data.cate != 30)) & (jdata_user_action_data.a_type == 2)]
    else:
        jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & (jdata_user_action_data.cate == cate) & (jdata_user_action_data.a_type == 2)]
    jdata_user_action_data_future.drop_duplicates(['user_id', 'sku_id'], inplace=True)
    jdata_user_action_data_future_pivot_table = pd.pivot_table(jdata_user_action_data_future, index=['user_id'], values=['sku_id'], aggfunc=len)
    jdata_user_action_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_action_data_future_pivot_table.rename(columns={'sku_id':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_action_data_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserSkuFocusNumber(jdata_df, jdata_user_action_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserSkuFocusNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_skuFocusNumber')
        jdata_df = getUserSkuFocusNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_skuFocusNumber')
        jdata_df = getUserSkuFocusNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_skuFocusNumber')
        jdata_df = getUserSkuFocusNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_skuFocusNumber')
    return jdata_df

# 统计某用户在滑窗区间内对某个类目商品的关注总数，逐个月进行处理
def getUserFocusNumberMonthly(jdata_df, jdata_user_action_data, start_time, end_time, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate == 101) | (jdata_user_action_data.cate == 30)) & (jdata_user_action_data.a_type == 2)]
        else:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & ((jdata_user_action_data.cate != 101) & (jdata_user_action_data.cate != 30)) & (jdata_user_action_data.a_type == 2)]
    else:
        jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date >= start_time) & (jdata_user_action_data.a_date < end_time) & (jdata_user_action_data.cate == cate) & (jdata_user_action_data.a_type == 2)]
    jdata_user_action_data_future_pivot_table = pd.pivot_table(jdata_user_action_data_future, index=['user_id'], values=['a_num'], aggfunc="sum")
    jdata_user_action_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_action_data_future_pivot_table.rename(columns={'a_num':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_action_data_future_pivot_table, on=['user_id'], how='left')
#     print(len(jdata_train_df_dealMonth))
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserFocusNumber(jdata_df, jdata_user_action_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserFocusNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 101, 'user_cate101_last' + str(month) + 'Month_focusNumber')
        jdata_df = getUserFocusNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, 30, 'user_cate30_last' + str(month) + 'Month_focusNumber')
        jdata_df = getUserFocusNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -1, 'user_targetCate_last' + str(month) + 'Month_focusNumber')
        jdata_df = getUserFocusNumberMonthly(jdata_df, jdata_user_action_data, featureMonthBegin, featureMonthEnd, -2, 'user_relatedCate_last' + str(month) + 'Month_focusNumber')
    return jdata_df

# 统计某用户在滑窗区间内的评价次数，逐个月进行处理
def getUserCommentNumberMonthly(jdata_df, jdata_user_comment_score_data, start_time, end_time, score_level, newColName):
    jdata_train_df_dealMonth = jdata_df
    if score_level == 0:
        jdata_user_comment_score_data_future = jdata_user_comment_score_data[(jdata_user_comment_score_data.comment_create_tm >= start_time) & (jdata_user_comment_score_data.comment_create_tm < end_time)]
    else:
        jdata_user_comment_score_data_future = jdata_user_comment_score_data[(jdata_user_comment_score_data.comment_create_tm >= start_time) & (jdata_user_comment_score_data.comment_create_tm < end_time) & (jdata_user_comment_score_data.score_level == score_level)]
    jdata_user_comment_score_data_future_pivot_table = pd.pivot_table(jdata_user_comment_score_data_future, index=['user_id'], values=['score_level'], aggfunc=len)
    jdata_user_comment_score_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_comment_score_data_future_pivot_table.rename(columns={'score_level':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_comment_score_data_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    return jdata_train_df_dealMonth

def getUserCommentNumber(jdata_df, jdata_user_comment_score_data, FeatureMonthList):
    for featureMonthBegin, featureMonthEnd, month in FeatureMonthList:
        jdata_df = getUserCommentNumberMonthly(jdata_df, jdata_user_comment_score_data, featureMonthBegin, featureMonthEnd, 1, 'user_score1_last' + str(month) + 'Month_commentNumber')
        jdata_df = getUserCommentNumberMonthly(jdata_df, jdata_user_comment_score_data, featureMonthBegin, featureMonthEnd, 2, 'user_score2_last' + str(month) + 'Month_commentNumber')
        jdata_df = getUserCommentNumberMonthly(jdata_df, jdata_user_comment_score_data, featureMonthBegin, featureMonthEnd, 3, 'user_score3_last' + str(month) + 'Month_commentNumber')
        jdata_df = getUserCommentNumberMonthly(jdata_df, jdata_user_comment_score_data, featureMonthBegin, featureMonthEnd, 0, 'user_score0_last' + str(month) + 'Month_commentNumber')
    return jdata_df

# 统计某用户平均每个月购买某个类目商品的次数，逐个月进行处理
def getUserMonthlyBuyNumberMonthly(jdata_df, jdata_user_order_data, cate, newColName, time_boundary, length):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30)) & (jdata_user_order_data.o_date < time_boundary)]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30)) & (jdata_user_order_data.o_date < time_boundary)]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.cate == cate) & (jdata_user_order_data.o_date < time_boundary)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['sku_id'], aggfunc=len)
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'sku_id':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName] / length
    return jdata_train_df_dealMonth

def getUserMonthlyBuyNumber(jdata_df, jdata_user_order_data, time_boundary, length):
    jdata_df = getUserMonthlyBuyNumberMonthly(jdata_df, jdata_user_order_data, 101, 'user_101Cate_monthlyBuyNumber', time_boundary, length)
    jdata_df = getUserMonthlyBuyNumberMonthly(jdata_df, jdata_user_order_data, 30, 'user_30Cate_monthlyBuyNumber', time_boundary, length)
    jdata_df = getUserMonthlyBuyNumberMonthly(jdata_df, jdata_user_order_data, -1, 'user_targetCate_monthlyBuyNumber', time_boundary, length)
    jdata_df = getUserMonthlyBuyNumberMonthly(jdata_df, jdata_user_order_data, -2, 'user_relatedCate_monthlyBuyNumber', time_boundary, length)
    return jdata_df

# 统计某用户平均每个月购买某个类目商品的下单次数，逐个月进行处理
def getUserMonthlyOrderNumberMonthly(jdata_df, jdata_user_order_data, cate, newColName, time_boundary, length):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30)) & (jdata_user_order_data.o_date < time_boundary)]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30)) & (jdata_user_order_data.o_date < time_boundary)]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.cate == cate) & (jdata_user_order_data.o_date < time_boundary)]
    jdata_user_order_data_buy_future.drop_duplicates(['user_id', 'o_id'], inplace=True)
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['o_id'], aggfunc=len)
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'o_id':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName] / length
    return jdata_train_df_dealMonth

def getUserMonthlyOrderNumber(jdata_df, jdata_user_order_data, time_boundary, length):
    jdata_df = getUserMonthlyOrderNumberMonthly(jdata_df, jdata_user_order_data, 101, 'user_101Cate_monthlyOrderNumber', time_boundary, length)
    jdata_df = getUserMonthlyOrderNumberMonthly(jdata_df, jdata_user_order_data, 30, 'user_30Cate_monthlyOrderNumber', time_boundary, length)
    jdata_df = getUserMonthlyOrderNumberMonthly(jdata_df, jdata_user_order_data, -1, 'user_targetCate_monthlyOrderNumber', time_boundary, length)
    jdata_df = getUserMonthlyOrderNumberMonthly(jdata_df, jdata_user_order_data, -2, 'user_relatedCate_monthlyOrderNumber', time_boundary, length)
    return jdata_df

# 统计某用户平均每个月购买某个类目商品的件数，逐个月进行处理
def getUserMonthlyBuyCountMonthly(jdata_df, jdata_user_order_data, cate, newColName, time_boundary, length):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30)) & (jdata_user_order_data.o_date < time_boundary)]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30)) & (jdata_user_order_data.o_date < time_boundary)]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.cate == cate) & (jdata_user_order_data.o_date < time_boundary)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['o_sku_num'], aggfunc="sum")
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'o_sku_num':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName] / length
    return jdata_train_df_dealMonth

def getUserMonthlyBuyCount(jdata_df, jdata_user_order_data, time_boundary, length):
    jdata_df = getUserMonthlyBuyCountMonthly(jdata_df, jdata_user_order_data, 101, 'user_101Cate_monthlyBuyCount', time_boundary, length)
    jdata_df = getUserMonthlyBuyCountMonthly(jdata_df, jdata_user_order_data, 30, 'user_30Cate_monthlyBuyCount', time_boundary, length)
    jdata_df = getUserMonthlyBuyCountMonthly(jdata_df, jdata_user_order_data, -1, 'user_targetCate_monthlyBuyCount', time_boundary, length)
    jdata_df = getUserMonthlyBuyCountMonthly(jdata_df, jdata_user_order_data, -2, 'user_relatedCate_monthlyBuyCount', time_boundary, length)
    return jdata_df

# 统计某用户平均每个月购买某个类目商品的天数，逐个月进行处理
def getUserMonthlyBuyDayMonthly(jdata_df, jdata_user_order_data, cate, newColName, time_boundary, length):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30)) & (jdata_user_order_data.o_date < time_boundary)]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30)) & (jdata_user_order_data.o_date < time_boundary)]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.cate == cate) & (jdata_user_order_data.o_date < time_boundary)]
    jdata_user_order_data_buy_future.drop_duplicates(['user_id', 'o_date'], inplace=True)
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['o_date'], aggfunc=len)
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'o_date':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName] / length
    return jdata_train_df_dealMonth

def getUserMonthlyBuyDay(jdata_df, jdata_user_order_data, time_boundary, length):
    jdata_df = getUserMonthlyBuyDayMonthly(jdata_df, jdata_user_order_data, 101, 'user_101Cate_monthlyBuyDay', time_boundary, length)
    jdata_df = getUserMonthlyBuyDayMonthly(jdata_df, jdata_user_order_data, 30, 'user_30Cate_monthlyBuyDay', time_boundary, length)
    jdata_df = getUserMonthlyBuyDayMonthly(jdata_df, jdata_user_order_data, -1, 'user_targetCate_monthlyBuyDay', time_boundary, length)
    jdata_df = getUserMonthlyBuyDayMonthly(jdata_df, jdata_user_order_data, -2, 'user_relatedCate_monthlyBuyDay', time_boundary, length)
    return jdata_df

# 统计某用户历史购买某个类目商品价格的最小值，逐个月进行处理
def getUserMonthlyBuyPriceMinMonthly(jdata_df, jdata_user_order_data, cate, newColName, time_boundary):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30)) & (jdata_user_order_data.o_date < time_boundary)]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30)) & (jdata_user_order_data.o_date < time_boundary)]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.cate == cate) & (jdata_user_order_data.o_date < time_boundary)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['price'], aggfunc="min")
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'price':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserMonthlyBuyPriceMin(jdata_df, jdata_user_order_data, time_boundary):
    jdata_df = getUserMonthlyBuyPriceMinMonthly(jdata_df, jdata_user_order_data, 101, 'user_101Cate_monthlyBuyPriceMin', time_boundary)
    jdata_df = getUserMonthlyBuyPriceMinMonthly(jdata_df, jdata_user_order_data, 30, 'user_30Cate_monthlyBuyPriceMin', time_boundary)
    jdata_df = getUserMonthlyBuyPriceMinMonthly(jdata_df, jdata_user_order_data, -1, 'user_targetCate_monthlyBuyPriceMin', time_boundary)
    jdata_df = getUserMonthlyBuyPriceMinMonthly(jdata_df, jdata_user_order_data, -2, 'user_relatedCate_monthlyBuyPriceMin', time_boundary)
    return jdata_df

# 统计某用户历史购买某个类目商品价格的最大值，逐个月进行处理
def getUserMonthlyBuyPriceMaxMonthly(jdata_df, jdata_user_order_data, cate, newColName, time_boundary):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30)) & (jdata_user_order_data.o_date < time_boundary)]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30)) & (jdata_user_order_data.o_date < time_boundary)]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.cate == cate) & (jdata_user_order_data.o_date < time_boundary)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['price'], aggfunc="max")
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'price':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserMonthlyBuyPriceMax(jdata_df, jdata_user_order_data, time_boundary):
    jdata_df = getUserMonthlyBuyPriceMaxMonthly(jdata_df, jdata_user_order_data, 101, 'user_101Cate_monthlyBuyPriceMax', time_boundary)
    jdata_df = getUserMonthlyBuyPriceMaxMonthly(jdata_df, jdata_user_order_data, 30, 'user_30Cate_monthlyBuyPriceMax', time_boundary)
    jdata_df = getUserMonthlyBuyPriceMaxMonthly(jdata_df, jdata_user_order_data, -1, 'user_targetCate_monthlyBuyPriceMax', time_boundary)
    jdata_df = getUserMonthlyBuyPriceMaxMonthly(jdata_df, jdata_user_order_data, -2, 'user_relatedCate_monthlyBuyPriceMax', time_boundary)
    return jdata_df

# 统计某用户历史购买某个类目商品价格的均值，逐个月进行处理
def getUserMonthlyBuyPriceMeanMonthly(jdata_df, jdata_user_order_data, cate, newColName, time_boundary):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30)) & (jdata_user_order_data.o_date < time_boundary)]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30)) & (jdata_user_order_data.o_date < time_boundary)]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.cate == cate) & (jdata_user_order_data.o_date < time_boundary)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['price'], aggfunc="mean")
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'price':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserMonthlyBuyPriceMean(jdata_df, jdata_user_order_data, time_boundary):
    jdata_df = getUserMonthlyBuyPriceMeanMonthly(jdata_df, jdata_user_order_data, 101, 'user_101Cate_monthlyBuyPriceMean', time_boundary)
    jdata_df = getUserMonthlyBuyPriceMeanMonthly(jdata_df, jdata_user_order_data, 30, 'user_30Cate_monthlyBuyPriceMean', time_boundary)
    jdata_df = getUserMonthlyBuyPriceMeanMonthly(jdata_df, jdata_user_order_data, -1, 'user_targetCate_monthlyBuyPriceMean', time_boundary)
    jdata_df = getUserMonthlyBuyPriceMeanMonthly(jdata_df, jdata_user_order_data, -2, 'user_relatedCate_monthlyBuyPriceMean', time_boundary)
    return jdata_df

# 统计某用户历史购买某个类目商品参数一的最小值，逐个月进行处理
def getUserMonthlyBuyPara1MinMonthly(jdata_df, jdata_user_order_data, cate, newColName, time_boundary):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30)) & (jdata_user_order_data.o_date < time_boundary)]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30)) & (jdata_user_order_data.o_date < time_boundary)]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.cate == cate) & (jdata_user_order_data.o_date < time_boundary)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['para_1'], aggfunc="min")
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'para_1':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserMonthlyBuyPara1Min(jdata_df, jdata_user_order_data, time_boundary):
    jdata_df = getUserMonthlyBuyPara1MinMonthly(jdata_df, jdata_user_order_data, 101, 'user_101Cate_monthlyBuyPara1Min', time_boundary)
    jdata_df = getUserMonthlyBuyPara1MinMonthly(jdata_df, jdata_user_order_data, 30, 'user_30Cate_monthlyBuyPara1Min', time_boundary)
    jdata_df = getUserMonthlyBuyPara1MinMonthly(jdata_df, jdata_user_order_data, -1, 'user_targetCate_monthlyBuyPara1Min', time_boundary)
    jdata_df = getUserMonthlyBuyPara1MinMonthly(jdata_df, jdata_user_order_data, -2, 'user_relatedCate_monthlyBuyPara1Min', time_boundary)
    return jdata_df

# 统计某用户历史购买某个类目商品参数一的最大值，逐个月进行处理
def getUserMonthlyBuyPara1MaxMonthly(jdata_df, jdata_user_order_data, cate, newColName, time_boundary):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30)) & (jdata_user_order_data.o_date < time_boundary)]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30)) & (jdata_user_order_data.o_date < time_boundary)]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.cate == cate) & (jdata_user_order_data.o_date < time_boundary)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['para_1'], aggfunc="max")
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'para_1':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserMonthlyBuyPara1Max(jdata_df, jdata_user_order_data, time_boundary):
    jdata_df = getUserMonthlyBuyPara1MaxMonthly(jdata_df, jdata_user_order_data, 101, 'user_101Cate_monthlyBuyPara1Max', time_boundary)
    jdata_df = getUserMonthlyBuyPara1MaxMonthly(jdata_df, jdata_user_order_data, 30, 'user_30Cate_monthlyBuyPara1Max', time_boundary)
    jdata_df = getUserMonthlyBuyPara1MaxMonthly(jdata_df, jdata_user_order_data, -1, 'user_targetCate_monthlyBuyPara1Max', time_boundary)
    jdata_df = getUserMonthlyBuyPara1MaxMonthly(jdata_df, jdata_user_order_data, -2, 'user_relatedCate_monthlyBuyPara1Max', time_boundary)
    return jdata_df

# 统计某用户历史购买某个类目商品参数一的均值，逐个月进行处理
def getUserMonthlyBuyPara1MeanMonthly(jdata_df, jdata_user_order_data, cate, newColName, time_boundary):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30)) & (jdata_user_order_data.o_date < time_boundary)]
        else:
            jdata_user_order_data_buy_future = jdata_user_order_data[((jdata_user_order_data.cate != 101) & (jdata_user_order_data.cate != 30)) & (jdata_user_order_data.o_date < time_boundary)]
    else:
        jdata_user_order_data_buy_future = jdata_user_order_data[(jdata_user_order_data.cate == cate) & (jdata_user_order_data.o_date < time_boundary)]
    jdata_user_order_data_buy_future_pivot_table = pd.pivot_table(jdata_user_order_data_buy_future, index=['user_id'], values=['para_1'], aggfunc="mean")
    jdata_user_order_data_buy_future_pivot_table.reset_index(inplace=True)
    jdata_user_order_data_buy_future_pivot_table.rename(columns={'para_1':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_order_data_buy_future_pivot_table, on=['user_id'], how='left')
    return jdata_train_df_dealMonth

def getUserMonthlyBuyPara1Mean(jdata_df, jdata_user_order_data, time_boundary):
    jdata_df = getUserMonthlyBuyPara1MeanMonthly(jdata_df, jdata_user_order_data, 101, 'user_101Cate_monthlyBuyPara1Mean', time_boundary)
    jdata_df = getUserMonthlyBuyPara1MeanMonthly(jdata_df, jdata_user_order_data, 30, 'user_30Cate_monthlyBuyPara1Mean', time_boundary)
    jdata_df = getUserMonthlyBuyPara1MeanMonthly(jdata_df, jdata_user_order_data, -1, 'user_targetCate_monthlyBuyPara1Mean', time_boundary)
    jdata_df = getUserMonthlyBuyPara1MeanMonthly(jdata_df, jdata_user_order_data, -2, 'user_relatedCate_monthlyBuyPara1Mean', time_boundary)
    return jdata_df

# 统计某用户平均每个月对某个类目商品的关注总数，逐个月进行处理
def getUserMonthlyFocusNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & ((jdata_user_action_data.cate == 101) | (jdata_user_action_data.cate == 30)) & (jdata_user_action_data.a_type == 2)]
        else:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & ((jdata_user_action_data.cate != 101) & (jdata_user_action_data.cate != 30)) & (jdata_user_action_data.a_type == 2)]
    else:
        jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & (jdata_user_action_data.cate == cate) & (jdata_user_action_data.a_type == 2)]
    jdata_user_action_data_future_pivot_table = pd.pivot_table(jdata_user_action_data_future, index=['user_id'], values=['a_num'], aggfunc="sum")
    jdata_user_action_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_action_data_future_pivot_table.rename(columns={'a_num':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_action_data_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName] / length
    return jdata_train_df_dealMonth

def getUserMonthlyFocusNumber(jdata_df, jdata_user_action_data, time_boundary, length):
    jdata_df = getUserMonthlyFocusNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, 101, 'user_cate101_monthlyFocusNumber')
    jdata_df = getUserMonthlyFocusNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, 30, 'user_cate30_monthlyFocusNumber')
    jdata_df = getUserMonthlyFocusNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, -1, 'user_targetCate_monthlyFocusNumber')
    jdata_df = getUserMonthlyFocusNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, -2, 'user_relatedCate_monthlyFocusNumber')
    return jdata_df

# 统计某用户平均每个月对某个类目商品的浏览总数，逐个月进行处理
def getUserMonthlyBrowseNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & ((jdata_user_action_data.cate == 101) | (jdata_user_action_data.cate == 30)) & (jdata_user_action_data.a_type == 1)]
        else:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & ((jdata_user_action_data.cate != 101) & (jdata_user_action_data.cate != 30)) & (jdata_user_action_data.a_type == 1)]
    else:
        jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & (jdata_user_action_data.cate == cate) & (jdata_user_action_data.a_type == 1)]
    jdata_user_action_data_future_pivot_table = pd.pivot_table(jdata_user_action_data_future, index=['user_id'], values=['a_num'], aggfunc="sum")
    jdata_user_action_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_action_data_future_pivot_table.rename(columns={'a_num':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_action_data_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName] / length
    return jdata_train_df_dealMonth

def getUserMonthlyBrowseNumber(jdata_df, jdata_user_action_data, time_boundary, length):
    jdata_df = getUserMonthlyBrowseNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, 101, 'user_cate101_monthlyBrowseNumber')
    jdata_df = getUserMonthlyBrowseNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, 30, 'user_cate30_monthlyBrowseNumber')
    jdata_df = getUserMonthlyBrowseNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, -1, 'user_targetCate_monthlyBrowseNumber')
    jdata_df = getUserMonthlyBrowseNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, -2, 'user_relatedCate_monthlyBrowseNumber')
    return jdata_df

# 统计某用户平均每个月对某个类目商品的操作总数，逐个月进行处理
def getUserMonthlyActionNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & ((jdata_user_action_data.cate == 101) | (jdata_user_action_data.cate == 30))]
        else:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & ((jdata_user_action_data.cate != 101) & (jdata_user_action_data.cate != 30))]
    else:
        jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & (jdata_user_action_data.cate == cate)]
    jdata_user_action_data_future_pivot_table = pd.pivot_table(jdata_user_action_data_future, index=['user_id'], values=['a_num'], aggfunc="sum")
    jdata_user_action_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_action_data_future_pivot_table.rename(columns={'a_num':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_action_data_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName] / length
    return jdata_train_df_dealMonth

def getUserMonthlyActionNumber(jdata_df, jdata_user_action_data, time_boundary, length):
    jdata_df = getUserMonthlyActionNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, 101, 'user_cate101_monthlyActionNumber')
    jdata_df = getUserMonthlyActionNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, 30, 'user_cate30_monthlyActionNumber')
    jdata_df = getUserMonthlyActionNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, -1, 'user_targetCate_monthlyActionNumber')
    jdata_df = getUserMonthlyActionNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, -2, 'user_relatedCate_monthlyActionNumber')
    return jdata_df

# 统计某用户平均每个月对某个类目商品的操作天数，逐个月进行处理
def getUserMonthlyActionDayNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & ((jdata_user_action_data.cate == 101) | (jdata_user_action_data.cate == 30))]
        else:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & ((jdata_user_action_data.cate != 101) & (jdata_user_action_data.cate != 30))]
    else:
        jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & (jdata_user_action_data.cate == cate)]
    jdata_user_action_data_future.drop_duplicates(['user_id', 'a_date'], inplace=True)
    jdata_user_action_data_future_pivot_table = pd.pivot_table(jdata_user_action_data_future, index=['user_id'], values=['a_date'], aggfunc=len)
    jdata_user_action_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_action_data_future_pivot_table.rename(columns={'a_date':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_action_data_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName] / length
    return jdata_train_df_dealMonth

def getUserMonthlyActionDayNumber(jdata_df, jdata_user_action_data, time_boundary, length):
    jdata_df = getUserMonthlyActionDayNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, 101, 'user_cate101_monthlyActionDayNumber')
    jdata_df = getUserMonthlyActionDayNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, 30, 'user_cate30_monthlyActionDayNumber')
    jdata_df = getUserMonthlyActionDayNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, -1, 'user_targetCate_monthlyActionDayNumber')
    jdata_df = getUserMonthlyActionDayNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, -2, 'user_relatedCate_monthlyActionDayNumber')
    return jdata_df

# 统计某用户平均每个月对某个类目商品的操作商品数，逐个月进行处理
def getUserMonthlyActionSkuNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, cate, newColName):
    jdata_train_df_dealMonth = jdata_df
    if ((cate == -1) | (cate == -2)):
        if cate == -1:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & ((jdata_user_action_data.cate == 101) | (jdata_user_action_data.cate == 30))]
        else:
            jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & ((jdata_user_action_data.cate != 101) & (jdata_user_action_data.cate != 30))]
    else:
        jdata_user_action_data_future = jdata_user_action_data[(jdata_user_action_data.a_date < time_boundary) & (jdata_user_action_data.cate == cate)]
    jdata_user_action_data_future.drop_duplicates(['user_id', 'sku_id'], inplace=True)
    jdata_user_action_data_future_pivot_table = pd.pivot_table(jdata_user_action_data_future, index=['user_id'], values=['sku_id'], aggfunc=len)
    jdata_user_action_data_future_pivot_table.reset_index(inplace=True)
    jdata_user_action_data_future_pivot_table.rename(columns={'sku_id':newColName}, inplace=True)
    jdata_train_df_dealMonth = pd.merge(jdata_train_df_dealMonth, jdata_user_action_data_future_pivot_table, on=['user_id'], how='left')
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName].fillna(0)
    jdata_train_df_dealMonth[newColName] = jdata_train_df_dealMonth[newColName] / length
    return jdata_train_df_dealMonth

def getUserMonthlyActionSkuNumber(jdata_df, jdata_user_action_data, time_boundary, length):
    jdata_df = getUserMonthlyActionSkuNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, 101, 'user_cate101_monthlyActionSkuCount')
    jdata_df = getUserMonthlyActionSkuNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, 30, 'user_cate30_monthlyActionSkuCount')
    jdata_df = getUserMonthlyActionSkuNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, -1, 'user_targetCate_monthlyActionSkuCount')
    jdata_df = getUserMonthlyActionSkuNumberMonthly(jdata_df, jdata_user_action_data, time_boundary, length, -2, 'user_relatedCate_monthlyActionSkuCount')
    return jdata_df

#定义特征工程函数
def featureDeal(jdata_train_df_201708, jdata_train_df_201707, jdata_train_df_201706, jdata_train_df_201705, jdata_train_df_201704, jdata_train_df_201703, jdata_test_df, FeatureMonthList_201708, FeatureMonthList_201707, FeatureMonthList_201706, FeatureMonthList_201705, FeatureMonthList_201704, FeatureMonthList_201703, FeatureMonthList_201709, jdata_user_order_data, jdata_user_action_data, jdata_user_comment_score_data):
    jdata_train_df_201708 = getUserOrderNumber(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserOrderNumber(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserOrderNumber(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserOrderNumber(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserOrderNumber(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserOrderNumber(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserOrderNumber(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyNumber(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyNumber(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyNumber(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyNumber(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyNumber(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyNumber(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyNumber(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyPriceMin(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyPriceMin(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyPriceMin(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyPriceMin(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyPriceMin(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyPriceMin(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyPriceMin(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyPriceMax(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyPriceMax(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyPriceMax(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyPriceMax(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyPriceMax(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyPriceMax(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyPriceMax(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyPriceMean(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyPriceMean(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyPriceMean(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyPriceMean(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyPriceMean(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyPriceMean(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyPriceMean(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyPriceSum(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyPriceSum(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyPriceSum(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyPriceSum(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyPriceSum(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyPriceSum(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyPriceSum(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyPara1Min(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyPara1Min(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyPara1Min(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyPara1Min(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyPara1Min(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyPara1Min(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyPara1Min(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyPara1Max(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyPara1Max(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyPara1Max(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyPara1Max(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyPara1Max(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyPara1Max(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyPara1Max(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyPara1Mean(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyPara1Mean(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyPara1Mean(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyPara1Mean(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyPara1Mean(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyPara1Mean(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyPara1Mean(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyDayNumber(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyDayNumber(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyDayNumber(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyDayNumber(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyDayNumber(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyDayNumber(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyDayNumber(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyCount(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyCount(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyCount(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyCount(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyCount(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyCount(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyCount(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyDayMin(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyDayMin(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyDayMin(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyDayMin(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyDayMin(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyDayMin(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyDayMin(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyDayMax(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyDayMax(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyDayMax(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyDayMax(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyDayMax(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyDayMax(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyDayMax(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyDayMean(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyDayMean(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyDayMean(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyDayMean(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyDayMean(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyDayMean(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyDayMean(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBuyMonthNumber(jdata_train_df_201708, jdata_user_order_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBuyMonthNumber(jdata_train_df_201707, jdata_user_order_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBuyMonthNumber(jdata_train_df_201706, jdata_user_order_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBuyMonthNumber(jdata_train_df_201705, jdata_user_order_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBuyMonthNumber(jdata_train_df_201704, jdata_user_order_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBuyMonthNumber(jdata_train_df_201703, jdata_user_order_data, FeatureMonthList_201703)
    jdata_test_df = getUserBuyMonthNumber(jdata_test_df, jdata_user_order_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserSkuActionNumber(jdata_train_df_201708, jdata_user_action_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserSkuActionNumber(jdata_train_df_201707, jdata_user_action_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserSkuActionNumber(jdata_train_df_201706, jdata_user_action_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserSkuActionNumber(jdata_train_df_201705, jdata_user_action_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserSkuActionNumber(jdata_train_df_201704, jdata_user_action_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserSkuActionNumber(jdata_train_df_201703, jdata_user_action_data, FeatureMonthList_201703)
    jdata_test_df = getUserSkuActionNumber(jdata_test_df, jdata_user_action_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserSkuActionDayNumber(jdata_train_df_201708, jdata_user_action_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserSkuActionDayNumber(jdata_train_df_201707, jdata_user_action_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserSkuActionDayNumber(jdata_train_df_201706, jdata_user_action_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserSkuActionDayNumber(jdata_train_df_201705, jdata_user_action_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserSkuActionDayNumber(jdata_train_df_201704, jdata_user_action_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserSkuActionDayNumber(jdata_train_df_201703, jdata_user_action_data, FeatureMonthList_201703)
    jdata_test_df = getUserSkuActionDayNumber(jdata_test_df, jdata_user_action_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserActionNumber(jdata_train_df_201708, jdata_user_action_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserActionNumber(jdata_train_df_201707, jdata_user_action_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserActionNumber(jdata_train_df_201706, jdata_user_action_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserActionNumber(jdata_train_df_201705, jdata_user_action_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserActionNumber(jdata_train_df_201704, jdata_user_action_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserActionNumber(jdata_train_df_201703, jdata_user_action_data, FeatureMonthList_201703)
    jdata_test_df = getUserActionNumber(jdata_test_df, jdata_user_action_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserSkuBrowseNumber(jdata_train_df_201708, jdata_user_action_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserSkuBrowseNumber(jdata_train_df_201707, jdata_user_action_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserSkuBrowseNumber(jdata_train_df_201706, jdata_user_action_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserSkuBrowseNumber(jdata_train_df_201705, jdata_user_action_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserSkuBrowseNumber(jdata_train_df_201704, jdata_user_action_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserSkuBrowseNumber(jdata_train_df_201703, jdata_user_action_data, FeatureMonthList_201703)
    jdata_test_df = getUserSkuBrowseNumber(jdata_test_df, jdata_user_action_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserSkuBrowseDayNumber(jdata_train_df_201708, jdata_user_action_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserSkuBrowseDayNumber(jdata_train_df_201707, jdata_user_action_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserSkuBrowseDayNumber(jdata_train_df_201706, jdata_user_action_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserSkuBrowseDayNumber(jdata_train_df_201705, jdata_user_action_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserSkuBrowseDayNumber(jdata_train_df_201704, jdata_user_action_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserSkuBrowseDayNumber(jdata_train_df_201703, jdata_user_action_data, FeatureMonthList_201703)
    jdata_test_df = getUserSkuBrowseDayNumber(jdata_test_df, jdata_user_action_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserBrowseNumber(jdata_train_df_201708, jdata_user_action_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserBrowseNumber(jdata_train_df_201707, jdata_user_action_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserBrowseNumber(jdata_train_df_201706, jdata_user_action_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserBrowseNumber(jdata_train_df_201705, jdata_user_action_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserBrowseNumber(jdata_train_df_201704, jdata_user_action_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserBrowseNumber(jdata_train_df_201703, jdata_user_action_data, FeatureMonthList_201703)
    jdata_test_df = getUserBrowseNumber(jdata_test_df, jdata_user_action_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserSkuFocusNumber(jdata_train_df_201708, jdata_user_action_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserSkuFocusNumber(jdata_train_df_201707, jdata_user_action_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserSkuFocusNumber(jdata_train_df_201706, jdata_user_action_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserSkuFocusNumber(jdata_train_df_201705, jdata_user_action_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserSkuFocusNumber(jdata_train_df_201704, jdata_user_action_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserSkuFocusNumber(jdata_train_df_201703, jdata_user_action_data, FeatureMonthList_201703)
    jdata_test_df = getUserSkuFocusNumber(jdata_test_df, jdata_user_action_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserFocusNumber(jdata_train_df_201708, jdata_user_action_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserFocusNumber(jdata_train_df_201707, jdata_user_action_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserFocusNumber(jdata_train_df_201706, jdata_user_action_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserFocusNumber(jdata_train_df_201705, jdata_user_action_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserFocusNumber(jdata_train_df_201704, jdata_user_action_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserFocusNumber(jdata_train_df_201703, jdata_user_action_data, FeatureMonthList_201703)
    jdata_test_df = getUserFocusNumber(jdata_test_df, jdata_user_action_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserCommentNumber(jdata_train_df_201708, jdata_user_comment_score_data, FeatureMonthList_201708)
    jdata_train_df_201707 = getUserCommentNumber(jdata_train_df_201707, jdata_user_comment_score_data, FeatureMonthList_201707)
    jdata_train_df_201706 = getUserCommentNumber(jdata_train_df_201706, jdata_user_comment_score_data, FeatureMonthList_201706)
    jdata_train_df_201705 = getUserCommentNumber(jdata_train_df_201705, jdata_user_comment_score_data, FeatureMonthList_201705)
    jdata_train_df_201704 = getUserCommentNumber(jdata_train_df_201704, jdata_user_comment_score_data, FeatureMonthList_201704)
    jdata_train_df_201703 = getUserCommentNumber(jdata_train_df_201703, jdata_user_comment_score_data, FeatureMonthList_201703)
    jdata_test_df = getUserCommentNumber(jdata_test_df, jdata_user_comment_score_data, FeatureMonthList_201709)

    jdata_train_df_201708 = getUserMonthlyBuyNumber(jdata_train_df_201708, jdata_user_order_data, datetime.datetime(2017, 8, 1), 11)
    jdata_train_df_201707 = getUserMonthlyBuyNumber(jdata_train_df_201707, jdata_user_order_data, datetime.datetime(2017, 7, 1), 10)
    jdata_train_df_201706 = getUserMonthlyBuyNumber(jdata_train_df_201706, jdata_user_order_data, datetime.datetime(2017, 6, 1), 9)
    jdata_train_df_201705 = getUserMonthlyBuyNumber(jdata_train_df_201705, jdata_user_order_data, datetime.datetime(2017, 5, 1), 8)
    jdata_train_df_201704 = getUserMonthlyBuyNumber(jdata_train_df_201704, jdata_user_order_data, datetime.datetime(2017, 4, 1), 7)
    jdata_train_df_201703 = getUserMonthlyBuyNumber(jdata_train_df_201703, jdata_user_order_data, datetime.datetime(2017, 3, 1), 6)
    jdata_test_df = getUserMonthlyBuyNumber(jdata_test_df, jdata_user_order_data, datetime.datetime(2017, 9, 1), 12)

    jdata_train_df_201708 = getUserMonthlyOrderNumber(jdata_train_df_201708, jdata_user_order_data, datetime.datetime(2017, 8, 1), 11)
    jdata_train_df_201707 = getUserMonthlyOrderNumber(jdata_train_df_201707, jdata_user_order_data, datetime.datetime(2017, 7, 1), 10)
    jdata_train_df_201706 = getUserMonthlyOrderNumber(jdata_train_df_201706, jdata_user_order_data, datetime.datetime(2017, 6, 1), 9)
    jdata_train_df_201705 = getUserMonthlyOrderNumber(jdata_train_df_201705, jdata_user_order_data, datetime.datetime(2017, 5, 1), 8)
    jdata_train_df_201704 = getUserMonthlyOrderNumber(jdata_train_df_201704, jdata_user_order_data, datetime.datetime(2017, 4, 1), 7)
    jdata_train_df_201703 = getUserMonthlyOrderNumber(jdata_train_df_201703, jdata_user_order_data, datetime.datetime(2017, 3, 1), 6)
    jdata_test_df = getUserMonthlyOrderNumber(jdata_test_df, jdata_user_order_data, datetime.datetime(2017, 9, 1), 12)

    jdata_train_df_201708 = getUserMonthlyBuyCount(jdata_train_df_201708, jdata_user_order_data, datetime.datetime(2017, 8, 1), 11)
    jdata_train_df_201707 = getUserMonthlyBuyCount(jdata_train_df_201707, jdata_user_order_data, datetime.datetime(2017, 7, 1), 10)
    jdata_train_df_201706 = getUserMonthlyBuyCount(jdata_train_df_201706, jdata_user_order_data, datetime.datetime(2017, 6, 1), 9)
    jdata_train_df_201705 = getUserMonthlyBuyCount(jdata_train_df_201705, jdata_user_order_data, datetime.datetime(2017, 5, 1), 8)
    jdata_train_df_201704 = getUserMonthlyBuyCount(jdata_train_df_201704, jdata_user_order_data, datetime.datetime(2017, 4, 1), 7)
    jdata_train_df_201703 = getUserMonthlyBuyCount(jdata_train_df_201703, jdata_user_order_data, datetime.datetime(2017, 3, 1), 6)
    jdata_test_df = getUserMonthlyBuyCount(jdata_test_df, jdata_user_order_data, datetime.datetime(2017, 9, 1), 12)

    jdata_train_df_201708 = getUserMonthlyBuyDay(jdata_train_df_201708, jdata_user_order_data, datetime.datetime(2017, 8, 1), 11)
    jdata_train_df_201707 = getUserMonthlyBuyDay(jdata_train_df_201707, jdata_user_order_data, datetime.datetime(2017, 7, 1), 10)
    jdata_train_df_201706 = getUserMonthlyBuyDay(jdata_train_df_201706, jdata_user_order_data, datetime.datetime(2017, 6, 1), 9)
    jdata_train_df_201705 = getUserMonthlyBuyDay(jdata_train_df_201705, jdata_user_order_data, datetime.datetime(2017, 5, 1), 8)
    jdata_train_df_201704 = getUserMonthlyBuyDay(jdata_train_df_201704, jdata_user_order_data, datetime.datetime(2017, 4, 1), 7)
    jdata_train_df_201703 = getUserMonthlyBuyDay(jdata_train_df_201703, jdata_user_order_data, datetime.datetime(2017, 3, 1), 6)
    jdata_test_df = getUserMonthlyBuyDay(jdata_test_df, jdata_user_order_data, datetime.datetime(2017, 9, 1), 12)

    jdata_train_df_201708 = getUserMonthlyBuyPriceMin(jdata_train_df_201708, jdata_user_order_data, datetime.datetime(2017, 8, 1))
    jdata_train_df_201707 = getUserMonthlyBuyPriceMin(jdata_train_df_201707, jdata_user_order_data, datetime.datetime(2017, 7, 1))
    jdata_train_df_201706 = getUserMonthlyBuyPriceMin(jdata_train_df_201706, jdata_user_order_data, datetime.datetime(2017, 6, 1))
    jdata_train_df_201705 = getUserMonthlyBuyPriceMin(jdata_train_df_201705, jdata_user_order_data, datetime.datetime(2017, 5, 1))
    jdata_train_df_201704 = getUserMonthlyBuyPriceMin(jdata_train_df_201704, jdata_user_order_data, datetime.datetime(2017, 4, 1))
    jdata_train_df_201703 = getUserMonthlyBuyPriceMin(jdata_train_df_201703, jdata_user_order_data, datetime.datetime(2017, 3, 1))
    jdata_test_df = getUserMonthlyBuyPriceMin(jdata_test_df, jdata_user_order_data, datetime.datetime(2017, 9, 1))

    jdata_train_df_201708 = getUserMonthlyBuyPriceMax(jdata_train_df_201708, jdata_user_order_data, datetime.datetime(2017, 8, 1))
    jdata_train_df_201707 = getUserMonthlyBuyPriceMax(jdata_train_df_201707, jdata_user_order_data, datetime.datetime(2017, 7, 1))
    jdata_train_df_201706 = getUserMonthlyBuyPriceMax(jdata_train_df_201706, jdata_user_order_data, datetime.datetime(2017, 6, 1))
    jdata_train_df_201705 = getUserMonthlyBuyPriceMax(jdata_train_df_201705, jdata_user_order_data, datetime.datetime(2017, 5, 1))
    jdata_train_df_201704 = getUserMonthlyBuyPriceMax(jdata_train_df_201704, jdata_user_order_data, datetime.datetime(2017, 4, 1))
    jdata_train_df_201703 = getUserMonthlyBuyPriceMax(jdata_train_df_201703, jdata_user_order_data, datetime.datetime(2017, 3, 1))
    jdata_test_df = getUserMonthlyBuyPriceMax(jdata_test_df, jdata_user_order_data, datetime.datetime(2017, 9, 1))

    jdata_train_df_201708 = getUserMonthlyBuyPriceMean(jdata_train_df_201708, jdata_user_order_data, datetime.datetime(2017, 8, 1))
    jdata_train_df_201707 = getUserMonthlyBuyPriceMean(jdata_train_df_201707, jdata_user_order_data, datetime.datetime(2017, 7, 1))
    jdata_train_df_201706 = getUserMonthlyBuyPriceMean(jdata_train_df_201706, jdata_user_order_data, datetime.datetime(2017, 6, 1))
    jdata_train_df_201705 = getUserMonthlyBuyPriceMean(jdata_train_df_201705, jdata_user_order_data, datetime.datetime(2017, 5, 1))
    jdata_train_df_201704 = getUserMonthlyBuyPriceMean(jdata_train_df_201704, jdata_user_order_data, datetime.datetime(2017, 4, 1))
    jdata_train_df_201703 = getUserMonthlyBuyPriceMean(jdata_train_df_201703, jdata_user_order_data, datetime.datetime(2017, 3, 1))
    jdata_test_df = getUserMonthlyBuyPriceMean(jdata_test_df, jdata_user_order_data, datetime.datetime(2017, 9, 1))

    jdata_train_df_201708 = getUserMonthlyBuyPara1Min(jdata_train_df_201708, jdata_user_order_data, datetime.datetime(2017, 8, 1))
    jdata_train_df_201707 = getUserMonthlyBuyPara1Min(jdata_train_df_201707, jdata_user_order_data, datetime.datetime(2017, 7, 1))
    jdata_train_df_201706 = getUserMonthlyBuyPara1Min(jdata_train_df_201706, jdata_user_order_data, datetime.datetime(2017, 6, 1))
    jdata_train_df_201705 = getUserMonthlyBuyPara1Min(jdata_train_df_201705, jdata_user_order_data, datetime.datetime(2017, 5, 1))
    jdata_train_df_201704 = getUserMonthlyBuyPara1Min(jdata_train_df_201704, jdata_user_order_data, datetime.datetime(2017, 4, 1))
    jdata_train_df_201703 = getUserMonthlyBuyPara1Min(jdata_train_df_201703, jdata_user_order_data, datetime.datetime(2017, 3, 1))
    jdata_test_df = getUserMonthlyBuyPara1Min(jdata_test_df, jdata_user_order_data, datetime.datetime(2017, 9, 1))

    jdata_train_df_201708 = getUserMonthlyBuyPara1Max(jdata_train_df_201708, jdata_user_order_data, datetime.datetime(2017, 8, 1))
    jdata_train_df_201707 = getUserMonthlyBuyPara1Max(jdata_train_df_201707, jdata_user_order_data, datetime.datetime(2017, 7, 1))
    jdata_train_df_201706 = getUserMonthlyBuyPara1Max(jdata_train_df_201706, jdata_user_order_data, datetime.datetime(2017, 6, 1))
    jdata_train_df_201705 = getUserMonthlyBuyPara1Max(jdata_train_df_201705, jdata_user_order_data, datetime.datetime(2017, 5, 1))
    jdata_train_df_201704 = getUserMonthlyBuyPara1Max(jdata_train_df_201704, jdata_user_order_data, datetime.datetime(2017, 4, 1))
    jdata_train_df_201703 = getUserMonthlyBuyPara1Max(jdata_train_df_201703, jdata_user_order_data, datetime.datetime(2017, 3, 1))
    jdata_test_df = getUserMonthlyBuyPara1Max(jdata_test_df, jdata_user_order_data, datetime.datetime(2017, 9, 1))

    jdata_train_df_201708 = getUserMonthlyBuyPara1Mean(jdata_train_df_201708, jdata_user_order_data, datetime.datetime(2017, 8, 1))
    jdata_train_df_201707 = getUserMonthlyBuyPara1Mean(jdata_train_df_201707, jdata_user_order_data, datetime.datetime(2017, 7, 1))
    jdata_train_df_201706 = getUserMonthlyBuyPara1Mean(jdata_train_df_201706, jdata_user_order_data, datetime.datetime(2017, 6, 1))
    jdata_train_df_201705 = getUserMonthlyBuyPara1Mean(jdata_train_df_201705, jdata_user_order_data, datetime.datetime(2017, 5, 1))
    jdata_train_df_201704 = getUserMonthlyBuyPara1Mean(jdata_train_df_201704, jdata_user_order_data, datetime.datetime(2017, 4, 1))
    jdata_train_df_201703 = getUserMonthlyBuyPara1Mean(jdata_train_df_201703, jdata_user_order_data, datetime.datetime(2017, 3, 1))
    jdata_test_df = getUserMonthlyBuyPara1Mean(jdata_test_df, jdata_user_order_data, datetime.datetime(2017, 9, 1))

    jdata_train_df_201708 = getUserMonthlyFocusNumber(jdata_train_df_201708, jdata_user_action_data, datetime.datetime(2017, 8, 1), 11)
    jdata_train_df_201707 = getUserMonthlyFocusNumber(jdata_train_df_201707, jdata_user_action_data, datetime.datetime(2017, 7, 1), 10)
    jdata_train_df_201706 = getUserMonthlyFocusNumber(jdata_train_df_201706, jdata_user_action_data, datetime.datetime(2017, 6, 1), 9)
    jdata_train_df_201705 = getUserMonthlyFocusNumber(jdata_train_df_201705, jdata_user_action_data, datetime.datetime(2017, 5, 1), 8)
    jdata_train_df_201704 = getUserMonthlyFocusNumber(jdata_train_df_201704, jdata_user_action_data, datetime.datetime(2017, 4, 1), 7)
    jdata_train_df_201703 = getUserMonthlyFocusNumber(jdata_train_df_201703, jdata_user_action_data, datetime.datetime(2017, 3, 1), 6)
    jdata_test_df = getUserMonthlyFocusNumber(jdata_test_df, jdata_user_action_data, datetime.datetime(2017, 9, 1), 12)

    jdata_train_df_201708 = getUserMonthlyBrowseNumber(jdata_train_df_201708, jdata_user_action_data, datetime.datetime(2017, 8, 1), 11)
    jdata_train_df_201707 = getUserMonthlyBrowseNumber(jdata_train_df_201707, jdata_user_action_data, datetime.datetime(2017, 7, 1), 10)
    jdata_train_df_201706 = getUserMonthlyBrowseNumber(jdata_train_df_201706, jdata_user_action_data, datetime.datetime(2017, 6, 1), 9)
    jdata_train_df_201705 = getUserMonthlyBrowseNumber(jdata_train_df_201705, jdata_user_action_data, datetime.datetime(2017, 5, 1), 8)
    jdata_train_df_201704 = getUserMonthlyBrowseNumber(jdata_train_df_201704, jdata_user_action_data, datetime.datetime(2017, 4, 1), 7)
    jdata_train_df_201703 = getUserMonthlyBrowseNumber(jdata_train_df_201703, jdata_user_action_data, datetime.datetime(2017, 3, 1), 6)
    jdata_test_df = getUserMonthlyBrowseNumber(jdata_test_df, jdata_user_action_data, datetime.datetime(2017, 9, 1), 12)

    jdata_train_df_201708 = getUserMonthlyActionNumber(jdata_train_df_201708, jdata_user_action_data, datetime.datetime(2017, 8, 1), 11)
    jdata_train_df_201707 = getUserMonthlyActionNumber(jdata_train_df_201707, jdata_user_action_data, datetime.datetime(2017, 7, 1), 10)
    jdata_train_df_201706 = getUserMonthlyActionNumber(jdata_train_df_201706, jdata_user_action_data, datetime.datetime(2017, 6, 1), 9)
    jdata_train_df_201705 = getUserMonthlyActionNumber(jdata_train_df_201705, jdata_user_action_data, datetime.datetime(2017, 5, 1), 8)
    jdata_train_df_201704 = getUserMonthlyActionNumber(jdata_train_df_201704, jdata_user_action_data, datetime.datetime(2017, 4, 1), 7)
    jdata_train_df_201703 = getUserMonthlyActionNumber(jdata_train_df_201703, jdata_user_action_data, datetime.datetime(2017, 3, 1), 6)
    jdata_test_df = getUserMonthlyActionNumber(jdata_test_df, jdata_user_action_data, datetime.datetime(2017, 9, 1), 12)

    jdata_train_df_201708 = getUserMonthlyActionDayNumber(jdata_train_df_201708, jdata_user_action_data, datetime.datetime(2017, 8, 1), 11)
    jdata_train_df_201707 = getUserMonthlyActionDayNumber(jdata_train_df_201707, jdata_user_action_data, datetime.datetime(2017, 7, 1), 10)
    jdata_train_df_201706 = getUserMonthlyActionDayNumber(jdata_train_df_201706, jdata_user_action_data, datetime.datetime(2017, 6, 1), 9)
    jdata_train_df_201705 = getUserMonthlyActionDayNumber(jdata_train_df_201705, jdata_user_action_data, datetime.datetime(2017, 5, 1), 8)
    jdata_train_df_201704 = getUserMonthlyActionDayNumber(jdata_train_df_201704, jdata_user_action_data, datetime.datetime(2017, 4, 1), 7)
    jdata_train_df_201703 = getUserMonthlyActionDayNumber(jdata_train_df_201703, jdata_user_action_data, datetime.datetime(2017, 3, 1), 6)
    jdata_test_df = getUserMonthlyActionDayNumber(jdata_test_df, jdata_user_action_data, datetime.datetime(2017, 9, 1), 12)

    jdata_train_df_201708 = getUserMonthlyActionSkuNumber(jdata_train_df_201708, jdata_user_action_data, datetime.datetime(2017, 8, 1), 11)
    jdata_train_df_201707 = getUserMonthlyActionSkuNumber(jdata_train_df_201707, jdata_user_action_data, datetime.datetime(2017, 7, 1), 10)
    jdata_train_df_201706 = getUserMonthlyActionSkuNumber(jdata_train_df_201706, jdata_user_action_data, datetime.datetime(2017, 6, 1), 9)
    jdata_train_df_201705 = getUserMonthlyActionSkuNumber(jdata_train_df_201705, jdata_user_action_data, datetime.datetime(2017, 5, 1), 8)
    jdata_train_df_201704 = getUserMonthlyActionSkuNumber(jdata_train_df_201704, jdata_user_action_data, datetime.datetime(2017, 4, 1), 7)
    jdata_train_df_201703 = getUserMonthlyActionSkuNumber(jdata_train_df_201703, jdata_user_action_data, datetime.datetime(2017, 3, 1), 6)
    jdata_test_df = getUserMonthlyActionSkuNumber(jdata_test_df, jdata_user_action_data, datetime.datetime(2017, 9, 1), 12)
    return jdata_train_df_201708, jdata_train_df_201707, jdata_train_df_201706, jdata_train_df_201705, jdata_train_df_201704, jdata_train_df_201703, jdata_test_df

# 导出训练集和测试集预处理结果
def exportDf(df, fileName):
    df.to_csv('../temp/%s.csv' % fileName, header=True, index=False)

# 导出预测结果
def exportResult(df, fileName):
    df.to_csv('../result/%s.csv' % fileName, header=True, index=False)

class XgbModel:
    def __init__(self, feaNames=None, params={}):
        self.feaNames = feaNames
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric':'auc',
            'silent': True,
            'eta': 0.05,
            'max_depth': 7,
            'gamma': 11,
            'subsample': 0.9,
            'colsample_bytree': 0.85,
            'min_child_weight': 1,
            'max_delta_step': 1,
            'lambda': 30,
#             'nthread': 20,
        }

        for k,v in params.items():
            self.params[k] = v
        self.clf = None

    def train(self, X, y, train_size=1, test_size=0.1, verbose=True, num_boost_round=1000, early_stopping_rounds=3):
        X = X.astype(float)
        if train_size==1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            X_train, y_train = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feaNames)
        dval = xgb.DMatrix(X_test, label=y_test, feature_names=self.feaNames)
        watchlist = [(dtrain,'train'),(dval,'val')]
        clf = xgb.train(
            self.params, dtrain,
            num_boost_round = num_boost_round,
            evals = watchlist,
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval=verbose
        )
        self.clf = clf

    def trainCV(self, X, y, nFold=3, verbose=True, num_boost_round=7000, early_stopping_rounds=10):
        X = X.astype(float)
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feaNames)
        cvResult = xgb.cv(
            self.params, dtrain,
            num_boost_round = num_boost_round,
            nfold = nFold,
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval=verbose
        )
        clf = xgb.train(
            self.params, dtrain,
            num_boost_round = cvResult.shape[0],
        )
        self.clf = clf

    def gridSearch(self, X, y, nFold=3, verbose=1, num_boost_round=450):
        paramsGrids = {
            # 'n_estimators': [50+5*i for i in range(0,30)],
            'gamma': [0,0.01,0.05,0.1,0.5,1,5,10,50,100],
            'max_depth': list(range(3,7)),
            'min_child_weight': list(range(0,10)),
            'subsample': [1-0.05*i for i in range(0,8)],
            'colsample_bytree': [1-0.05*i for i in range(0,10)],
            # 'reg_alpha': [0+2*i for i in range(0,10)],
            'reg_lambda': [0+50*i for i in range(0,10)],
            'max_delta_step': [0+1*i for i in range(0,8)],
        }
        for k,v in paramsGrids.items():
            gsearch = GridSearchCV(
                estimator = xgb.XGBClassifier(
                    max_depth = self.params['max_depth'],
                    gamma = self.params['gamma'],
                    learning_rate = self.params['eta'],
                    max_delta_step = self.params['max_delta_step'],
                    min_child_weight = self.params['min_child_weight'],
                    subsample = self.params['subsample'],
                    colsample_bytree = self.params['colsample_bytree'],
                    silent = self.params['silent'],
                    reg_lambda = self.params['lambda'],
                    n_estimators = num_boost_round
                ),
                # param_grid = paramsGrids,
                param_grid = {k:v},
                scoring = 'roc_auc',
                cv = nFold,
                verbose = verbose,
                n_jobs = 4
            )
            gsearch.fit(X, y)
            print(pd.DataFrame(gsearch.cv_results_))
            print(gsearch.best_params_)
        exit()

    def predict(self, X):
        X = X.astype(float)
        return self.clf.predict(xgb.DMatrix(X, feature_names=self.feaNames))

    def getFeaScore(self, show=False):
        fscore = self.clf.get_score()
        feaNames = fscore.keys()
        scoreDf = pd.DataFrame(index=feaNames, columns=['importance'])
        for k,v in fscore.items():
            scoreDf.loc[k, 'importance'] = v
        if show:
            print(scoreDf.sort_index(by=['importance'], ascending=False))
        return scoreDf

# 划分训练集和测试集
def trainTestSplit(df, splitDate=pd.to_datetime('2018-09-23'), trainPeriod=3, testPeriod=1):
    trainDf = df[(df.context_timestamp<splitDate)&(df.context_timestamp>=splitDate-timedelta(days=trainPeriod))]
    testDf = df[(df.context_timestamp>=splitDate)&(df.context_timestamp<splitDate+timedelta(days=testPeriod))]
    return (trainDf, testDf)

# 获取stacking下一层数据集
def getOof(clf, trainX, trainY, testX, nFold=5, stratify=False):
    oofTrain = np.zeros(trainX.shape[0])
    oofTest = np.zeros(testX.shape[0])
    oofTestSkf = np.zeros((testX.shape[0], nFold))
    if stratify:
        kf = StratifiedKFold(n_splits=nFold, shuffle=True)
    else:
        kf = KFold(n_splits=nFold, shuffle=True)
    for i, (trainIdx, testIdx) in enumerate(kf.split(trainX, trainY)):
        kfTrainX = trainX[trainIdx]
        kfTrainY = trainY[trainIdx]
        kfTestX = trainX[testIdx]
        clf.trainCV(kfTrainX, kfTrainY, verbose=False)
        oofTrain[testIdx] = clf.predict(kfTestX)
        oofTestSkf[:,i] = clf.predict(testX)
    oofTest[:] = oofTestSkf.mean(axis=1)
    return oofTrain, oofTest

def main():
    # 特征处理
    startTime = datetime.datetime.now()

    print('~~~~~~~~~~~~~~开始导入数据~~~~~~~~~~~~~~~~~~~')

    #首先导入用户基本信息，作为每个月的样本，构造训练集和测试集
    #用户基本信息表--jdata_user_basic_info
    jdata_user_basic_info_data = pd.read_csv('../data/jdata_user_basic_info.csv')

    #用户订单表--jdata_user_order
    jdata_user_order_data = pd.read_csv('../data/jdata_user_order.csv')

    #导入剩余的3张表
    #用户行为表--jdata_user_action
    jdata_user_action_data = pd.read_csv('../data/jdata_user_action.csv')

    jdata_user_action_data['sku_id'] = jdata_user_action_data['sku_id'].fillna(-1)
    jdata_user_action_data = jdata_user_action_data[jdata_user_action_data.sku_id != -1]

    #SKU基本信息表--jdata_sku_basic_info
    jdata_sku_basic_info_data = pd.read_csv('../data/jdata_sku_basic_info.csv')

    #评论分数数据表--jdata_user_comment_score
    jdata_user_comment_score_data = pd.read_csv('../data/jdata_user_comment_score.csv')

    print('~~~~~~~~~~~~~~导入数据完毕~~~~~~~~~~~~~~~~~~~')

    jdata_user_order_data = dealDateColumns(jdata_user_order_data, 'o_date')
    jdata_user_action_data = dealDateColumns(jdata_user_action_data, 'a_date')
    jdata_user_comment_score_data = dealDateColumns(jdata_user_comment_score_data, 'comment_create_tm')

    #首先过滤出目标类目订单记录用于训练集打标签
    jdata_user_order_data = pd.merge(jdata_user_order_data, jdata_sku_basic_info_data[['sku_id', 'cate', 'price', 'para_1']], on=['sku_id'], how='left')
    jdata_user_action_data = pd.merge(jdata_user_action_data, jdata_sku_basic_info_data[['sku_id', 'cate']], on=['sku_id'], how='left')
    jdata_user_order_data_target = jdata_user_order_data[(jdata_user_order_data.cate == 101) | (jdata_user_order_data.cate == 30)]

    jdata_train_df_201708 = getTrainDfMonthly(jdata_user_order_data_target, jdata_user_basic_info_data, 8, 2017, datetime.datetime(2017, 5, 1), datetime.datetime(2017, 8, 1))
    jdata_train_df_201707 = getTrainDfMonthly(jdata_user_order_data_target, jdata_user_basic_info_data, 7, 2017, datetime.datetime(2017, 4, 1), datetime.datetime(2017, 7, 1))
    jdata_train_df_201706 = getTrainDfMonthly(jdata_user_order_data_target, jdata_user_basic_info_data, 6, 2017, datetime.datetime(2017, 3, 1), datetime.datetime(2017, 6, 1))
    jdata_train_df_201705 = getTrainDfMonthly(jdata_user_order_data_target, jdata_user_basic_info_data, 5, 2017, datetime.datetime(2017, 2, 1), datetime.datetime(2017, 5, 1))
    jdata_train_df_201704 = getTrainDfMonthly(jdata_user_order_data_target, jdata_user_basic_info_data, 4, 2017, datetime.datetime(2017, 1, 1), datetime.datetime(2017, 4, 1))
    jdata_train_df_201703 = getTrainDfMonthly(jdata_user_order_data_target, jdata_user_basic_info_data, 3, 2017, datetime.datetime(2016, 12, 1), datetime.datetime(2017, 3, 1))

    #构造测试集
    jdata_test_df = jdata_user_basic_info_data.copy()
    jdata_test_df['year'] = 2017
    jdata_test_df['month'] = 9

    print('~~~~~~~~~~~~~~训练集和测试集数据构造完毕~~~~~~~~~~~~~~~~~~~')

    # 为训练集和测试集定义滑窗区间
    FeatureMonthList_201708 = [(datetime.datetime(2017, 7, 1), datetime.datetime(2017, 8, 1), 1),
                              (datetime.datetime(2017, 7, 15), datetime.datetime(2017, 8, 1), 0.5),
                              (datetime.datetime(2017, 5, 1), datetime.datetime(2017, 8, 1), 3),
                              (datetime.datetime(2017, 2, 1), datetime.datetime(2017, 8, 1), 6)]

    FeatureMonthList_201707 = [(datetime.datetime(2017, 6, 1), datetime.datetime(2017, 7, 1), 1),
                              (datetime.datetime(2017, 6, 15), datetime.datetime(2017, 7, 1), 0.5),
                              (datetime.datetime(2017, 4, 1), datetime.datetime(2017, 7, 1), 3),
                              (datetime.datetime(2017, 1, 1), datetime.datetime(2017, 7, 1), 6)]

    FeatureMonthList_201706 = [(datetime.datetime(2017, 5, 1), datetime.datetime(2017, 6, 1), 1),
                              (datetime.datetime(2017, 5, 15), datetime.datetime(2017, 6, 1), 0.5),
                              (datetime.datetime(2017, 3, 1), datetime.datetime(2017, 6, 1), 3),
                              (datetime.datetime(2016, 12, 1), datetime.datetime(2017, 6, 1), 6)]

    FeatureMonthList_201705 = [(datetime.datetime(2017, 4, 1), datetime.datetime(2017, 5, 1), 1),
                              (datetime.datetime(2017, 4, 15), datetime.datetime(2017, 5, 1), 0.5),
                              (datetime.datetime(2017, 2, 1), datetime.datetime(2017, 5, 1), 3),
                              (datetime.datetime(2016, 11, 1), datetime.datetime(2017, 5, 1), 6)]

    FeatureMonthList_201704 = [(datetime.datetime(2017, 3, 1), datetime.datetime(2017, 4, 1), 1),
                               (datetime.datetime(2017, 3, 15), datetime.datetime(2017, 4, 1), 0.5),
                              (datetime.datetime(2017, 1, 1), datetime.datetime(2017, 4, 1), 3),
                              (datetime.datetime(2016, 10, 1), datetime.datetime(2017, 4, 1), 6)]

    FeatureMonthList_201703 = [(datetime.datetime(2017, 2, 1), datetime.datetime(2017, 3, 1), 1),
                              (datetime.datetime(2017, 2, 15), datetime.datetime(2017, 3, 1), 0.5),
                              (datetime.datetime(2016, 12, 1), datetime.datetime(2017, 3, 1), 3),
                              (datetime.datetime(2016, 9, 1), datetime.datetime(2017, 3, 1), 6)]

    FeatureMonthList_201709 = [(datetime.datetime(2017, 8, 1), datetime.datetime(2017, 9, 1), 1),
                              (datetime.datetime(2017, 8, 15), datetime.datetime(2017, 9, 1), 0.5),
                              (datetime.datetime(2017, 6, 1), datetime.datetime(2017, 9, 1), 3),
                              (datetime.datetime(2017, 3, 1), datetime.datetime(2017, 9, 1), 6)]

    print('~~~~~~~~~~~~~~开始进行特征工程~~~~~~~~~~~~~~~~~~~')

    jdata_train_df_201708, jdata_train_df_201707, jdata_train_df_201706, jdata_train_df_201705, jdata_train_df_201704, jdata_train_df_201703, jdata_test_df = featureDeal(jdata_train_df_201708, jdata_train_df_201707, jdata_train_df_201706, jdata_train_df_201705, jdata_train_df_201704, jdata_train_df_201703, jdata_test_df, FeatureMonthList_201708, FeatureMonthList_201707, FeatureMonthList_201706, FeatureMonthList_201705, FeatureMonthList_201704, FeatureMonthList_201703, FeatureMonthList_201709, jdata_user_order_data, jdata_user_action_data, jdata_user_comment_score_data)
    print('jdata_train_df_201708 len : ', len(jdata_train_df_201708))
    print('jdata_train_df_201708 feature len : ', len(jdata_train_df_201708.columns.values))

    jdata_train_df = pd.concat([jdata_train_df_201703, jdata_train_df_201704, jdata_train_df_201705, jdata_train_df_201706, jdata_train_df_201707, jdata_train_df_201708])
    print('train_df len : ', len(jdata_train_df))

    print('~~~~~~~~~~~~~~特征工程处理完毕~~~~~~~~~~~~~~~~~~~')

    exportDf(jdata_train_df, 'jdata_train_df_b')
    exportDf(jdata_test_df, 'jdata_test_df_b')
    print('pretreatment time:', datetime.datetime.now()-startTime)

    print('~~~~~~~~~~~~~~训练集，测试集数据导出完毕~~~~~~~~~~~~~~~~~~~')

    print('~~~~~~~~~~~~~~开始训练~~~~~~~~~~~~~~~~~~~')

    fea = [

        'age', 'sex', 'user_lv_cd',
         'user_cate101_last1Month_orderNumber', 'user_cate30_last1Month_orderNumber', 'user_targetCate_last1Month_orderNumber', 'user_relatedCate_last1Month_orderNumber',
         'user_cate101_last0.5Month_orderNumber', 'user_cate30_last0.5Month_orderNumber', 'user_targetCate_last0.5Month_orderNumber', 'user_relatedCate_last0.5Month_orderNumber',
         'user_cate101_last3Month_orderNumber', 'user_cate30_last3Month_orderNumber', 'user_targetCate_last3Month_orderNumber', 'user_relatedCate_last3Month_orderNumber',
         'user_cate101_last6Month_orderNumber', 'user_cate30_last6Month_orderNumber', 'user_targetCate_last6Month_orderNumber', 'user_relatedCate_last6Month_orderNumber',
         'user_cate101_last1Month_buyNumber', 'user_cate30_last1Month_buyNumber', 'user_targetCate_last1Month_buyNumber', 'user_relatedCate_last1Month_buyNumber',
         'user_cate101_last0.5Month_buyNumber', 'user_cate30_last0.5Month_buyNumber', 'user_targetCate_last0.5Month_buyNumber', 'user_relatedCate_last0.5Month_buyNumber',
         'user_cate101_last3Month_buyNumber', 'user_cate30_last3Month_buyNumber', 'user_targetCate_last3Month_buyNumber', 'user_relatedCate_last3Month_buyNumber',
         'user_cate101_last6Month_buyNumber', 'user_cate30_last6Month_buyNumber', 'user_targetCate_last6Month_buyNumber', 'user_relatedCate_last6Month_buyNumber',
         'user_cate101_last1Month_buyPriceMin', 'user_cate30_last1Month_buyPriceMin', 'user_targetCate_last1Month_buyPriceMin', 'user_relatedCate_last1Month_buyPriceMin',
         'user_cate101_last0.5Month_buyPriceMin', 'user_cate30_last0.5Month_buyPriceMin', 'user_targetCate_last0.5Month_buyPriceMin', 'user_relatedCate_last0.5Month_buyPriceMin',
         'user_cate101_last3Month_buyPriceMin', 'user_cate30_last3Month_buyPriceMin', 'user_targetCate_last3Month_buyPriceMin', 'user_relatedCate_last3Month_buyPriceMin',
         'user_cate101_last6Month_buyPriceMin', 'user_cate30_last6Month_buyPriceMin', 'user_targetCate_last6Month_buyPriceMin', 'user_relatedCate_last6Month_buyPriceMin',
         'user_cate101_last1Month_buyPriceMax', 'user_cate30_last1Month_buyPriceMax', 'user_targetCate_last1Month_buyPriceMax', 'user_relatedCate_last1Month_buyPriceMax',
         'user_cate101_last0.5Month_buyPriceMax', 'user_cate30_last0.5Month_buyPriceMax', 'user_targetCate_last0.5Month_buyPriceMax', 'user_relatedCate_last0.5Month_buyPriceMax',
         'user_cate101_last3Month_buyPriceMax', 'user_cate30_last3Month_buyPriceMax', 'user_targetCate_last3Month_buyPriceMax', 'user_relatedCate_last3Month_buyPriceMax',
         'user_cate101_last6Month_buyPriceMax', 'user_cate30_last6Month_buyPriceMax', 'user_targetCate_last6Month_buyPriceMax', 'user_relatedCate_last6Month_buyPriceMax',
         'user_cate101_last1Month_buyPriceMean', 'user_cate30_last1Month_buyPriceMean', 'user_targetCate_last1Month_buyPriceMean', 'user_relatedCate_last1Month_buyPriceMean',
         'user_cate101_last0.5Month_buyPriceMean', 'user_cate30_last0.5Month_buyPriceMean', 'user_targetCate_last0.5Month_buyPriceMean', 'user_relatedCate_last0.5Month_buyPriceMean',
         'user_cate101_last3Month_buyPriceMean', 'user_cate30_last3Month_buyPriceMean', 'user_targetCate_last3Month_buyPriceMean', 'user_relatedCate_last3Month_buyPriceMean',
         'user_cate101_last6Month_buyPriceMean', 'user_cate30_last6Month_buyPriceMean', 'user_targetCate_last6Month_buyPriceMean', 'user_relatedCate_last6Month_buyPriceMean',
         'user_cate101_last1Month_buyPriceSum', 'user_cate30_last1Month_buyPriceSum', 'user_targetCate_last1Month_buyPriceSum', 'user_relatedCate_last1Month_buyPriceSum',
         'user_cate101_last0.5Month_buyPriceSum', 'user_cate30_last0.5Month_buyPriceSum', 'user_targetCate_last0.5Month_buyPriceSum', 'user_relatedCate_last0.5Month_buyPriceSum',
         'user_cate101_last3Month_buyPriceSum', 'user_cate30_last3Month_buyPriceSum', 'user_targetCate_last3Month_buyPriceSum', 'user_relatedCate_last3Month_buyPriceSum',
         'user_cate101_last6Month_buyPriceSum', 'user_cate30_last6Month_buyPriceSum', 'user_targetCate_last6Month_buyPriceSum', 'user_relatedCate_last6Month_buyPriceSum',
         'user_cate101_last1Month_buyPara1Min', 'user_cate30_last1Month_buyPara1Min', 'user_targetCate_last1Month_buyPara1Min', 'user_relatedCate_last1Month_buyPara1Min',
         'user_cate101_last0.5Month_buyPara1Min', 'user_cate30_last0.5Month_buyPara1Min', 'user_targetCate_last0.5Month_buyPara1Min', 'user_relatedCate_last0.5Month_buyPara1Min',
         'user_cate101_last3Month_buyPara1Min', 'user_cate30_last3Month_buyPara1Min', 'user_targetCate_last3Month_buyPara1Min', 'user_relatedCate_last3Month_buyPara1Min',
         'user_cate101_last6Month_buyPara1Min', 'user_cate30_last6Month_buyPara1Min', 'user_targetCate_last6Month_buyPara1Min', 'user_relatedCate_last6Month_buyPara1Min',
         'user_cate101_last1Month_buyPara1Max', 'user_cate30_last1Month_buyPara1Max', 'user_targetCate_last1Month_buyPara1Max', 'user_relatedCate_last1Month_buyPara1Max',
         'user_cate101_last0.5Month_buyPara1Max', 'user_cate30_last0.5Month_buyPara1Max', 'user_targetCate_last0.5Month_buyPara1Max', 'user_relatedCate_last0.5Month_buyPara1Max',
         'user_cate101_last3Month_buyPara1Max', 'user_cate30_last3Month_buyPara1Max', 'user_targetCate_last3Month_buyPara1Max', 'user_relatedCate_last3Month_buyPara1Max',
         'user_cate101_last6Month_buyPara1Max', 'user_cate30_last6Month_buyPara1Max', 'user_targetCate_last6Month_buyPara1Max', 'user_relatedCate_last6Month_buyPara1Max',
         'user_cate101_last1Month_buyPara1Mean', 'user_cate30_last1Month_buyPara1Mean', 'user_targetCate_last1Month_buyPara1Mean', 'user_relatedCate_last1Month_buyPara1Mean',
         'user_cate101_last0.5Month_buyPara1Mean', 'user_cate30_last0.5Month_buyPara1Mean', 'user_targetCate_last0.5Month_buyPara1Mean', 'user_relatedCate_last0.5Month_buyPara1Mean',
         'user_cate101_last3Month_buyPara1Mean', 'user_cate30_last3Month_buyPara1Mean', 'user_targetCate_last3Month_buyPara1Mean', 'user_relatedCate_last3Month_buyPara1Mean',
         'user_cate101_last6Month_buyPara1Mean', 'user_cate30_last6Month_buyPara1Mean', 'user_targetCate_last6Month_buyPara1Mean', 'user_relatedCate_last6Month_buyPara1Mean',
         'user_cate101_last1Month_buyDayNumber', 'user_cate30_last1Month_buyDayNumber', 'user_targetCate_last1Month_buyDayNumber', 'user_relatedCate_last1Month_buyDayNumber',
         'user_cate101_last0.5Month_buyDayNumber', 'user_cate30_last0.5Month_buyDayNumber', 'user_targetCate_last0.5Month_buyDayNumber', 'user_relatedCate_last0.5Month_buyDayNumber',
         'user_cate101_last3Month_buyDayNumber', 'user_cate30_last3Month_buyDayNumber', 'user_targetCate_last3Month_buyDayNumber', 'user_relatedCate_last3Month_buyDayNumber',
         'user_cate101_last6Month_buyDayNumber', 'user_cate30_last6Month_buyDayNumber', 'user_targetCate_last6Month_buyDayNumber', 'user_relatedCate_last6Month_buyDayNumber',
         'user_cate101_last1Month_buyCount', 'user_cate30_last1Month_buyCount', 'user_targetCate_last1Month_buyCount', 'user_relatedCate_last1Month_buyCount',
         'user_cate101_last0.5Month_buyCount', 'user_cate30_last0.5Month_buyCount', 'user_targetCate_last0.5Month_buyCount', 'user_relatedCate_last0.5Month_buyCount',
         'user_cate101_last3Month_buyCount', 'user_cate30_last3Month_buyCount', 'user_targetCate_last3Month_buyCount', 'user_relatedCate_last3Month_buyCount',
         'user_cate101_last6Month_buyCount', 'user_cate30_last6Month_buyCount', 'user_targetCate_last6Month_buyCount', 'user_relatedCate_last6Month_buyCount',
         'user_cate101_last1Month_buyDayMin', 'user_cate30_last1Month_buyDayMin', 'user_targetCate_last1Month_buyDayMin', 'user_relatedCate_last1Month_buyDayMin',
         'user_cate101_last0.5Month_buyDayMin', 'user_cate30_last0.5Month_buyDayMin', 'user_targetCate_last0.5Month_buyDayMin', 'user_relatedCate_last0.5Month_buyDayMin',
         'user_cate101_last3Month_buyDayMin', 'user_cate30_last3Month_buyDayMin', 'user_targetCate_last3Month_buyDayMin', 'user_relatedCate_last3Month_buyDayMin',
         'user_cate101_last6Month_buyDayMin', 'user_cate30_last6Month_buyDayMin', 'user_targetCate_last6Month_buyDayMin','user_relatedCate_last6Month_buyDayMin',
         'user_cate101_last1Month_buyDayMax', 'user_cate30_last1Month_buyDayMax', 'user_targetCate_last1Month_buyDayMax', 'user_relatedCate_last1Month_buyDayMax',
         'user_cate101_last0.5Month_buyDayMax', 'user_cate30_last0.5Month_buyDayMax', 'user_targetCate_last0.5Month_buyDayMax', 'user_relatedCate_last0.5Month_buyDayMax',
         'user_cate101_last3Month_buyDayMax', 'user_cate30_last3Month_buyDayMax', 'user_targetCate_last3Month_buyDayMax', 'user_relatedCate_last3Month_buyDayMax',
         'user_cate101_last6Month_buyDayMax', 'user_cate30_last6Month_buyDayMax', 'user_targetCate_last6Month_buyDayMax', 'user_relatedCate_last6Month_buyDayMax',
         'user_cate101_last1Month_buyDayMean', 'user_cate30_last1Month_buyDayMean', 'user_targetCate_last1Month_buyDayMean', 'user_relatedCate_last1Month_buyDayMean',
         'user_cate101_last0.5Month_buyDayMean', 'user_cate30_last0.5Month_buyDayMean', 'user_targetCate_last0.5Month_buyDayMean', 'user_relatedCate_last0.5Month_buyDayMean',
         'user_cate101_last3Month_buyDayMean', 'user_cate30_last3Month_buyDayMean', 'user_targetCate_last3Month_buyDayMean', 'user_relatedCate_last3Month_buyDayMean',
         'user_cate101_last6Month_buyDayMean', 'user_cate30_last6Month_buyDayMean', 'user_targetCate_last6Month_buyDayMean', 'user_relatedCate_last6Month_buyDayMean',
         'user_cate101_last1Month_buyMonthNumber', 'user_cate30_last1Month_buyMonthNumber', 'user_targetCate_last1Month_buyMonthNumber', 'user_relatedCate_last1Month_buyMonthNumber',
         'user_cate101_last0.5Month_buyMonthNumber', 'user_cate30_last0.5Month_buyMonthNumber', 'user_targetCate_last0.5Month_buyMonthNumber', 'user_relatedCate_last0.5Month_buyMonthNumber',
         'user_cate101_last3Month_buyMonthNumber', 'user_cate30_last3Month_buyMonthNumber', 'user_targetCate_last3Month_buyMonthNumber', 'user_relatedCate_last3Month_buyMonthNumber',
         'user_cate101_last6Month_buyMonthNumber', 'user_cate30_last6Month_buyMonthNumber', 'user_targetCate_last6Month_buyMonthNumber', 'user_relatedCate_last6Month_buyMonthNumber',
         'user_cate101_last1Month_skuActionNumber', 'user_cate30_last1Month_skuActionNumber', 'user_targetCate_last1Month_skuActionNumber', 'user_relatedCate_last1Month_skuActionNumber',
         'user_cate101_last0.5Month_skuActionNumber', 'user_cate30_last0.5Month_skuActionNumber', 'user_targetCate_last0.5Month_skuActionNumber', 'user_relatedCate_last0.5Month_skuActionNumber',
         'user_cate101_last3Month_skuActionNumber', 'user_cate30_last3Month_skuActionNumber', 'user_targetCate_last3Month_skuActionNumber', 'user_relatedCate_last3Month_skuActionNumber',
         'user_cate101_last6Month_skuActionNumber', 'user_cate30_last6Month_skuActionNumber', 'user_targetCate_last6Month_skuActionNumber', 'user_relatedCate_last6Month_skuActionNumber',
         'user_cate101_last1Month_skuActionDayNumber', 'user_cate30_last1Month_skuActionDayNumber', 'user_targetCate_last1Month_skuActionDayNumber', 'user_relatedCate_last1Month_skuActionDayNumber',
         'user_cate101_last0.5Month_skuActionDayNumber', 'user_cate30_last0.5Month_skuActionDayNumber', 'user_targetCate_last0.5Month_skuActionDayNumber', 'user_relatedCate_last0.5Month_skuActionDayNumber',
         'user_cate101_last3Month_skuActionDayNumber', 'user_cate30_last3Month_skuActionDayNumber', 'user_targetCate_last3Month_skuActionDayNumber', 'user_relatedCate_last3Month_skuActionDayNumber',
         'user_cate101_last6Month_skuActionDayNumber', 'user_cate30_last6Month_skuActionDayNumber', 'user_targetCate_last6Month_skuActionDayNumber', 'user_relatedCate_last6Month_skuActionDayNumber',
         'user_cate101_last1Month_actionNumber', 'user_cate30_last1Month_actionNumber', 'user_targetCate_last1Month_actionNumber', 'user_relatedCate_last1Month_actionNumber',
         'user_cate101_last0.5Month_actionNumber', 'user_cate30_last0.5Month_actionNumber', 'user_targetCate_last0.5Month_actionNumber', 'user_relatedCate_last0.5Month_actionNumber',
         'user_cate101_last3Month_actionNumber', 'user_cate30_last3Month_actionNumber', 'user_targetCate_last3Month_actionNumber', 'user_relatedCate_last3Month_actionNumber',
         'user_cate101_last6Month_actionNumber', 'user_cate30_last6Month_actionNumber', 'user_targetCate_last6Month_actionNumber', 'user_relatedCate_last6Month_actionNumber',
         'user_cate101_last1Month_skuBrowseNumber', 'user_cate30_last1Month_skuBrowseNumber', 'user_targetCate_last1Month_skuBrowseNumber', 'user_relatedCate_last1Month_skuBrowseNumber',
         'user_cate101_last0.5Month_skuBrowseNumber', 'user_cate30_last0.5Month_skuBrowseNumber', 'user_targetCate_last0.5Month_skuBrowseNumber', 'user_relatedCate_last0.5Month_skuBrowseNumber',
         'user_cate101_last3Month_skuBrowseNumber', 'user_cate30_last3Month_skuBrowseNumber', 'user_targetCate_last3Month_skuBrowseNumber', 'user_relatedCate_last3Month_skuBrowseNumber',
         'user_cate101_last6Month_skuBrowseNumber', 'user_cate30_last6Month_skuBrowseNumber', 'user_targetCate_last6Month_skuBrowseNumber', 'user_relatedCate_last6Month_skuBrowseNumber',
         'user_cate101_last1Month_skuBrowseDayNumber', 'user_cate30_last1Month_skuBrowseDayNumber', 'user_targetCate_last1Month_skuBrowseDayNumber', 'user_relatedCate_last1Month_skuBrowseDayNumber',
         'user_cate101_last0.5Month_skuBrowseDayNumber', 'user_cate30_last0.5Month_skuBrowseDayNumber', 'user_targetCate_last0.5Month_skuBrowseDayNumber', 'user_relatedCate_last0.5Month_skuBrowseDayNumber',
         'user_cate101_last3Month_skuBrowseDayNumber', 'user_cate30_last3Month_skuBrowseDayNumber', 'user_targetCate_last3Month_skuBrowseDayNumber', 'user_relatedCate_last3Month_skuBrowseDayNumber',
         'user_cate101_last6Month_skuBrowseDayNumber', 'user_cate30_last6Month_skuBrowseDayNumber', 'user_targetCate_last6Month_skuBrowseDayNumber', 'user_relatedCate_last6Month_skuBrowseDayNumber',
         'user_cate101_last1Month_browseNumber', 'user_cate30_last1Month_browseNumber', 'user_targetCate_last1Month_browseNumber', 'user_relatedCate_last1Month_browseNumber',
         'user_cate101_last0.5Month_browseNumber', 'user_cate30_last0.5Month_browseNumber', 'user_targetCate_last0.5Month_browseNumber', 'user_relatedCate_last0.5Month_browseNumber',
         'user_cate101_last3Month_browseNumber', 'user_cate30_last3Month_browseNumber', 'user_targetCate_last3Month_browseNumber', 'user_relatedCate_last3Month_browseNumber',
         'user_cate101_last6Month_browseNumber', 'user_cate30_last6Month_browseNumber', 'user_targetCate_last6Month_browseNumber', 'user_relatedCate_last6Month_browseNumber',
         'user_cate101_last1Month_skuFocusNumber', 'user_cate30_last1Month_skuFocusNumber', 'user_targetCate_last1Month_skuFocusNumber', 'user_relatedCate_last1Month_skuFocusNumber',
         'user_cate101_last0.5Month_skuFocusNumber', 'user_cate30_last0.5Month_skuFocusNumber', 'user_targetCate_last0.5Month_skuFocusNumber', 'user_relatedCate_last0.5Month_skuFocusNumber',
         'user_cate101_last3Month_skuFocusNumber', 'user_cate30_last3Month_skuFocusNumber', 'user_targetCate_last3Month_skuFocusNumber', 'user_relatedCate_last3Month_skuFocusNumber',
         'user_cate101_last6Month_skuFocusNumber', 'user_cate30_last6Month_skuFocusNumber', 'user_targetCate_last6Month_skuFocusNumber', 'user_relatedCate_last6Month_skuFocusNumber',
         'user_cate101_last1Month_focusNumber', 'user_cate30_last1Month_focusNumber', 'user_targetCate_last1Month_focusNumber', 'user_relatedCate_last1Month_focusNumber',
         'user_cate101_last0.5Month_focusNumber', 'user_cate30_last0.5Month_focusNumber', 'user_targetCate_last0.5Month_focusNumber', 'user_relatedCate_last0.5Month_focusNumber',
         'user_cate101_last3Month_focusNumber', 'user_cate30_last3Month_focusNumber', 'user_targetCate_last3Month_focusNumber', 'user_relatedCate_last3Month_focusNumber',
         'user_cate101_last6Month_focusNumber', 'user_cate30_last6Month_focusNumber', 'user_targetCate_last6Month_focusNumber', 'user_relatedCate_last6Month_focusNumber',
         'user_score1_last1Month_commentNumber', 'user_score2_last1Month_commentNumber', 'user_score3_last1Month_commentNumber', 'user_score0_last1Month_commentNumber',
         'user_score1_last0.5Month_commentNumber', 'user_score2_last0.5Month_commentNumber', 'user_score3_last0.5Month_commentNumber', 'user_score0_last0.5Month_commentNumber',
         'user_score1_last3Month_commentNumber', 'user_score2_last3Month_commentNumber', 'user_score3_last3Month_commentNumber', 'user_score0_last3Month_commentNumber',
         'user_score1_last6Month_commentNumber', 'user_score2_last6Month_commentNumber', 'user_score3_last6Month_commentNumber', 'user_score0_last6Month_commentNumber',
         'user_101Cate_monthlyBuyNumber', 'user_30Cate_monthlyBuyNumber', 'user_targetCate_monthlyBuyNumber',
         'user_relatedCate_monthlyBuyNumber', 'user_101Cate_monthlyOrderNumber',
         'user_30Cate_monthlyOrderNumber', 'user_targetCate_monthlyOrderNumber',
         'user_relatedCate_monthlyOrderNumber', 'user_101Cate_monthlyBuyCount',
         'user_30Cate_monthlyBuyCount', 'user_targetCate_monthlyBuyCount',
         'user_relatedCate_monthlyBuyCount', 'user_101Cate_monthlyBuyDay',
         'user_30Cate_monthlyBuyDay', 'user_targetCate_monthlyBuyDay',
         'user_relatedCate_monthlyBuyDay', 'user_101Cate_monthlyBuyPriceMin',
         'user_30Cate_monthlyBuyPriceMin', 'user_targetCate_monthlyBuyPriceMin',
         'user_relatedCate_monthlyBuyPriceMin', 'user_101Cate_monthlyBuyPriceMax',
         'user_30Cate_monthlyBuyPriceMax', 'user_targetCate_monthlyBuyPriceMax',
         'user_relatedCate_monthlyBuyPriceMax', 'user_101Cate_monthlyBuyPriceMean',
         'user_30Cate_monthlyBuyPriceMean', 'user_targetCate_monthlyBuyPriceMean',
         'user_relatedCate_monthlyBuyPriceMean', 'user_101Cate_monthlyBuyPara1Min',
         'user_30Cate_monthlyBuyPara1Min', 'user_targetCate_monthlyBuyPara1Min',
         'user_relatedCate_monthlyBuyPara1Min', 'user_101Cate_monthlyBuyPara1Max',
         'user_30Cate_monthlyBuyPara1Max', 'user_targetCate_monthlyBuyPara1Max',
         'user_relatedCate_monthlyBuyPara1Max', 'user_101Cate_monthlyBuyPara1Mean',
         'user_30Cate_monthlyBuyPara1Mean', 'user_targetCate_monthlyBuyPara1Mean',
         'user_relatedCate_monthlyBuyPara1Mean', 'user_cate101_monthlyFocusNumber',
         'user_cate30_monthlyFocusNumber', 'user_targetCate_monthlyFocusNumber',
         'user_relatedCate_monthlyFocusNumber', 'user_cate101_monthlyBrowseNumber',
         'user_cate30_monthlyBrowseNumber', 'user_targetCate_monthlyBrowseNumber',
         'user_relatedCate_monthlyBrowseNumber', 'user_cate101_monthlyActionNumber',
         'user_cate30_monthlyActionNumber', 'user_targetCate_monthlyActionNumber',
         'user_relatedCate_monthlyActionNumber',
         'user_cate101_monthlyActionDayNumber', 'user_cate30_monthlyActionDayNumber',
         'user_targetCate_monthlyActionDayNumber',
         'user_relatedCate_monthlyActionDayNumber',
         'user_cate101_monthlyActionSkuCount', 'user_cate30_monthlyActionSkuCount',
         'user_targetCate_monthlyActionSkuCount',
         'user_relatedCate_monthlyActionSkuCount',

        ]

    train_dataset_x = jdata_train_df[(jdata_train_df.month <= 8) & (jdata_train_df.month >= 3)]
    train_dataset_y = jdata_train_df['is_order'][(jdata_train_df.month <= 8) & (jdata_train_df.month >= 3)]

    xgbModel = XgbModel(feaNames=fea)
    modelName = "xgb_b_keng"

    xgbModel.trainCV(train_dataset_x[fea].values, train_dataset_y.values)
    xgbModel.getFeaScore(show=True)
    jdata_test_df.loc[:,'predicted_is_order'] = xgbModel.predict(jdata_test_df[fea].values)
    jdata_test_df = jdata_test_df.sort_index(by='predicted_is_order', ascending=False)
    order_jdata_test_df = jdata_test_df[['user_id', 'predicted_is_order']]

    exportDf(order_jdata_test_df[['user_id', 'predicted_is_order']], 'jdata_b_keng_result2')
    print('~~~~~~~~~~~~~~训练完毕~~~~~~~~~~~~~~~~~~~')
    print('all time:', datetime.datetime.now()-startTime)

if __name__ == '__main__':
    main()
