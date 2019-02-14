#!/usr/bin/env python
# -*-coding:utf-8-*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import csv
from datetime import *
import math, os

from sklearn.preprocessing import *
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
import xgboost as xgb
from sklearn.externals import joblib

from yuna import yuna_feaFactory2

# 导入数据
def importDf(url, sep=',', na_values='-1', header='infer', index_col=None, colNames=None):
    df = pd.read_csv(url, sep=sep, na_values='-1', header=header, index_col=index_col, names=colNames)
    return df

# 缩放字段至0-1
def scalerFea(df, cols):
    df.dropna(inplace=True, subset=cols)
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols].values)
    return df,scaler

# 矩估计法计算贝叶斯平滑参数
def countBetaParamByMME(inputArr):
    EX = inputArr.mean()
    EX2 = (inputArr ** 2).mean()
    alpha = (EX*(EX-EX2)) / (EX2 - EX**2)
    beta = alpha * (1/EX - 1)
    return alpha,beta

# 对numpy数组进行贝叶斯平滑处理
def biasSmooth(aArr, bArr, method='MME', alpha=None, beta=None):
    ratioArr = aArr / bArr
    if method=='MME':
        alpha,beta = countBetaParamByMME(ratioArr[ratioArr==ratioArr])
    resultArr = (aArr+alpha) / (bArr+alpha+beta)
    return resultArr

# 导出预测结果
def exportResult(df, fileName, header=True, index=False, sep=','):
    df.to_csv('./%s' % fileName, sep=sep, header=header, index=index)

# 格式化数据集
def timeTransform(df, col):
    df.loc[:,col] = pd.to_datetime(df[col])
    return df

# 按日期统计过去几天的数目，总和
def statDateLenSum(df, index, values, columns='date_range', colVals=[], statLen=None, skipLen=1, stepLen=1):
    if len(colVals)==0:
        colVals=pd.date_range(start='2016-09-01',end='2017-09-01', freq='D')
    tempDf = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=[len,np.sum])
    for dt in np.setdiff1d(colVals,tempDf['len'].columns):
        tempDf.loc[:, pd.IndexSlice['len',dt]] = np.nan
    for dt in np.setdiff1d(colVals,tempDf['sum'].columns):
        tempDf.loc[:, pd.IndexSlice['sum',dt]] = np.nan
    tempDf.sort_index(axis=1, inplace=True)
    idxList = [i for i in range(skipLen,len(tempDf.columns.levels[-1])) if (i-skipLen)%stepLen==0]
    if statLen==None:
        for i in idxList:
            dt = tempDf.columns.levels[-1][i]
            tempDf.loc[:,pd.IndexSlice['addup_len',dt]] = tempDf['len'].iloc[:,:i].sum(axis=1)
            tempDf.loc[:,pd.IndexSlice['addup_sum',dt]] = tempDf['sum'].iloc[:,:i].sum(axis=1)
    else:
        for i in idxList:
            dt = tempDf.columns.levels[-1][i]
            tempDf.loc[:,pd.IndexSlice['addup_len',dt]] = tempDf['len'].iloc[:,i-statLen:i].sum(axis=1)
            tempDf.loc[:,pd.IndexSlice['addup_sum',dt]] = tempDf['sum'].iloc[:,i-statLen:i].sum(axis=1)
    tempDf.columns.names = ['type','date_range']
    tempDf = tempDf[['addup_len','addup_sum']].stack()
    return tempDf

class FeaFactory():
    def __init__(self, dfs):
        startTime = datetime.now()
        dfs = self.dataFormatter(dfs)
        dfs['action_df'].dropna(subset=['sku_id'], inplace=True)
        dfs['action_df'].sku_id.astype(int)

        dfs['order_df']['year'] = dfs['order_df'].o_date.dt.year
        dfs['order_df']['month'] = dfs['order_df'].o_date.dt.month
        dfs['order_df']['day'] = dfs['order_df'].o_date.dt.day
        dfs['order_df']['day_of_month_end'] = pd.Index(dfs['order_df']['o_date']).shift(1,freq='MS')
        dfs['order_df']['day_of_month_end'] = (dfs['order_df']['day_of_month_end'] - dfs['order_df']['o_date']).dt.days
        dfs['order_df'] = dfs['order_df'].merge(dfs['sku_df'], how='left', on='sku_id')
        dfs['order_df']['sku_price_sum'] = dfs['order_df']['price'] * dfs['order_df']['o_sku_num']
        tempDf = pd.pivot_table(dfs['order_df'], index=['o_id','cate'], values='sku_price_sum', aggfunc=np.sum)
        dfs['order_df'] = dfs['order_df'].merge(tempDf.rename(columns={'sku_price_sum':'o_cate_price_sum'}), how='left', left_on=['o_id','cate'], right_index=True)
        tempDf = tempDf['sku_price_sum'].unstack('cate')
        tempDf['o_price_sum'] = tempDf.sum(axis=1)
        dfs['order_df'] = dfs['order_df'].merge(tempDf[['o_price_sum']], how='left', left_on='o_id', right_index=True)

        dfs['action_df']['year'] = dfs['action_df'].a_date.dt.year
        dfs['action_df']['month'] = dfs['action_df'].a_date.dt.month
        dfs['action_df']['day'] = dfs['action_df'].a_date.dt.day
        dfs['action_df']['day_of_month_end'] = pd.Index(dfs['action_df']['a_date']).shift(1,freq='MS')
        dfs['action_df']['day_of_month_end'] = (dfs['action_df']['day_of_month_end'] - dfs['action_df']['a_date']).dt.days
        dfs['action_df'] = dfs['action_df'].merge(dfs['sku_df'], how='left', on='sku_id')

        self.userDf = dfs['user_df']
        self.skuDf = dfs['sku_df']
        self.actionDf = dfs['action_df']
        self.orderDf = dfs['order_df']
        self.commDf = dfs['comm_df']

        self.startDate = dfs['action_df'].a_date.min()
        self.endDate = dfs['action_df'].a_date.max()

        tempDf = pd.pivot_table(self.commDf, index=['o_id'], values=['score_level','comment_create_tm'], aggfunc={'score_level':np.mean, 'comment_create_tm':np.max})
        orderCommDf = self.orderDf.merge(tempDf, how='left', left_on='o_id', right_index=True)
        self.orderCommDf = orderCommDf

        cateList = set(self.skuDf.cate)
        taskCate = set([101,30])
        otherCate = cateList - taskCate
        self.cateList = list(cateList)
        self.taskCate = list(taskCate)
        self.otherCate = list(otherCate)
        print('init fea featory:', datetime.now() - startTime)

    # 数据格式化
    def dataFormatter(self, dfs):
        dfs['action_df'] = timeTransform(dfs['action_df'], 'a_date')
        dfs['order_df'] = timeTransform(dfs['order_df'], 'o_date')
        dfs['comm_df'] = timeTransform(dfs['comm_df'], 'comment_create_tm')
        dfs['user_df'].loc[dfs['user_df'].sex==2, 'sex'] = np.nan
        return dfs

    # 初始化数据集
    def initDf(self, dateList, predictLen=31, **params):
        startTime = datetime.now()
        endDate = dateList[-1]
        freq = (dateList[1] - dateList[0]).days

        # 构建index
        userList = self.userDf['user_id'].values
        tempIdx = pd.MultiIndex.from_product([userList,dateList], names=['user_id', 'date_range'])
        df = pd.DataFrame(index=tempIdx)
        df.reset_index(inplace=True)

        # 替原始数据集划分时间区间
        self.actionDf['a_date_id'] = (self.actionDf['a_date'] - endDate).dt.days
        self.actionDf['date_range_id'] = self.actionDf['a_date_id'] // freq
        self.actionDf['date_range'] = endDate + pd.to_timedelta(self.actionDf['date_range_id']*freq, unit='d')
        self.orderDf['o_date_id'] = (self.orderDf['o_date'] - endDate).dt.days
        self.orderDf['date_range_id'] = self.orderDf['o_date_id'] // freq
        self.orderDf['date_range'] = endDate + pd.to_timedelta(self.orderDf['date_range_id']*freq, unit='d')

        # 剔除历史未购买的用户
        tempDf = pd.pivot_table(self.actionDf, index='user_id', values='a_date', aggfunc=np.min)
        tempDf.columns = ['user_first_date']
        df = df.merge(tempDf, how='left', left_on='user_id', right_index=True)
        tempDf = pd.pivot_table(self.orderDf, index='user_id', values='o_date', aggfunc=np.min)
        tempDf.columns = ['user_first_order']
        df = df.merge(tempDf, how='left', left_on='user_id', right_index=True)
        df.loc[df.user_first_date.isnull(),'user_first_date'] = df.loc[df.user_first_date.isnull(),'user_first_order']
        tempIdx = df[df.date_range<=df.user_first_order].index
        df.drop(tempIdx, inplace=True)

        # 计算数据集label
        skuList = self.skuDf[self.skuDf.cate.isin([101,30])]['sku_id'].values
        labelDf = pd.DataFrame(index=userList)
        for dt in dateList[:-1]:
            tempDf = self.orderDf[(self.orderDf.sku_id.isin(skuList))&(self.orderDf.o_date>=dt)&(self.orderDf.o_date<dt+timedelta(days=predictLen))]
            tempDf = pd.pivot_table(tempDf, index=['user_id'], values=['o_date'], aggfunc=np.min)
            labelDf[dt] = tempDf
        df = df.merge(labelDf.stack().to_frame().rename(columns={0:'day_label'}), how='left', left_on=['user_id','date_range'], right_index=True)
        df['day_label'] = (df['day_label'] - df['date_range']).dt.days
        df['buy_label'] = df['day_label'].notnull().astype(int)

        print('init df:', datetime.now() - startTime)
        return df

    # 用户基本信息
    def addUserInfo(self, df, **params):
        df = df.merge(self.userDf, how='left', on='user_id')
        df['user_his_days'] = (df['date_range'] - df['user_first_date']).dt.days
        return df

    # 距离上次行为时间
    def addLastActionTime(self, df, **params):
        startTime = datetime.now()
        tempDf = pd.pivot_table(self.actionDf, index=['date_range'], columns=['user_id','cate','a_type'], values='a_date', aggfunc=np.max)
        if params['dateList'][-1] not in tempDf.index:
            tempDf.loc[params['dateList'][-1], :] = np.nan
        tempDf = tempDf.shift(1)
        tempDf.fillna(method='ffill', inplace=True)
        tempDf = tempDf.stack(level=['user_id','a_type'])
        tempDf['all'] = tempDf.max(axis=1)
        tempDf['task'] = tempDf[self.taskCate].max(axis=1)
        tempDf['other'] = tempDf[self.otherCate].max(axis=1)
        tempDf = tempDf.stack(level=['cate'])
        tempDf = (tempDf.index.get_level_values('date_range').values - tempDf).dt.days
        tempDf.index.set_levels(['view','follow'], level='a_type', inplace=True)
        tempDf = tempDf.unstack(level=['cate','a_type'])
        tempDf.columns = ['user_cate%s_last_%s_timedelta'%(x[0],x[1]) for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['date_range','user_id'], right_index=True)
        df.fillna({k:999 for k in tempDf.columns.values}, inplace=True)

        tempDf = pd.pivot_table(self.orderDf, index=['date_range'], columns=['user_id','cate'], values='o_date', aggfunc=np.max)
        if params['dateList'][-1] not in tempDf.index:
            tempDf.loc[params['dateList'][-1], :] = np.nan
        tempDf = tempDf.shift(1)
        tempDf.fillna(method='ffill', inplace=True)
        tempDf = tempDf.stack(level=['user_id'])
        tempDf['all'] = tempDf.max(axis=1)
        tempDf['task'] = tempDf[self.taskCate].max(axis=1)
        tempDf['other'] = tempDf[self.otherCate].max(axis=1)
        df = df.merge(tempDf.rename(columns={x:'user_cate%s_last_order_time'%x for x in tempDf.columns}), how='left', left_on=['date_range','user_id'], right_index=True)
        tempDf = tempDf.stack(level=['cate'])
        tempDf = (tempDf.index.get_level_values('date_range').values - tempDf).dt.days
        tempDf = tempDf.unstack(level=['cate'])
        tempDf.columns = ['user_cate%s_last_order_timedelta'%x for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['date_range','user_id'], right_index=True)
        df.fillna({k:999 for k in tempDf.columns.values}, inplace=True)
        df.drop(df[df.user_cateall_last_order_timedelta>31*3].index, inplace=True)  # 根据赛题说明删除3个月内无购买行为的用户
        for x in self.cateList+['all']:
            df['user_cate%s_last_order_view_timedelta'%x] = df['user_cate%s_last_order_timedelta'%x] - df['user_cate%s_last_view_timedelta'%x]
            df['user_cate%s_last_order_follow_timedelta'%x] = df['user_cate%s_last_order_timedelta'%x] - df['user_cate%s_last_follow_timedelta'%x]
        print('user last timedelta:', datetime.now() - startTime)
        return df

    # 最近一次订单id
    def addLastOrderId(self, df, **params):
        startTime = datetime.now()
        for x in self.cateList+['all']:
            df['user_cate%s_last_order_time'%x] = df['date_range'] - pd.to_timedelta(df['user_cate%s_last_order_timedelta'%x], unit='d')
            if x=='all':
                tempDf = self.orderCommDf.drop_duplicates(['user_id','o_date'])
            else:
                tempDf = self.orderCommDf[self.orderCommDf.cate==x].drop_duplicates(['user_id','o_date'])
            df = df.merge(tempDf[['user_id','o_date','o_id']].rename(columns={'o_id':'cate%s_last_order'%x}), how='left', left_on=['user_id','user_cate%s_last_order_time'%x],right_on=['user_id','o_date'])
            del df['o_date']
        print('last order id:', datetime.now()-startTime)
        return df

    # 距离特殊日期的天数
    def addSpecialDatedelta(self, df, **params):
        startTime = datetime.now()
        df['date1111_daydelta'] = (df['date_range'] - date(2016,11,11)).dt.days
        df['date618_daydelta'] = (df['date_range'] - date(2016,6,18)).dt.days
        print('special date daydelta:', datetime.now()-startTime)
        return df

    # 最近一次订单评价记录
    def addLastComm(self, df, **params):
        startTime = datetime.now()
        for x in self.cateList+['all']:
            if x=='all':
                tempDf = self.orderCommDf.drop_duplicates(['user_id','o_date'])
            else:
                tempDf = self.orderCommDf[self.orderCommDf.cate==x].drop_duplicates(['user_id','o_date'])
            df = df.merge(tempDf[['user_id','o_date','o_id','comment_create_tm','score_level']], how='left', left_on=['user_id','user_cate%s_last_order_time'%x],right_on=['user_id','o_date'])
            del df['o_date']
            df.loc[df.comment_create_tm>df.date_range, ['comment_create_tm','score_level']] = np.nan
            df.rename(columns={'o_id':'cate%s_last_order'%x,'comment_create_tm':'cate%s_last_order_comm_time'%x,'score_level':'cate%s_last_order_comm'%x}, inplace=True)
        print('last order comment:', datetime.now()-startTime)
        return df

    # 最近一次订单价格
    def addLastOrderPrice(self, df, **params):
        startTime = datetime.now()
        tempDf = self.orderDf.drop_duplicates(['o_id'])[['o_id','o_price_sum']].rename(columns={'o_price_sum':'user_cateall_last_order_price'})
        df = df.merge(tempDf, how='left', left_on='cateall_last_order', right_on='o_id')
        del df['o_id']
        for x in self.cateList:
            tempDf = self.orderDf[self.orderDf.cate==x][['o_id','o_price_sum']].drop_duplicates(['o_id']).rename(columns={'o_price_sum':'user_cate%s_last_order_price'%x})
            df = df.merge(tempDf, how='left', left_on='cate%s_last_order'%x, right_on='o_id')
            del df['o_id']
        print('last order price:', datetime.now()-startTime)
        return df

    # 最近一次订单参数1均值
    def addLastOrderPara1(self, df, **params):
        startTime = datetime.now()
        tempDf = pd.pivot_table(self.orderDf, index=['o_id','cate'], values=['para_1'], aggfunc=np.mean)
        tempDf.reset_index(inplace=True)
        for x in self.cateList:
            df = df.merge(tempDf[tempDf.cate==x][['o_id','para_1']].rename(columns={'para_1':'user_cate%s_last_order_para1'%x}), how='left', left_on=['cate%s_last_order'%x], right_on=['o_id'])
            del df['o_id']
        print('last order para1:', datetime.now()-startTime)
        return df

    # 统计浏览与关注的次数
    def addActionTimes(self, df, dayLen=None, feaFlag=None, **params):
        startTime = datetime.now()
        tempDf = statDateLenSum(self.actionDf, index=['user_id','cate','a_type'], values='a_num', columns='a_date', colVals=params['statDateVals'], statLen=dayLen, skipLen=params['skipLen'], stepLen=params['stepLen'])
        del tempDf['addup_len']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.droplevel()
        tempDf.fillna(0,inplace=True)
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.set_levels(['view','follow'], level='a_type')
        tempDf.columns = ['user_cate%s_%s_%s'%(x[0],feaFlag,x[1]) for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
        df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        if dayLen==None:
            for k in tempDf.columns.values:
                df[k] = df[k] / df['user_his_days']
        print('stat %s view follow times:'%feaFlag, datetime.now() - startTime)
        return df

    # 统计浏览与关注的天数
    def addActionDays(self, df, dayLen=None, feaFlag=None, **params):
        startTime = datetime.now()
        tempDf = statDateLenSum(self.actionDf.drop_duplicates(subset=['user_id','cate','a_type','a_date']), index=['user_id','cate','a_type'], values='a_num', columns='a_date', colVals=params['statDateVals'], statLen=dayLen, skipLen=params['skipLen'], stepLen=params['stepLen'])
        tempDf = tempDf['addup_len'].unstack(level='cate')
        tempDf.fillna(0,inplace=True)
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf = tempDf.rename(index={1:'view', 2:'follow'}, level='a_type').unstack(level='a_type')
        tempDf.columns = ['user_cate%s_%s_%sday'%(cate,feaFlag,type) for cate,type in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
        df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        if dayLen==None:
            for k in tempDf.columns.values:
                df[k] = df[k] / df['user_his_days']
        print('stat %s view follow days:'%feaFlag, datetime.now() - startTime)
        return df

    # 统计订单数与平均下单日
    def addOrderCountAndDay(self, df, dayLen=None, feaFlag=None, **params):
        startTime = datetime.now()
        tempDf = statDateLenSum(self.orderDf.drop_duplicates(subset=['o_id','cate']), index=['user_id','cate'], values='day', columns='o_date', colVals=params['statDateVals'], statLen=dayLen, skipLen=params['skipLen'], stepLen=params['stepLen'])
        tempDf = tempDf.stack(level=['type']).unstack(level='cate')
        tempDf.fillna(0,inplace=True)
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf = tempDf.rename(columns=str).stack().unstack(level='type').rename(columns={'addup_len':'len','addup_sum':'sum'})
        tempDf['mean'] = tempDf['sum'] / tempDf['len']
        del tempDf['sum']
        tempDf = tempDf.unstack(level='cate')
        tempDf.columns = [('user_cate%s_%s_order'%(cate,feaFlag) if type=='len' else 'user_cate%s_%s_order_daymean'%(cate,feaFlag)) for type,cate in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
        df.fillna({'user_cate%s_%s_order'%(k,feaFlag):0 for k in self.cateList}, inplace=True)
        if dayLen==None:
            for k in self.cateList:
                df['user_cate%s_%s_order'%(k,feaFlag)] = df['user_cate%s_%s_order'%(k,feaFlag)] / df['user_his_days']
        print('stat %s order times & order day mean:'%feaFlag, datetime.now() - startTime)
        return df

    # 统计下单天数
    def addOrderDays(self, df, dayLen=None, feaFlag=None, **params):
        startTime = datetime.now()
        tempDf = statDateLenSum(self.orderDf.drop_duplicates(subset=['user_id','cate','o_date']), index=['user_id','cate'], values='o_sku_num', columns='o_date', colVals=params['statDateVals'], statLen=dayLen, skipLen=params['skipLen'], stepLen=params['stepLen'])
        del tempDf['addup_sum']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.droplevel()
        tempDf.fillna(0,inplace=True)
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf.columns = ['user_cate%s_%s_orderday'%(x,feaFlag) for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
        df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        if dayLen==None:
            for k in tempDf.columns.values:
                df[k] = df[k] / df['user_his_days']
        print('stat %s order days:'%feaFlag, datetime.now() - startTime)
        return df

    # 统计下单商品数及商品件数
    def addOrderSku(self, df, dayLen=None, feaFlag=None, **params):
        startTime = datetime.now()
        tempDf = statDateLenSum(self.orderDf, index=['user_id','cate'], values='o_sku_num', columns='o_date', colVals=params['statDateVals'], statLen=dayLen, skipLen=params['skipLen'], stepLen=params['stepLen'])
        tempDf = tempDf.stack().unstack(level='cate')
        tempDf.fillna(0,inplace=True)
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf = tempDf.rename(index={'addup_len':'len','addup_sum':'sum'}, level='type').unstack(level='type')
        tempDf.columns = ['user_cate%s_%s_ordersku_%s'%(cate,feaFlag,type) for cate,type in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
        df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        if dayLen==None:
            for k in tempDf.columns.values:
                df[k] = df[k] / df['user_his_days']
        print('stat %s order sku & sku num:'%feaFlag, datetime.now() - startTime)
        return df

    # 统计最早下单日和最晚下单日
    def addOrderDayMinMax(self, df, dayLen=None, feaFlag=None, **params):
        startTime = datetime.now()
        tempDf = self.orderDf.drop_duplicates(['user_id','cate','o_date']).set_index(['user_id','cate','o_date'])[['day']].unstack('o_date')
        for dt in np.setdiff1d(params['statDateVals'],tempDf.columns.levels[-1]):
            tempDf.loc[:, pd.IndexSlice['day',dt]] = np.nan
        tempDf.sort_index(axis=1, inplace=True)
        idxList = [i for i in range(params['skipLen'],len(tempDf.columns.levels[-1])) if (i-params['skipLen'])%params['stepLen']==0]
        for i in idxList:
            if dayLen!=None:
                dt = tempDf.columns.levels[-1][i]
                tempDf.loc[:,pd.IndexSlice['min',dt]] = tempDf['day'].iloc[:,i-dayLen:i].min(axis=1)
                tempDf.loc[:,pd.IndexSlice['max',dt]] = tempDf['day'].iloc[:,i-dayLen:i].max(axis=1)
            else:
                dt = tempDf.columns.levels[-1][i]
                tempDf.loc[:,pd.IndexSlice['min',dt]] = tempDf['day'].iloc[:,:i].min(axis=1)
                tempDf.loc[:,pd.IndexSlice['max',dt]] = tempDf['day'].iloc[:,:i].max(axis=1)
        tempDf.columns.names = ['type','date_range']
        tempDf = tempDf.stack(level=['date_range'])[['min','max']].unstack('cate')
        tempDf.loc[:,pd.IndexSlice['min','all']] = tempDf['min'].min(axis=1)
        tempDf.loc[:,pd.IndexSlice['max','all']] = tempDf['max'].max(axis=1)
        tempDf.loc[:,pd.IndexSlice['min','task']] = tempDf['min'][self.taskCate].min(axis=1)
        tempDf.loc[:,pd.IndexSlice['max','task']] = tempDf['max'][self.taskCate].max(axis=1)
        tempDf.loc[:,pd.IndexSlice['min','other']] = tempDf['min'][self.otherCate].min(axis=1)
        tempDf.loc[:,pd.IndexSlice['max','other']] = tempDf['max'][self.otherCate].max(axis=1)
        tempDf.columns = ['user_cate%s_%s_order_day%s'%(cate,feaFlag,type) for type,cate in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
        print('stat %s order min & max day:'%feaFlag, datetime.now() - startTime)
        return df

    # 统计用户历史评价
    def addUserComm(self, df, dayLen=None, feaFlag=None, **params):
        startTime = datetime.now()
        tempDf = self.orderCommDf.drop_duplicates(['user_id','cate','o_id'])
        tempDf['comment_date'] = pd.to_datetime(tempDf['comment_create_tm'].dt.date)
        tempDf = statDateLenSum(tempDf, index=['user_id','cate'], values='score_level', columns='comment_date', colVals=params['statDateVals'], statLen=dayLen, skipLen=params['skipLen'], stepLen=params['stepLen'])
        tempDf = tempDf.stack(level=['type']).unstack(level='cate')
        tempDf.fillna(0,inplace=True)
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf = tempDf.rename(columns=str).stack().unstack(level='type').rename(columns={'addup_len':'len','addup_sum':'sum'})
        tempDf['mean'] = tempDf['sum'] / tempDf['len']
        tempDf = tempDf['mean'].unstack(level='cate')
        tempDf.columns = ['user_cate%s_%s_comm_mean'%(cate,feaFlag) for cate in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
        print('stat %s user comm mean:'%feaFlag, datetime.now() - startTime)
        return df

    # 统计各类别平均下单天数间隔
    def getOrderDateDelta(self):
        startTime = datetime.now()
        orderDf = self.orderDf.drop_duplicates(subset=['user_id','o_date','cate']).sort_index(by=['user_id','cate','o_date'])
        orderDf['last_user'] = orderDf['user_id'].shift(1) == orderDf['user_id']
        orderDf['last_cate'] = orderDf['cate'].shift(1) == orderDf['cate']
        orderDf['last_o_date'] = orderDf['o_date'].shift(1)
        orderDf['last_cate_o_daydelta'] = (orderDf['o_date'] - orderDf['last_o_date']).dt.days
        orderDf.loc[~(orderDf.last_user&orderDf.last_cate), 'last_cate_o_daydelta'] = np.nan
        self.orderDf = self.orderDf.merge(orderDf[['o_id','cate','last_cate_o_daydelta']], how='left')
        tempDf = pd.pivot_table(orderDf, index=['user_id'], columns='cate', values='last_cate_o_daydelta', aggfunc=np.mean)
        self.cateDayDelta = tempDf.apply(lambda x: (x[x>0]//1).value_counts().index[0])
        print('stat cate order aver daydelta:', datetime.now() - startTime)

    # 统计用户下单间隔
    def addOrderDatedeltaAndPredict(self, df, dayLen=None, feaFlag=None, **params):
        startTime = datetime.now()
        tempDf = statDateLenSum(self.orderDf, ['user_id','cate'], 'last_cate_o_daydelta', columns='o_date', colVals=params['statDateVals'], statLen=dayLen, skipLen=params['skipLen'], stepLen=params['stepLen'])
        tempDf['addup_mean'] = tempDf['addup_sum'] / (tempDf['addup_len'] - 1)
        tempDf.loc[tempDf.addup_mean==0,'addup_mean'] = np.nan
        tempDf.drop(['addup_len','addup_sum'], axis=1, inplace=True)
        tempDf = tempDf.unstack(level='cate')
        tempDf.columns = tempDf.columns.droplevel()
        tempDf.columns = ['user_cate%s_%s_order_daydelta_mean'%(x,feaFlag) for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
        print('stat %s user cate daydelta mean:'%feaFlag, datetime.now() - startTime)

        # 根据平均间隔推算各类目下次购买时间
        startTime = datetime.now()
        for x in self.cateList:
            temp = df['user_cate%s_%s_order_daydelta_mean'%(x,feaFlag)].fillna(self.cateDayDelta[x])
            df['cate%s_next_order_pred_by_%s'%(x,feaFlag)] = temp - df['user_cate%s_last_order_timedelta'%x]
        print('stat %s user next cate order predict:'%feaFlag, datetime.now() - startTime)
        return df

    # 购买商品的均价和最大最小值
    def addItemPrice(self, df, dayLen=None, feaFlag=None, **params):
        startTime = datetime.now()
        tempDf = statDateLenSum(self.orderDf.drop_duplicates(subset=['user_id','sku_id','o_id']), index=['user_id','cate'], values='price', columns='o_date', colVals=params['statDateVals'], statLen=dayLen, skipLen=params['skipLen'], stepLen=params['stepLen'])
        tempDf = tempDf.stack(level=['type']).unstack(level='cate')
        tempDf.fillna(0,inplace=True)
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf = tempDf.rename(columns=str).stack().unstack(level='type')
        tempDf['mean'] = tempDf['addup_sum'] / tempDf['addup_len']
        tempDf = tempDf['mean'].unstack(level='cate')
        tempDf.columns = ['user_cate%s_%s_ordersku_price_mean'%(cate,feaFlag) for cate in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
        print('stat %s order sku price mean:'%feaFlag, datetime.now() - startTime)

        startTime = datetime.now()
        tempDf = pd.pivot_table(self.orderDf.drop_duplicates(subset=['user_id','sku_id','o_id']), index=['user_id','cate'], values='price', columns='o_date', aggfunc=[np.min,np.max])
        for dt in np.setdiff1d(params['statDateVals'],tempDf.columns.levels[-1]):
            tempDf.loc[:, pd.IndexSlice['amax',dt]] = np.nan
            tempDf.loc[:, pd.IndexSlice['amin',dt]] = np.nan
        tempDf.sort_index(axis=1, inplace=True)
        idxList = [i for i in range(params['skipLen'],len(params['statDateVals'])) if (i-params['skipLen'])%params['stepLen']==0]
        for i in idxList:
            if dayLen!=None:
                dt = tempDf.columns.levels[-1][i]
                tempDf.loc[:,pd.IndexSlice['min',dt]] = tempDf['amin'].iloc[:,i-dayLen:i].min(axis=1)
                tempDf.loc[:,pd.IndexSlice['max',dt]] = tempDf['amax'].iloc[:,i-dayLen:i].max(axis=1)
            else:
                dt = tempDf.columns.levels[-1][i]
                tempDf.loc[:,pd.IndexSlice['min',dt]] = tempDf['amin'].iloc[:,:i].min(axis=1)
                tempDf.loc[:,pd.IndexSlice['max',dt]] = tempDf['amax'].iloc[:,:i].max(axis=1)
        tempDf.columns.names = ['type','date_range']
        tempDf = tempDf[['min','max']].stack(level=['date_range']).unstack('cate')
        tempDf.loc[:,pd.IndexSlice['min','all']] = tempDf['min'].min(axis=1)
        tempDf.loc[:,pd.IndexSlice['max','all']] = tempDf['max'].max(axis=1)
        tempDf.loc[:,pd.IndexSlice['min','task']] = tempDf['min'][self.taskCate].min(axis=1)
        tempDf.loc[:,pd.IndexSlice['max','task']] = tempDf['max'][self.taskCate].max(axis=1)
        tempDf.loc[:,pd.IndexSlice['min','other']] = tempDf['min'][self.otherCate].min(axis=1)
        tempDf.loc[:,pd.IndexSlice['max','other']] = tempDf['max'][self.otherCate].max(axis=1)
        tempDf.columns = ['user_cate%s_%s_ordersku_price_%s'%(cate,feaFlag,type) for type,cate in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
        print('stat %s order sku price maxmin:'%feaFlag, datetime.now() - startTime)
        return df

    # 订单及订单各类目均价
    def addOrderPrice(self, df, dayLen=None, feaFlag=None, **params):
        startTime = datetime.now()
        tempDf = statDateLenSum(self.orderDf.drop_duplicates(subset=['user_id','o_id','cate']), index=['user_id','cate'], values='o_cate_price_sum', columns='o_date', colVals=params['statDateVals'], statLen=dayLen, skipLen=params['skipLen'], stepLen=params['stepLen'])
        tempDf['mean'] = tempDf['addup_sum'] / tempDf['addup_len']
        tempDf = tempDf['mean'].unstack(level='cate')
        tempDf.columns = ['user_%s_order_cate%s_price_mean'%(feaFlag,cate) for cate in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
        # 计算最近一次订单与均值的差
        for cate in self.cateList:
            df['user_cate%s_last_order_price_delta_by_%s'%(cate,feaFlag)] = df['user_cate%s_last_order_price'%cate] - df['user_%s_order_cate%s_price_mean'%(feaFlag,cate)]
        print('stat %s order cate price mean:'%feaFlag, datetime.now() - startTime)

        startTime = datetime.now()
        tempDf = statDateLenSum(self.orderDf.drop_duplicates(subset=['user_id','o_id']), index=['user_id'], values='o_price_sum', columns='o_date', colVals=params['statDateVals'], statLen=dayLen, skipLen=params['skipLen'], stepLen=params['stepLen'])
        tempDf['user_%s_order_price_mean'%feaFlag] = tempDf['addup_sum'] / tempDf['addup_len']
        df = df.merge(tempDf[['user_%s_order_price_mean'%feaFlag]], how='left', left_on=['user_id','date_range'], right_index=True)
        # 计算最近一次订单与均值的差
        df['user_last_order_price_delta_by_%s'%(feaFlag)] = df['user_cateall_last_order_price'] - df['user_%s_order_price_mean'%feaFlag]
        print('stat %s order price mean:'%feaFlag, datetime.now() - startTime)
        return df

    # 用户下单总价
    def addUserPrice(self, df, dayLen=None, feaFlag=None, **params):
        startTime = datetime.now()
        tempDf = statDateLenSum(self.orderDf.drop_duplicates(subset=['user_id','cate','o_id']), index=['user_id','cate'], values='o_cate_price_sum', columns='o_date', colVals=params['statDateVals'], statLen=dayLen, skipLen=params['skipLen'], stepLen=params['stepLen'])
        tempDf = tempDf['addup_sum'].unstack('cate')
        tempDf.fillna(0,inplace=True)
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf.columns = ['user_cate%s_%s_price'%(cate,feaFlag) for cate in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
        if dayLen==None:
            for k in tempDf.columns.values:
                df[k] = df[k] / df['user_his_days']
        print('stat %s user price mean:'%feaFlag, datetime.now() - startTime)
        return df

    # 购买商品的参数一均值
    def addItemPara1(self, df, dayLen=None, feaFlag=None, **params):
        startTime = datetime.now()
        tempDf = statDateLenSum(self.orderDf.drop_duplicates(subset=['user_id','sku_id','o_id']), index=['user_id','cate'], values='para_1', columns='o_date', colVals=params['statDateVals'], statLen=dayLen, skipLen=params['skipLen'], stepLen=params['stepLen'])
        tempDf['mean'] = tempDf['addup_sum'] / tempDf['addup_len']
        tempDf = tempDf['mean'].unstack(level='cate')
        tempDf.columns = ['user_cate%s_%s_ordersku_para1_mean'%(cate,feaFlag) for cate in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
        # 计算最近一次订单与均值的差
        for cate in self.cateList:
            df['user_cate%s_last_ordersku_para1_delta_by_%s'%(cate,feaFlag)] = df['user_cate%s_last_order_para1'%cate] - df['user_cate%s_%s_ordersku_para1_mean'%(cate,feaFlag)]
        print('stat %s order sku para1 mean:'%feaFlag, datetime.now() - startTime)
        return df

    # 下单最多的地区
    def addOrderArea(self, df, dayLen=None, feaFlag=None, **params):
        startTime = datetime.now()
        tempDf = statDateLenSum(self.orderDf.drop_duplicates(subset=['o_id']), index=['user_id','o_area'], values='o_id', columns='o_date', colVals=params['statDateVals'], statLen=dayLen, skipLen=params['skipLen'], stepLen=params['stepLen'])
        tempDf = tempDf[tempDf.addup_len>0]['addup_len'].unstack(['o_area'])
        tempDf['user_%s_order_area'%feaFlag] = tempDf.idxmax(axis=1)
        df = df.merge(tempDf[['user_%s_order_area'%feaFlag]], how='left', left_on=['user_id','date_range'], right_index=True)
        print('stat %s order sku price mean:'%feaFlag, datetime.now() - startTime)
        return df

# 获取日期列表
def getDateList(startDate=None, endDate='2017-05-01', periods=None, freq=30):
    if periods == None:
        periods = (pd.to_datetime(endDate) - pd.to_datetime(startDate)).days // freq + 1
        startDate = None
    dateList = pd.date_range(start=startDate, end=endDate, periods=periods, freq='%dD'%freq)
    return dateList

# 新增额外的特征
def addExtraFea(feaFactory, df):
    dateList = pd.to_datetime(np.unique(df.date_range))
    params = {
        'dateList': dateList,
        'statDateVals': pd.date_range(start=feaFactory.startDate, end=feaFactory.endDate+timedelta(days=1), freq='D'),
        'stepLen': (dateList[1] - dateList[0]).days,
        'skipLen': (dateList[0] - feaFactory.startDate).days
        }
    print(len(df))
    feaFactory.getOrderDateDelta()
    df = feaFactory.addActionTimes(df, dayLen=7, feaFlag='last7day', **params)
    df = feaFactory.addActionDays(df, dayLen=7, feaFlag='last7day', **params)
    df = feaFactory.addOrderSku(df, dayLen=7, feaFlag='last7day', **params)
    df = feaFactory.addOrderDatedeltaAndPredict(df, dayLen=7, feaFlag='last7day', **params)

    df = feaFactory.addLastOrderId(df, **params)
    df = feaFactory.addLastOrderPrice(df, **params)
    df = feaFactory.addLastOrderPara1(df, **params)
    df = feaFactory.addItemPrice(df, feaFlag='perday', **params)
    df = feaFactory.addItemPrice(df, dayLen=7, feaFlag='last7day', **params)
    df = feaFactory.addItemPrice(df, dayLen=15, feaFlag='last15day', **params)
    df = feaFactory.addItemPrice(df, dayLen=30, feaFlag='last30day', **params)
    df = feaFactory.addItemPrice(df, dayLen=90, feaFlag='last90day', **params)
    df = feaFactory.addOrderPrice(df, feaFlag='perday', **params)
    df = feaFactory.addOrderPrice(df, dayLen=7, feaFlag='last7day', **params)
    df = feaFactory.addOrderPrice(df, dayLen=15, feaFlag='last15day', **params)
    df = feaFactory.addOrderPrice(df, dayLen=30, feaFlag='last30day', **params)
    df = feaFactory.addOrderPrice(df, dayLen=90, feaFlag='last90day', **params)
    df = feaFactory.addUserPrice(df, feaFlag='perday', **params)
    df = feaFactory.addUserPrice(df, dayLen=7, feaFlag='last7day', **params)
    df = feaFactory.addUserPrice(df, dayLen=15, feaFlag='last15day', **params)
    df = feaFactory.addUserPrice(df, dayLen=30, feaFlag='last30day', **params)
    df = feaFactory.addUserPrice(df, dayLen=90, feaFlag='last90day', **params)
    df = feaFactory.addItemPara1(df, feaFlag='perday', **params)
    df = feaFactory.addItemPara1(df, dayLen=7, feaFlag='last7day', **params)
    df = feaFactory.addItemPara1(df, dayLen=15, feaFlag='last15day', **params)
    df = feaFactory.addItemPara1(df, dayLen=30, feaFlag='last30day', **params)
    df = feaFactory.addItemPara1(df, dayLen=90, feaFlag='last90day', **params)
    df = feaFactory.addOrderArea(df, feaFlag='perday', **params)
    df = feaFactory.addOrderArea(df, dayLen=7, feaFlag='last7day', **params)
    df = feaFactory.addOrderArea(df, dayLen=15, feaFlag='last15day', **params)
    df = feaFactory.addOrderArea(df, dayLen=30, feaFlag='last30day', **params)
    df = feaFactory.addOrderArea(df, dayLen=90, feaFlag='last90day', **params)
    return df

# xgb模型方法封装
class XgbModel:
    def __init__(self, feaNames=None, params={}):
        self.feaNames = feaNames
        self.params = {
            'objective': 'reg:linear',
            'eval_metric':'rmse',
            'silent': True,
            'eta': 0.1,
            'max_depth': 4,
            'gamma': 0.5,
            'subsample': 0.95,
            'colsample_bytree': 1,
            'min_child_weight': 9,
            # 'scale_pos_weight': 1.2,
            'lambda': 500,
            # 'nthread': 20,
            # 'seed': 0,
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

    def trainCV(self, X, y, nFold=3, verbose=True, num_boost_round=8000, early_stopping_rounds=10, weight=None):
        X = X.astype(float)
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feaNames)
        if weight!=None:
            dtrain.set_weight(weight)
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

    def gridSearch(self, X, y, nFold=3, verbose=1, num_boost_round=100):
        # print('grid')
        # exit()
        paramsGrids = {
            # 'n_estimators': [50+5*i for i in range(0,30)],
            'max_depth': list(range(3,8)),
            'subsample': [1-0.05*i for i in range(0,6)],
            'colsample_bytree': [1-0.05*i for i in range(0,6)],
            'gamma': [0,0.05,0.1,0.5,1,5,10,20,50,100],
            'reg_lambda': [0,5,10,20,50,100,150,200,300,400,500,700,1000],
            'min_child_weight': list(range(0,15)),
            # 'reg_alpha': [0+2*i for i in range(0,10)],
            # 'scale_pos_weight': [1+0.2*i for i in range(10)],
            # 'max_delta_step': [0+1*i for i in range(0,8)],
        }
        for k,v in paramsGrids.items():
            gsearch = GridSearchCV(
                estimator = xgb.XGBClassifier(
                    max_depth = self.params['max_depth'],
                    gamma = self.params['gamma'],
                    learning_rate = self.params['eta'],
                    # max_delta_step = self.params['max_delta_step'],
                    min_child_weight = self.params['min_child_weight'],
                    subsample = self.params['subsample'],
                    colsample_bytree = self.params['colsample_bytree'],
                    # scale_pos_weight = self.params['scale_pos_weight'],
                    silent = self.params['silent'],
                    reg_lambda = self.params['lambda'],
                    n_estimators = num_boost_round,
                ),
                # param_grid = paramsGrids,
                param_grid = {k:v},
                scoring = 'neg_mean_squared_error',
                cv = nFold,
                verbose = verbose,
                n_jobs = 8
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

# 获取前5w用户
def getRankDf(df, rankCol='buy_predict'):
    resultDf = df.sort_index(by=[rankCol], ascending=False).iloc[:50000]
    return resultDf

# 训练方案
def trainPlan(trainDf, testDf, fea, mode='single', showFeaScore=True):
    if mode=='single':
        dateModel = XgbModel(feaNames=fea)
        dateModel.trainCV(trainDf[fea].values, trainDf['day_label'].values)
        testDf.loc[:,'day_predict'] = dateModel.predict(testDf[fea].values)
        if showFeaScore:
            feaScoreDf = dateModel.getFeaScore(show=True)
        else:
            feaScoreDf = dateModel.getFeaScore()
    elif mode=='stacking':
        fea2 = fea + ['day_predict1']
        dateModel = XgbModel(feaNames=fea)
        dateModel2 = XgbModel(feaNames=fea2)
        trainDf.loc[:,'day_predict1'],testDf.loc[:,'day_predict1'] = getOof(dateModel, trainDf[fea].values, trainDf['day_label'].values, testDf[fea].values, stratify=True)
        dateModel2.trainCV(trainDf[fea2].values, trainDf['day_label'].values)
        testDf.loc[:,'day_predict'] = dateModel2.predict(testDf[fea2].values)
        if showFeaScore:
            print('stacking L1:')
            dateModel.getFeaScore(show=True)
            print('stacking L2:')
            feaScoreDf = dateModel2.getFeaScore(show=True)
        else:
            feaScoreDf = dateModel2.getFeaScore()
    testDf.loc[:,'day_predict'] = round(testDf['day_predict'])
    daySeries = testDf['day_predict'].copy()
    daySeries.loc[daySeries<0] = 0
    testDf['pred_date'] = (testDf['date_range'] + pd.to_timedelta(daySeries, unit='d')).dt.strftime('%Y-%m-%d')
    return trainDf,testDf,feaScoreDf

def score1(labelList):
    weightList = [1/(1+math.log(i)) for i in range(1,len(labelList)+1)]
    s1 = np.sum(np.array(weightList) * np.array(labelList))
    s1 /= np.sum(weightList)
    return s1

def score2(labelList, predList):
    fu = [10 / (10 + (x1-x2)**2) for x1,x2 in np.column_stack([predList,labelList]) if x2==x2]
    userLen = len(labelList[labelList==labelList])
    s2 = np.sum(fu) / userLen
    return s2

def main():
    # 数据导入
    startTime = datetime.now()
    userDf = importDf('../data/jdata_user_basic_info.csv')
    skuDf = importDf('../data/jdata_sku_basic_info.csv')
    actionDf = importDf('../data/jdata_user_action.csv')
    orderDf = importDf('../data/jdata_user_order.csv')
    commDf = importDf('../data/jdata_user_comment_score.csv')
    # userList = userDf.sample(frac=0.2, random_state=0)['user_id'].values
    # dfs = {
    #     'user_df': userDf[userDf.user_id.isin(userList)],
    #     'sku_df': skuDf,
    #     'action_df': actionDf[actionDf.user_id.isin(userList)],
    #     'order_df': orderDf[orderDf.user_id.isin(userList)],
    #     'comm_df': commDf[commDf.user_id.isin(userList)]
    #     }
    dfs = {
        'user_df': userDf,
        'sku_df': skuDf,
        'action_df': actionDf,
        'order_df': orderDf,
        'comm_df': commDf
        }
    if not(os.path.exists('../temp/yuna_feaFactory2.csv')):
        yuna_feaFactory2.main()
    df = importDf('../temp/yuna_feaFactory2.csv') # 前期保存的特征工程
    # df = importDf('../feaFactory2_sample.csv')
    # df = df[df.user_id.isin(userList)]
    df['date_range'] = pd.to_datetime(df['date_range'])
    df = df[df.date_range.isin(getDateList(startDate='2016-12-01', endDate='2017-09-01', freq=15))]
    print('import dataset:', datetime.now() - startTime)

    # 特征工程
    feaFactory = FeaFactory(dfs)
    df = addExtraFea(feaFactory, df)    # 补充后期增加的特征
    print('train user num:', df.date_range.value_counts())
    print('train day label:', df.day_label.value_counts())

    fea = [
        'age','sex','user_lv_cd','user_his_days',
        'date1111_daydelta','date618_daydelta',
        ]
    cateList = feaFactory.cateList
    fea.extend(['user_cate%s_perday_view'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_perday_viewday'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_perday_order'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_perday_orderday'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_perday_order_daymean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_perday_ordersku_len'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_perday_ordersku_sum'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_perday_order_daymin'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_perday_order_daymax'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_perday_comm_mean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_perday_order_daydelta_mean'%x for x in [101,30]])
    fea.extend(['cate%s_next_order_pred_by_perday'%x for x in [101,30]])
    fea.extend(['user_cate%s_perday_ordersku_price_mean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_perday_ordersku_price_min'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_perday_ordersku_price_max'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_perday_order_cate%s_price_mean'%x for x in [101,30]])
    fea.extend(['user_perday_order_price_mean'])
    fea.extend(['user_cate%s_perday_price'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last_order_price_delta_by_perday'%x for x in [101,30]])
    fea.extend(['user_last_order_price_delta_by_perday'])
    fea.extend(['user_cate%s_perday_ordersku_para1_mean'%x for x in [101,30]])
    fea.extend(['user_cate%s_last_ordersku_para1_delta_by_perday'%x for x in [101,30]])
    fea.extend(['user_perday_order_area'])
    fea.extend(['user_cate%s_last7day_view'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last7day_viewday'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last7day_order'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last7day_orderday'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last7day_order_daymean'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last7day_ordersku_len'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last7day_ordersku_sum'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last7day_order_daymin'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last7day_order_daymax'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last7day_comm_mean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last7day_order_daydelta_mean'%x for x in [101,30]])
    fea.extend(['cate%s_next_order_pred_by_last7day'%x for x in [101,30]])
    fea.extend(['user_cate%s_last7day_ordersku_price_mean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last7day_ordersku_price_min'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last7day_ordersku_price_max'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_last7day_order_cate%s_price_mean'%x for x in [101,30]])
    fea.extend(['user_last7day_order_price_mean'])
    fea.extend(['user_cate%s_last7day_price'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last_order_price_delta_by_last7day'%x for x in [101,30]])
    fea.extend(['user_last_order_price_delta_by_last7day'])
    fea.extend(['user_cate%s_last7day_ordersku_para1_mean'%x for x in [101,30]])
    fea.extend(['user_cate%s_last_ordersku_para1_delta_by_last7day'%x for x in [101,30]])
    fea.extend(['user_last7day_order_area'])
    fea.extend(['user_cate%s_last15day_view'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_viewday'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_order'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_orderday'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last15day_order_daymean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_ordersku_len'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_ordersku_sum'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last15day_order_daymin'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last15day_order_daymax'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_comm_mean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_order_daydelta_mean'%x for x in [101,30]])
    fea.extend(['cate%s_next_order_pred_by_last15day'%x for x in [101,30]])
    fea.extend(['user_cate%s_last15day_ordersku_price_mean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_ordersku_price_min'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_ordersku_price_max'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_last15day_order_cate%s_price_mean'%x for x in [101,30]])
    fea.extend(['user_last15day_order_price_mean'])
    fea.extend(['user_cate%s_last15day_price'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last_order_price_delta_by_last15day'%x for x in [101,30]])
    fea.extend(['user_last_order_price_delta_by_last15day'])
    fea.extend(['user_cate%s_last15day_ordersku_para1_mean'%x for x in [101,30]])
    fea.extend(['user_cate%s_last_ordersku_para1_delta_by_last15day'%x for x in [101,30]])
    fea.extend(['user_last15day_order_area'])
    fea.extend(['user_cate%s_last30day_view'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last30day_viewday'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last30day_order'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last30day_orderday'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last30day_order_daymean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last30day_ordersku_len'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last30day_ordersku_sum'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last30day_order_daymin'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last30day_order_daymax'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last30day_comm_mean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last30day_order_daydelta_mean'%x for x in [101,30]])
    fea.extend(['cate%s_next_order_pred_by_last30day'%x for x in [101,30]])
    fea.extend(['user_cate%s_last30day_ordersku_price_mean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last30day_ordersku_price_min'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last30day_ordersku_price_max'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_last30day_order_cate%s_price_mean'%x for x in [101,30]])
    fea.extend(['user_last30day_order_price_mean'])
    fea.extend(['user_cate%s_last30day_price'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last_order_price_delta_by_last30day'%x for x in [101,30]])
    fea.extend(['user_last_order_price_delta_by_last30day'])
    fea.extend(['user_cate%s_last30day_ordersku_para1_mean'%x for x in [101,30]])
    fea.extend(['user_cate%s_last_ordersku_para1_delta_by_last30day'%x for x in [101,30]])
    fea.extend(['user_last30day_order_area'])
    fea.extend(['user_cate%s_last90day_view'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last90day_viewday'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last90day_order'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last90day_orderday'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last90day_order_daymean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last90day_ordersku_len'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last90day_ordersku_sum'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last90day_order_daymin'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last90day_order_daymax'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last90day_comm_mean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last90day_order_daydelta_mean'%x for x in [101,30]])
    fea.extend(['cate%s_next_order_pred_by_last90day'%x for x in [101,30]])
    fea.extend(['user_cate%s_last90day_ordersku_price_mean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last90day_ordersku_price_min'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last90day_ordersku_price_max'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_last90day_order_cate%s_price_mean'%x for x in [101,30]])
    fea.extend(['user_last90day_order_price_mean'])
    fea.extend(['user_cate%s_last90day_price'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last_order_price_delta_by_last90day'%x for x in [101,30]])
    fea.extend(['user_last_order_price_delta_by_last90day'])
    fea.extend(['user_cate%s_last90day_ordersku_para1_mean'%x for x in [101,30]])
    fea.extend(['user_cate%s_last_ordersku_para1_delta_by_last90day'%x for x in [101,30]])
    fea.extend(['user_last90day_order_area'])

    fea.extend(['user_cate%s_last_view_timedelta'%x for x in cateList+['all']])
    fea.extend(['user_cate%s_last_order_timedelta'%x for x in cateList+['all']])
    fea.extend(['user_cate%s_last_order_view_timedelta'%x for x in cateList+['all']])
    fea.extend(['user_cate%s_last_order_price'%x for x in [101,30,'all']])
    fea.extend(['user_cate%s_last_order_para1'%x for x in [101,30]])

    print(df[fea].info(max_cols=500))
    fea2 = fea+['day_predict1']
    # print(df[fea].describe())
    # exportResult(df[['user_id','date_range','day_label']+fea], '../feaFactory2_final.csv')
    # exit()

    # # 模型调参
    # dt = date(2017, 8, 2)
    # trainDf = df[(df.date_range<dt)]
    # tempSeries = trainDf['date_range'] + pd.to_timedelta(trainDf['day_label'], unit='d')
    # trainDf.drop(trainDf[tempSeries>=dt].index, inplace=True)
    # trainDf = trainDf[trainDf.day_label.notnull()]
    # print('train num:', len(trainDf))
    # XgbModel(feaNames=fea).gridSearch(trainDf[fea].values, trainDf['day_label'].values)

    # 正式模型
    modelName = 'dateModel2'
    print(modelName)
    trainDf = df[df.date_range<date(2017,9,1)]
    trainDf = trainDf[trainDf.day_label.notnull()]
    print('train num:', len(trainDf))
    testDf = df[df.date_range==date(2017,9,1)]
    trainDf, testDf, _ = trainPlan(trainDf, testDf, fea, mode='single')
    print('pred value count:\n',testDf['day_predict'].value_counts())

    # 导出模型
    exportResult(testDf[['user_id','pred_date']], '../temp/yuna_%s_all.csv'%modelName)
    # exportResult(predictDf[['user_id','pred_date']], '%s.csv'%modelName)
    # exit()

    # # 本地模型验证
    # feaScoreDf = pd.DataFrame(index=fea2)
    # costDf = pd.DataFrame(index=['auc','score1','rmse','score2','score'])
    # for dt in pd.to_datetime(np.unique(df.date_range.values)[-4:-2]):
    #     trainDf = df[(df.date_range<dt)]
    #     tempSeries = trainDf['date_range'] + pd.to_timedelta(trainDf['day_label'], unit='d')
    #     trainDf.drop(trainDf[tempSeries>=dt].index, inplace=True)
    #     testDf = df[df.date_range==dt]
    #     trainDf = trainDf[trainDf.day_label.notnull()]
    #     testDf = testDf[testDf.day_label.notnull()]
    #     print('train num:', len(trainDf))
    #
    #     # dateModel.gridSearch(trainDf[fea].values, trainDf['day_label'].values)
    #     trainDf, testDf, scoreDf = trainPlan(trainDf, testDf, fea, mode='single')
    #     scoreDf.columns = [dt.strftime('%Y-%m-%d')]
    #     feaScoreDf = feaScoreDf.merge(scoreDf, how='left', left_index=True, right_index=True)
    #     rmse = metrics.mean_squared_error(testDf['day_label'].values, testDf['day_predict'].values)
    #     costDf.loc['rmse',dt.strftime('%Y-%m-%d')] = rmse
    #     print('异常预测值：',testDf[(testDf.day_predict<0)|(testDf.day_predict>=30)][['user_id','day_predict','day_label']])
    #     costDf.loc['score2',dt.strftime('%Y-%m-%d')] = score2(testDf['day_label'].values, testDf['day_predict'].values)
    # print(feaScoreDf.iloc[:60], feaScoreDf.iloc[60:120], feaScoreDf.iloc[120:180], feaScoreDf.iloc[180:240], feaScoreDf.iloc[240:])
    # print(costDf)

if __name__ == '__main__':
    finishTime = datetime.now()
    main()
    print('Finished program in: ', datetime.now() - finishTime)
