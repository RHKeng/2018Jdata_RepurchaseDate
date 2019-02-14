#!/usr/bin/env python
# -*-coding:utf-8-*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import csv
from datetime import *
import math

from sklearn.preprocessing import *
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
import xgboost as xgb
from sklearn.externals import joblib

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
def statDateLenSum(df, index, values, columns='date_range', colVals=pd.date_range(start='2016-09-01',end='2017-09-01', freq='D'), statLen=None, skipLen=1, stepLen=1):
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
        dfs['order_df'] = dfs['order_df'].merge(dfs['sku_df'][['sku_id','cate']], how='left', on='sku_id')
        dfs['order_df']['day_of_month_end'] = pd.Index(dfs['order_df']['o_date']).shift(1,freq='MS')
        dfs['order_df']['day_of_month_end'] = (dfs['order_df']['day_of_month_end'] - dfs['order_df']['o_date']).dt.days
        dfs['action_df']['year'] = dfs['action_df'].a_date.dt.year
        dfs['action_df']['month'] = dfs['action_df'].a_date.dt.month
        dfs['action_df']['day'] = dfs['action_df'].a_date.dt.day
        dfs['action_df'] = dfs['action_df'].merge(dfs['sku_df'][['sku_id','cate']], how='left', on='sku_id')
        dfs['action_df']['day_of_month_end'] = pd.Index(dfs['action_df']['a_date']).shift(1,freq='MS')
        dfs['action_df']['day_of_month_end'] = (dfs['action_df']['day_of_month_end'] - dfs['action_df']['a_date']).dt.days

        self.userDf = dfs['user_df']
        self.skuDf = dfs['sku_df']
        self.actionDf = dfs['action_df']
        self.orderDf = dfs['order_df']
        self.commDf = dfs['comm_df']
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

        # 构建index
        userList = self.userDf['user_id'].values
        tempIdx = pd.MultiIndex.from_product([userList,dateList], names=['user_id', 'date_range'])
        df = pd.DataFrame(index=tempIdx)
        df.reset_index(inplace=True)
        df['date_id'] = (df['date_range'] - params['endDate']).dt.days
        df['date_range_id'] = (df['date_range'] - params['endDate']).dt.days // params['freq']

        # 替原始数据集划分时间区间
        self.actionDf['a_date_id'] = (self.actionDf['a_date'] - params['endDate']).dt.days
        self.actionDf['date_range_id'] = self.actionDf['a_date_id'] // params['freq']
        self.actionDf['date_range'] = params['endDate'] + pd.to_timedelta(self.actionDf['date_range_id']*params['freq'], unit='d')
        self.orderDf['o_date_id'] = (self.orderDf['o_date'] - params['endDate']).dt.days
        self.orderDf['date_range_id'] = self.orderDf['o_date_id'] // params['freq']
        self.orderDf['date_range'] = params['endDate'] + pd.to_timedelta(self.orderDf['date_range_id']*params['freq'], unit='d')

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

    def addUserFea(self, df, **params):
        cateList = list(set(self.skuDf.cate))
        # 用户基础信息
        df = df.merge(self.userDf, how='left', on='user_id')
        df['user_his_days'] = (df['date_range'] - df['user_first_date']).dt.days

        # 距离上次行为时间
        startTime = datetime.now()
        tempDf = pd.pivot_table(self.actionDf, index=['date_range'], columns=['user_id','cate','a_type'], values='a_date', aggfunc=np.max)
        if params['endDate'] not in tempDf.index:
            tempDf.loc[params['endDate'], :] = np.nan
        tempDf = tempDf.shift(1)
        tempDf.fillna(method='ffill', inplace=True)
        tempDf = tempDf.stack(level=['user_id','a_type'])
        tempDf['all'] = tempDf.max(axis=1)
        tempDf['task'] = tempDf[[101,30]].max(axis=1)
        tempDf['other'] = tempDf[list(set(cateList)-set([101,30]))].max(axis=1)
        tempDf = tempDf.stack(level=['cate'])
        tempDf = (tempDf.index.get_level_values('date_range').values - tempDf).dt.days
        tempDf.index.set_levels(['view','follow'], level='a_type', inplace=True)
        tempDf = tempDf.unstack(level=['cate','a_type'])
        tempDf.columns = ['user_cate%s_last_%s_timedelta'%(x[0],x[1]) for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['date_range','user_id'], right_index=True)
        df.fillna({k:999 for k in tempDf.columns.values}, inplace=True)

        tempDf = pd.pivot_table(self.orderDf, index=['date_range'], columns=['user_id','cate'], values='o_date', aggfunc=np.max)
        if params['endDate'] not in tempDf.index:
            tempDf.loc[params['endDate'], :] = np.nan
        tempDf = tempDf.shift(1)
        tempDf.fillna(method='ffill', inplace=True)
        tempDf = tempDf.stack(level=['user_id'])
        tempDf['all'] = tempDf.max(axis=1)
        tempDf['task'] = tempDf[[101,30]].max(axis=1)
        tempDf['other'] = tempDf[list(set(cateList)-set([101,30]))].max(axis=1)
        df = df.merge(tempDf.rename(columns={x:'user_cate%s_last_order_time'%x for x in tempDf.columns}), how='left', left_on=['date_range','user_id'], right_index=True)
        tempDf = tempDf.stack(level=['cate'])
        tempDf = (tempDf.index.get_level_values('date_range').values - tempDf).dt.days
        tempDf = tempDf.unstack(level=['cate'])
        tempDf.columns = ['user_cate%s_last_order_timedelta'%x for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['date_range','user_id'], right_index=True)
        df.fillna({k:999 for k in tempDf.columns.values}, inplace=True)
        df.drop(df[df.user_cateall_last_order_timedelta>31*3].index, inplace=True)  # 根据赛题说明删除3个月内无购买行为的用户
        for x in cateList+['all']:
            df['user_cate%s_last_order_view_timedelta'%x] = df['user_cate%s_last_order_timedelta'%x] - df['user_cate%s_last_view_timedelta'%x]
            df['user_cate%s_last_order_follow_timedelta'%x] = df['user_cate%s_last_order_timedelta'%x] - df['user_cate%s_last_follow_timedelta'%x]
        print('user last timedelta:', datetime.now() - startTime)

        # 最近一次订单评价记录
        startTime = datetime.now()
        tempDf = pd.pivot_table(self.commDf, index=['o_id'], values=['score_level','comment_create_tm'], aggfunc={'score_level':np.mean, 'comment_create_tm':np.max})
        orderCommDf = self.orderDf.merge(tempDf, how='left', left_on='o_id', right_index=True)
        for x in cateList+['all']:
            if x=='all':
                tempDf = orderCommDf.drop_duplicates(['user_id','o_date'])
            else:
                tempDf = orderCommDf[orderCommDf.cate==x].drop_duplicates(['user_id','o_date'])
            df = df.merge(tempDf[['user_id','o_date','o_id','comment_create_tm','score_level']], how='left', left_on=['user_id','user_cate%s_last_order_time'%x],right_on=['user_id','o_date'])
            del df['o_date']
            df.loc[df.comment_create_tm>df.date_range, ['comment_create_tm','score_level']] = np.nan
            df.rename(columns={'o_id':'cate%s_last_order'%x,'comment_create_tm':'cate%s_last_order_comm_time'%x,'score_level':'cate%s_last_order_comm'%x}, inplace=True)
        print('last order comment:', datetime.now()-startTime)

        # 距离特殊日期的天数
        startTime = datetime.now()
        df['date1111_daydelta'] = (df['date_range'] - date(2016,11,11)).dt.days
        df['date618_daydelta'] = (df['date_range'] - date(2017,6,18)).dt.days
        print('special date daydelta:', datetime.now()-startTime)

        # 统计各类别平均下单天数间隔
        startTime = datetime.now()
        orderDf = self.orderDf.drop_duplicates(subset=['user_id','o_date','cate']).sort_index(by=['user_id','cate','o_date'])
        orderDf['last_user'] = orderDf['user_id'].shift(1) == orderDf['user_id']
        orderDf['last_cate'] = orderDf['cate'].shift(1) == orderDf['cate']
        orderDf['last_o_date'] = orderDf['o_date'].shift(1)
        orderDf['last_cate_o_daydelta'] = (orderDf['o_date'] - orderDf['last_o_date']).dt.days
        orderDf.loc[~(orderDf.last_user&orderDf.last_cate), 'last_cate_o_daydelta'] = np.nan
        self.orderDf = self.orderDf.merge(orderDf[['o_id','cate','last_cate_o_daydelta']], how='left')
        tempDf = pd.pivot_table(orderDf, index=['user_id'], columns='cate', values='last_cate_o_daydelta', aggfunc=np.mean)
        cateDayDelta = tempDf.apply(lambda x: (x[x>0]//1).value_counts().index[0])
        print(cateDayDelta)
        print('stat cate order aver daydelta:', datetime.now() - startTime)

        # 历史月行为统计
        def hisMonthStat(df, dayLen=None):
            feaFlag = 'last%sday'%(dayLen) if dayLen!=None else 'perday'
            skipLen = (pd.to_datetime(params['startDate']) - self.orderDf.o_date.min()).days
            if dayLen!=None:
                tempMonth = pd.to_datetime(params['startDate']) - timedelta(days=dayLen)
                statDateVals=pd.date_range(start=tempMonth,end='2017-09-01', freq='D')
            else:
                statDateVals=pd.date_range(start='2016-09-01',end='2017-09-01', freq='D')

            # 统计浏览与关注的次数
            startTime = datetime.now()
            if dayLen!=None:
                tempDf = self.actionDf[self.actionDf.a_date >= tempMonth]
                tempDf = statDateLenSum(tempDf, index=['user_id','cate','a_type'], values='a_num', columns='a_date', colVals=statDateVals, statLen=dayLen, skipLen=dayLen, stepLen=params['freq'])
            else:
                tempDf = statDateLenSum(self.actionDf, index=['user_id','cate','a_type'], values='a_num', columns='a_date', colVals=statDateVals, skipLen=skipLen, stepLen=params['freq'])                # tempDf = statDateLenSum(self.actionDf, index=['user_id','cate','a_type'], values='a_num')
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

            # 统计浏览与关注的天数
            startTime = datetime.now()
            if dayLen!=None:
                tempDf = self.actionDf[self.actionDf.a_date >= tempMonth]
                tempDf = statDateLenSum(tempDf.drop_duplicates(subset=['user_id','cate','a_type','a_date']), index=['user_id','cate','a_type'], values='a_num', columns='a_date', colVals=statDateVals, statLen=dayLen, skipLen=dayLen, stepLen=params['freq'])
            else:
                tempDf = statDateLenSum(self.actionDf.drop_duplicates(subset=['user_id','cate','a_type','a_date']), index=['user_id','cate','a_type'], values='a_num', columns='a_date', colVals=statDateVals, skipLen=skipLen, stepLen=params['freq'])
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

            # 统计订单数与平均下单日
            startTime = datetime.now()
            if dayLen!=None:
                tempDf = self.orderDf[self.orderDf.o_date >= tempMonth]
                tempDf = statDateLenSum(tempDf.drop_duplicates(subset=['o_id','cate']), index=['user_id','cate'], values='day', columns='o_date', colVals=statDateVals, statLen=dayLen, skipLen=dayLen, stepLen=params['freq'])
            else:
                tempDf = statDateLenSum(self.orderDf.drop_duplicates(subset=['o_id','cate']), index=['user_id','cate'], values='day', columns='o_date', colVals=statDateVals, skipLen=skipLen, stepLen=params['freq'])
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
            df.fillna({'user_cate%s_%s_order'%(k,feaFlag):0 for k in cateList}, inplace=True)
            if dayLen==None:
                for k in cateList:
                    df['user_cate%s_%s_order'%(k,feaFlag)] = df['user_cate%s_%s_order'%(k,feaFlag)] / df['user_his_days']
            print('stat %s order times & order day mean:'%feaFlag, datetime.now() - startTime)

            # 统计下单天数
            startTime = datetime.now()
            if dayLen!=None:
                tempDf = self.orderDf[self.orderDf.o_date >= tempMonth]
                tempDf = statDateLenSum(tempDf.drop_duplicates(subset=['user_id','cate','o_date']), index=['user_id','cate'], values='o_sku_num', columns='o_date', colVals=statDateVals, statLen=dayLen, skipLen=dayLen, stepLen=params['freq'])
            else:
                tempDf = statDateLenSum(self.orderDf.drop_duplicates(subset=['user_id','cate','o_date']), index=['user_id','cate'], values='o_sku_num', columns='o_date', colVals=statDateVals, skipLen=skipLen, stepLen=params['freq'])
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

            # 统计下单商品数及商品件数
            startTime = datetime.now()
            if dayLen!=None:
                tempDf = self.orderDf[self.orderDf.o_date >= tempMonth]
                tempDf = statDateLenSum(tempDf, index=['user_id','cate'], values='o_sku_num', columns='o_date', colVals=statDateVals, statLen=dayLen, skipLen=dayLen, stepLen=params['freq'])
            else:
                tempDf = statDateLenSum(self.orderDf, index=['user_id','cate'], values='o_sku_num', columns='o_date', colVals=statDateVals, skipLen=skipLen, stepLen=params['freq'])
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

            # 统计最早下单日和最晚下单日
            startTime = datetime.now()
            tempDf = self.orderDf.drop_duplicates(['user_id','cate','o_date']).set_index(['user_id','cate','o_date'])[['day']].unstack('o_date')
            for dt in np.setdiff1d(statDateVals,tempDf.columns.levels[-1]):
                tempDf.loc[:, pd.IndexSlice['day',dt]] = np.nan
            tempDf.sort_index(axis=1, inplace=True)
            idxList = [i for i in range(skipLen,len(tempDf.columns.levels[-1])) if (i-skipLen)%params['freq']==0]
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
            tempDf.loc[:,pd.IndexSlice['min','task']] = tempDf['min'][[101,30]].min(axis=1)
            tempDf.loc[:,pd.IndexSlice['max','task']] = tempDf['max'][[101,30]].max(axis=1)
            tempDf.loc[:,pd.IndexSlice['min','other']] = tempDf['min'][list(set(cateList)-set([101,30]))].min(axis=1)
            tempDf.loc[:,pd.IndexSlice['max','other']] = tempDf['max'][list(set(cateList)-set([101,30]))].max(axis=1)
            tempDf.columns = ['user_cate%s_%s_order_day%s'%(cate,feaFlag,type) for type,cate in tempDf.columns]
            df = df.merge(tempDf, how='left', left_on=['user_id','date_range'], right_index=True)
            print('stat %s order min & max day:'%feaFlag, datetime.now() - startTime)

            # 统计用户历史评价
            startTime = datetime.now()
            tempDf = orderCommDf.drop_duplicates(['user_id','cate','o_id'])
            tempDf['comment_date'] = pd.to_datetime(tempDf['comment_create_tm'].dt.date)
            if dayLen!=None:
                tempDf = tempDf[tempDf.comment_date >= tempMonth]
                tempDf = statDateLenSum(tempDf, index=['user_id','cate'], values='score_level', columns='comment_date', colVals=statDateVals, statLen=dayLen, skipLen=dayLen, stepLen=params['freq'])
            else:
                tempDf = statDateLenSum(tempDf, index=['user_id','cate'], values='score_level', columns='comment_date', colVals=statDateVals, skipLen=skipLen, stepLen=params['freq'])
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

            # 统计用户下单间隔
            startTime = datetime.now()
            if dayLen!=None:
                tempDf = self.orderDf[self.orderDf.o_date >= tempMonth]
                tempDf = statDateLenSum(tempDf, index=['user_id','cate'], values='last_cate_o_daydelta', columns='o_date', colVals=statDateVals, statLen=dayLen, skipLen=dayLen, stepLen=params['freq'])
            else:
                tempDf = statDateLenSum(self.orderDf, ['user_id','cate'], 'last_cate_o_daydelta', columns='o_date', colVals=statDateVals, skipLen=skipLen, stepLen=params['freq'])
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
            for x in set(self.skuDf.cate):
                temp = df['user_cate%s_%s_order_daydelta_mean'%(x,feaFlag)].fillna(cateDayDelta[x])
                df['cate%s_next_order_pred_by_%s'%(x,feaFlag)] = temp - df['user_cate%s_last_order_timedelta'%x]
            print('stat %s user next cate order predict:'%feaFlag, datetime.now() - startTime)

            return df
        df = hisMonthStat(df)
        df = hisMonthStat(df, dayLen=15)
        df = hisMonthStat(df, dayLen=30)
        df = hisMonthStat(df, dayLen=90)
        return df

    def getFeaDf(self, startDate=None, endDate='2017-09-01', periods=None, freq=30, predictLen=31):
        if periods == None:
            periods = (pd.to_datetime(endDate) - pd.to_datetime(startDate)).days // freq + 1
            startDate = None
        dateList = pd.date_range(start=startDate, end=endDate, periods=periods, freq='%dD'%freq)
        params = {
            'startDate': dateList[0],
            'endDate':dateList[-1],
            'freq':freq,
            }
        df = self.initDf(dateList, predictLen=predictLen, **params)
        df = self.addUserFea(df, **params)
        return df

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

def getRankDf(df, rankCol='buy_predict'):
    resultDf = df.sort_index(by=[rankCol], ascending=False).iloc[:50000]
    return resultDf

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
    # userList = userDf.sample(frac=0.1, random_state=0)['user_id'].values
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
    print('import dataset:', datetime.now() - startTime)

    # 特征工程
    feaFactory = FeaFactory(dfs)
    df = feaFactory.getFeaDf(startDate='2016-12-01', endDate='2017-09-01', freq=15)
    print('train user num:', df.date_range.value_counts())
    print('train day label:', df.day_label.value_counts())

    fea = [
        'age','sex','user_lv_cd','user_his_days',
        'date1111_daydelta','date618_daydelta',
        ]
    cateList = list(set(skuDf.cate))
    fea.extend(['user_cate%s_perday_view'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_perday_follow'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_perday_viewday'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_perday_followday'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_perday_sku'%x for x in [101,30,'all','task','other']])
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
    fea.extend(['user_cate%s_last15day_view'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last15day_follow'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_viewday'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last15day_followday'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last15day_sku'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_order'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_orderday'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_order_daymean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_ordersku_len'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_ordersku_sum'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_order_daymin'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_order_daymax'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_comm_mean'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last15day_order_daydelta_mean'%x for x in [101,30]])
    fea.extend(['cate%s_next_order_pred_by_last15day'%x for x in [101,30]])
    fea.extend(['user_cate%s_last30day_view'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last30day_follow'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last30day_viewday'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last30day_followday'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last30day_sku'%x for x in [101,30,'all','task','other']])
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
    fea.extend(['user_cate%s_last90day_view'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last90day_follow'%x for x in [101,30,'all','task','other']])
    fea.extend(['user_cate%s_last90day_viewday'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last90day_followday'%x for x in [101,30,'all','task','other']])
    # fea.extend(['user_cate%s_last90day_sku'%x for x in [101,30,'all','task','other']])
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

    fea.extend(['user_cate%s_last_view_timedelta'%x for x in cateList+['all']])
    # fea.extend(['user_cate%s_last_follow_timedelta'%x for x in cateList+['all']])
    fea.extend(['user_cate%s_last_order_timedelta'%x for x in cateList+['all']])
    fea.extend(['user_cate%s_last_order_view_timedelta'%x for x in cateList+['all']])
    # fea.extend(['user_cate%s_last_order_follow_timedelta'%x for x in cateList+['all']])
    print(df[fea].info(max_cols=500))

    # 导出特征
    exportResult(df[['user_id','date_range','day_label','buy_label']+fea], '../temp/yuna_feaFactory2.csv')

if __name__ == '__main__':
    finishTime = datetime.now()
    main()
    print('Finished program in: ', datetime.now() - finishTime)
