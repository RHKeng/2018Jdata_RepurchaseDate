#!/usr/bin/env python
# -*-coding:utf-8-*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import csv
from datetime import *
import math
from sklearn.preprocessing import MinMaxScaler

# 获取候选用户集
def getUserList():
    orderDf = pd.read_csv('../data/jdata_user_order.csv')
    orderDf['o_date'] = pd.to_datetime(orderDf['o_date'])
    orderDf['o_month'] = orderDf['o_date'].dt.month
    skuDf = pd.read_csv('../data/jdata_sku_basic_info.csv')
    orderDf = orderDf.merge(skuDf[['sku_id','cate']], how='left', on='sku_id')
    orderDf = orderDf[orderDf.cate.isin([101,30])]

    user578 = set(orderDf[orderDf.o_month.isin([5,7,8])]['user_id'])
    user78 = set(orderDf[orderDf.o_month.isin([7,8])]['user_id'])
    return user578

# 模型用户排序
def sortDf(dfK, dfL, dfW):
    dfK['rank'] = list(range(1,len(dfK)+1))
    dfL['rank'] = list(range(1,len(dfL)+1))
    dfW = dfW.sort_values(['o_num'], ascending=False)
    dfW['rank'] = list(range(1,len(dfW)+1))
    return dfK, dfL, dfW

# 截取前50000用户
def getHead(dfK, dfL, dfW):
    dfW = dfW.head(50000)
    dfL = dfL.head(50000)
    dfK = dfK.head(50000)
    return dfK, dfL, dfW

# 模型评分归一化
def scaleScore(dfK, dfL, dfW):
    scaler = MinMaxScaler()
    dfK['score'] = scaler.fit_transform(dfK[['predicted_is_order']])
    dfL['score'] = scaler.fit_transform(dfL[['pre']])
    dfW['score'] = scaler.fit_transform(dfW[['o_num']])
    return dfK, dfL, dfW

# 仅保留符合条件的用户记录
def filterUserList(dfK, dfL, dfW, user578):
    dfW = dfW[dfW.user_id.isin(user578)]
    dfL = dfL[dfL.user_id.isin(user578)]
    dfK = dfK[dfK.user_id.isin(user578)]
    return dfK, dfL, dfW

def main():
    # S1融合
    dfK = pd.read_csv('../temp/jdata_b_keng_result2.csv')
    dfW = pd.read_csv('../temp/submission.jw.csv.s1.csv')
    dfL = pd.read_csv('../temp/yinhu_all_user_id.csv')
    dfK, dfL, dfW = sortDf(dfK, dfL, dfW)
    dfK, dfL, dfW = getHead(dfK, dfL, dfW)
    dfK, dfL, dfW = scaleScore(dfK, dfL, dfW)
    user578 = getUserList()
    dfK, dfL, dfW = filterUserList(dfK, dfL, dfW, user578)
    df = pd.DataFrame({'user_id':list(user578)})
    df = df.merge(dfK[['user_id','rank','score']].rename(columns={'rank':'keng_rank', 'score':'keng_score'}), how='left', on='user_id')
    df = df.merge(dfW[['user_id','rank','score']].rename(columns={'rank':'wong_rank', 'score':'wong_score'}), how='left', on='user_id')
    df = df.merge(dfL[['user_id','rank','score']].rename(columns={'rank':'lake_rank', 'score':'lake_score'}), how='left', on='user_id')
    df.fillna({x:0 for x in ['lake_score','wong_score','keng_score']}, inplace=True)
    df['total_score'] = 0.9523*df['lake_score'] + 1*df['wong_score'] + 0.7256*df['keng_score']
    df = df.sort_values(['total_score'], ascending=False).head(50000)

    # S1与S2合并
    dfY = pd.read_csv('../temp/yuna_dateModel2_all.csv')
    df = df.merge(dfY, how='left', on='user_id')
    print('result count:', df[['user_id','pred_date']].count())
    print('result date:', df['pred_date'].value_counts())

    # 导出最终结果
    df[['user_id','pred_date']].to_csv('../result/jdata_b_submission.csv_test',index=False)

if __name__ == '__main__':
    finishTime = datetime.now()
    main()
    print('Finished program in: ', datetime.now() - finishTime)
