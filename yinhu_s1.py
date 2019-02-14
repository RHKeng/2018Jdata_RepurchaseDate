#!/usr/bin/env python
# -*-coding:utf-8-*-
# author: zhao yinhu


import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# import time
from sklearn.model_selection import train_test_split,StratifiedKFold,KFold,GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import scipy
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

#XGBoost model
class XgbModel:
    def __init__(self, feaNames=None, params={}):
        self.feaNames = feaNames
        self.params = {
            'objective': 'binary:logistic',
            'eta': 0.01,
            'colsample_bytree': 0.886,
            'min_child_weight': 1.2,
            'max_depth': 6,
            'subsample': 0.886,
            'gamma': 0.1,
            'lambda': 5,
            'eval_metric': 'logloss',
            'seed': 2018,
        }
        for k,v in params.items():
            self.params[k] = v
        self.clf = None

    def train(self, X, y, train_size=1, test_size=0.1, verbose=True, num_boost_round=2200, early_stopping_rounds=500):
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

    def trainCV(self, X, y, nFold=3, verbose=True, num_boost_round=2200, early_stopping_rounds=500):
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feaNames)
        watchlist = [(dtrain,'train')]
        cvResult = xgb.cv(
            self.params, dtrain,
            num_boost_round = num_boost_round,
            nfold = nFold,
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval=verbose
        )
        clf = xgb.train(
            self.params, dtrain,
            evals = watchlist,
            num_boost_round = cvResult.shape[0],
        )
        self.clf = clf

    def gridSearch(self, X, y, nFold=3, verbose=1, num_boost_round=150):
        paramsGrids = {
        'n_estimators': [50+5*i for i in range(0,30)],
        'gamma': [0+10*i for i in range(0,10)],
        'max_depth': list(range(3,10)),
        'min_child_weight': list(range(1,10)),
        #'subsample': [1-0.05*i for i in range(0,10)],
        'colsample_bytree': [1-0.05*i for i in range(0,10)],
        #'reg_alpha':[1000+100*i for i in range(0,20)],
        'max_delta_step': [0+1*i for i in range(0,8)]
        }
        gsearch = GridSearchCV(
            estimator = xgb.XGBClassifier(
                #max_depth = self.params['max_depth'],
                # gamma = self.params['gamma'],
                learning_rate = self.params['eta'],
                #max_delta_step = self.params['max_delta_step'],
                #min_child_weight = self.params['min_child_weight'],
                subsample = self.params['subsample'],
                #colsample_bytree = self.params['colsample_bytree'],
                silent = self.params['silent'],
                # reg_alpha = self.params['alpha'],
                #n_estimators = num_boost_round
            ),
            param_grid = paramsGrids,
            scoring = 'neg_log_loss',
            cv = nFold,
            verbose = verbose,
            n_jobs = -1
        )
        gsearch.fit(X, y)
        print(pd.DataFrame(gsearch.cv_results_))
        print(gsearch.best_params_)
        exit()

    def predict(self, X):
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

#导入数据
def importDf(url, sep=' ', header='infer', index_col=None):
	df = pd.read_csv(url, sep=sep, header=header, index_col=index_col)
	return df

#划窗特征函数
def window_features(df,jd_user_order,jd_sku_basic_info,end_day,gap):
    start_day=(datetime.strptime(end_day,'%Y-%m-%d')-timedelta(days=gap)).strftime('%Y-%m-%d')
    jd_user_order=jd_user_order[(jd_user_order.o_date>=start_day)&(jd_user_order.o_date<end_day)]
    jd_user_order=jd_user_order.merge(jd_sku_basic_info,'left','sku_id')
    jd_user_order=jd_user_order[jd_user_order.cate.isin([101,30])]
    order_total=jd_user_order.groupby('user_id')['o_sku_num'].sum().reset_index().rename(columns={'o_sku_num':'%d_day_order_total'%gap})
    df=pd.merge(df,order_total,'left','user_id')
    return df

def main():
	#特征列表
	features=['age','sex','user_lv_cd','a_type_1_mean','a_type_2_mean','a_type_1_max','a_type_2_max','a_type_1_min','a_type_2_min',\
		'a_type_1_var','a_type_2_var','action_days','last_action_day','action_gap_mean','action_gap_var','action_gap_min','action_gap_mid',\
		'a_type_1_total','action_month_total','all_a_type_1_var','all_a_type_2_var','all_action_days','all_action_month_total','all_last_action_day',\
		'all_action_gap_mean','all_action_gap_var','all_a_type_1_total','all_a_type_2_total','order_total','last_order_para1','last_order_para2',\
		'last_order_price','last_order_total','first_order_para1','last_order_gap','order_gap_mean','order_gap_var','order_gap_min','order_gap_mid',\
		'user_order_days','user_order_month_max','user_order_month_total','last_order_days_max','last_order_days_min','last_order_days_mean',\
		'last_order_days_mid','o_area_like','day_of_week_like','all_last_order_gap','cate_like','comment_total','last_comment_gap','action_no_order',\
		'10_day_order_total', '20_day_order_total', '30_day_order_total', '50_day_order_total', '70_day_order_total']

	#导入原始数据
	jd_user_order=pd.read_csv('../data/jdata_user_order.csv')
	jd_user_action=pd.read_csv('../data/jdata_user_action.csv')
	jd_user_basic_info=pd.read_csv('../data/jdata_user_basic_info.csv')
	jd_user_comment_score=pd.read_csv('../data/jdata_user_comment_score.csv')
	jd_sku_basic_info=pd.read_csv('../data/jdata_sku_basic_info.csv')

	#删除有空缺值的样本
	jd_user_action=jd_user_action.dropna()

	#格式化日期
	jd_user_order['o_date']=pd.to_datetime(jd_user_order['o_date'])
	jd_user_action['a_date']=pd.to_datetime(jd_user_action['a_date'])
	jd_user_comment_score['comment_create_tm']=pd.to_datetime(jd_user_comment_score['comment_create_tm'])

	#根据日期排序
	jd_user_order=jd_user_order.sort_values(['user_id','o_date'])
	jd_user_action=jd_user_action.sort_values(['user_id','a_date'])

	#年月日星期几
	jd_user_order['year']=jd_user_order['o_date'].dt.year
	jd_user_order['month']=jd_user_order['o_date'].dt.month
	jd_user_order['day']=jd_user_order['o_date'].dt.day
	jd_user_order['day_of_week']=jd_user_order['o_date'].dt.weekday

	jd_user_action['year']=jd_user_action['a_date'].dt.year
	jd_user_action['month']=jd_user_action['a_date'].dt.month
	jd_user_action['day']=jd_user_action['a_date'].dt.day
	jd_user_action['day_of_week']=jd_user_action['a_date'].dt.weekday

	#季度
	#jd_user_order['year_month']=str(jd_user_order['year'])+'-'+str(jd_user_order['month'])
	#jd_user_action['year_month']=str(jd_user_action['year'])+'-'+str(jd_user_action['month'])
	jd_user_order['year_month']=jd_user_order['o_date'].apply(lambda x:str(x)[0:7])
	jd_user_action['year_month']=jd_user_action['a_date'].apply(lambda x:str(x)[0:7])

	#用户距离上次行为的天数
	jd_user_order['dist_from_last_behav']=jd_user_order.groupby('user_id')['o_date'].diff()
	jd_user_order['dist_from_last_behav']=jd_user_order['dist_from_last_behav'].dt.days
	jd_user_action['dist_from_last_behav']=jd_user_action.groupby('user_id')['a_date'].diff()
	jd_user_action['dist_from_last_behav']=jd_user_action['dist_from_last_behav'].dt.days

	#组合数据
	jd_user_order_sku=pd.merge(jd_user_order,jd_sku_basic_info,on=['sku_id'],how='left')
	jd_user_order_all=pd.merge(jd_user_order_sku,jd_user_basic_info,on=['user_id'],how='left')
	jd_user_action_sku=pd.merge(jd_user_action,jd_sku_basic_info,on=['sku_id'],how='left')
	jd_user_action_all=pd.merge(jd_user_action_sku,jd_user_basic_info,on=['user_id'],how='left')
	#jd_user_comment_score_user_order=pd.merge(jd_user_comment_score,jd_user_order,on=['o_id'],how='left')
	#jd_user_comment_score_all=pd.merge(jd_user_comment_score_user_order,jd_sku_basic_info,on=['sku_id'],how='left')

	################################################################################################################
	################################################################################################################
	################################################################################################################
	#构造训练集
	#构建用户集合，考虑目标品类
	start_day="2017-06-03"
	end_day="2017-08-01"
	train_set=jd_user_order_all[(jd_user_order_all.o_date<end_day)&(jd_user_order_all.o_date>=start_day)]
	train_set=train_set[train_set.cate.isin([30,101])]
	train_set=train_set.drop_duplicates(subset='user_id')

	#提取特征
	#行为特征
	#提取跟目标品类有关的特征
	start_day="2017-01-03"
	end_day="2017-08-01"
	train_set_action=jd_user_action_all[(jd_user_action_all.a_date<end_day)&(jd_user_action_all.a_date>=start_day)]
	train_set_action=train_set_action[train_set_action.cate.isin([30,101])]
	a_type=pd.get_dummies(train_set_action['a_type'],prefix='a_type')
	train_set_action=pd.concat([train_set_action,a_type],axis=1)
	'''
	1、行为特征（最小、最大、均值、方差、总数）
	'''
	t=train_set_action[['user_id','sku_id','a_type_1','a_type_2']]
	t=t.groupby(['user_id', 'sku_id']).agg('sum').reset_index()

	#最小
	t1=t[['user_id','a_type_1','a_type_2']]
	t1=t1.groupby(['user_id']).agg('min').reset_index()
	t1.rename(columns={'a_type_1':'a_type_1_min','a_type_2':'a_type_2_min'},inplace=True)
	train_set=pd.merge(train_set,t1,on=['user_id'],how='left')

	#最大
	t1=t[['user_id','a_type_1','a_type_2']]
	t1=t1.groupby(['user_id']).agg('max').reset_index()
	t1.rename(columns={'a_type_1':'a_type_1_max','a_type_2':'a_type_2_max'},inplace=True)
	train_set=pd.merge(train_set,t1,on=['user_id'],how='left')

	#均值
	t1=t[['user_id','a_type_1','a_type_2']]
	t1=t1.groupby(['user_id']).agg('mean').reset_index()
	t1.rename(columns={'a_type_1':'a_type_1_mean','a_type_2':'a_type_2_mean'},inplace=True)
	train_set=pd.merge(train_set,t1,on=['user_id'],how='left')

	#方差
	t1=t[['user_id','a_type_1','a_type_2']]
	t1=t1.groupby(['user_id']).agg('var').reset_index()
	t1.rename(columns={'a_type_1':'a_type_1_var','a_type_2':'a_type_2_var'},inplace=True)
	train_set=pd.merge(train_set,t1,on=['user_id'],how='left')

	#浏览总数
	t1=t[['user_id','a_type_1']]
	t1=t1.groupby(['user_id']).sum().reset_index()
	t1.rename(columns={'a_type_1':'a_type_1_total'},inplace=True)
	train_set=pd.merge(train_set,t1,on=['user_id'],how='left')

	'''
	2、活动天数特征
	'''
	'''
	t=train_set_action[['user_id','a_date']]
	t1=t.groupby(['user_id','a_date']).size().reset_index()
	t1.rename(columns={0: 'action_times_each_day'})
	train_set=pd.merge(train_set,t1,on=['user_id'],how='left')

	t1=t.groupby(['user_id']).size().reset_index()
	t1.rename(columns={'action_times_each_day': 'action_days'})
	train_set=pd.merge(train_set,t1,on=['user_id'],how='left')
	'''
	t=train_set_action[['user_id','a_date']]
	t1=t.groupby(['user_id','a_date']).size().reset_index()
	t1=t1.groupby(['user_id']).size().reset_index()
	t1.rename(columns={0: 'action_days'},inplace=True)
	train_set=pd.merge(train_set,t1,on=['user_id'],how='left')

	'''
	3、活动的最后一天
	'''
	t=train_set_action[['user_id','a_date']]
	t=t.groupby('user_id').nth(-1).reset_index()
	t['last_action_day']=(pd.to_datetime(end_day)-t['a_date']).dt.days
	t=t[['user_id','last_action_day']]
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	'''
	4、活动天数间隔（最小、均值、中值、方差等等）
	'''
	t=train_set_action[['user_id','a_date']]
	t=t.drop_duplicates(subset=['user_id','a_date'])
	t['action_gap']=t.groupby('user_id')['a_date'].diff().dt.days
	t=t.groupby('user_id')['action_gap'].agg({'action_gap_mean':'mean','action_gap_var':'var','action_gap_min':'min','action_gap_mid':'median',}).reset_index()
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	'''
	5、活动月数特征
	'''
	t=train_set_action[['user_id','year_month']]
	t=t.groupby(['user_id','year_month']).size().reset_index()
	t=t.groupby('user_id').size().reset_index().rename(columns={0: 'action_month_total'})
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	#提取所有品类有关的特征
	train_set_action=jd_user_action_all[(jd_user_action_all.a_date<end_day)&(jd_user_action_all.a_date>=start_day)]
	a_type=pd.get_dummies(train_set_action['a_type'],prefix='a_type')
	train_set_action=pd.concat([train_set_action,a_type],axis=1)
	'''
	1、行为特征（总数、方差）
	'''
	t=train_set_action[['user_id','sku_id','a_type_1','a_type_2']]
	t=t.groupby(['user_id', 'sku_id']).agg('sum').reset_index()

	#总数
	t1=t[['user_id','a_type_1','a_type_2']]
	t1=t1.groupby(['user_id']).agg('sum').reset_index()
	t1.rename(columns={'a_type_1':'all_a_type_1_total','a_type_2':'all_a_type_2_total'},inplace=True)
	train_set=pd.merge(train_set,t1,on=['user_id'],how='left')

	#方差
	t1=t[['user_id','a_type_1','a_type_2']]
	t1=t1.groupby(['user_id']).agg('var').reset_index()
	t1.rename(columns={'a_type_1':'all_a_type_1_var','a_type_2':'all_a_type_2_var'},inplace=True)
	train_set=pd.merge(train_set,t1,on=['user_id'],how='left')
	'''
	2、活动天数特征
	'''
	t=train_set_action[['user_id','a_date']]
	t1=t.groupby(['user_id','a_date']).size().reset_index()
	t1=t1.groupby(['user_id']).size().reset_index()
	t1.rename(columns={0: 'all_action_days'},inplace=True)
	train_set=pd.merge(train_set,t1,on=['user_id'],how='left')

	'''
	3、活动的最后一天
	'''
	t=train_set_action[['user_id','a_date']]
	t=t.groupby('user_id').nth(-1).reset_index()
	t['all_last_action_day']=(pd.to_datetime(end_day)-t['a_date']).dt.days
	t=t[['user_id','all_last_action_day']]
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')
	'''
	4、活动月数特征
	'''
	t=train_set_action[['user_id','year_month']]
	t=t.groupby(['user_id','year_month']).size().reset_index()
	t=t.groupby('user_id').size().reset_index().rename(columns={0: 'all_action_month_total'})
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	'''
	5、行为天数间隔的统计
	'''
	t=train_set_action.groupby('user_id')['dist_from_last_behav'].agg({'all_action_gap_mean':'mean','all_action_gap_var':'var',}).reset_index()
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	#下单特征
	#提取跟目标品类有关的特征
	train_set_order=jd_user_order_all[(jd_user_order_all.o_date<end_day)&(jd_user_order_all.o_date>=start_day)]
	train_set_order=train_set_order[train_set_order.cate.isin([30,101])]
	'''
	1、下单总数
	'''
	t=train_set_order.groupby('user_id')['o_sku_num'].sum().reset_index().rename(columns={'o_sku_num':'order_total'})
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')
	'''
	2、最近一次下单的一些特征(para_1、para_2、o_sku_num)
	'''
	t=train_set_order.sort_values(by='o_date', ascending=False).drop_duplicates('user_id')
	t=t[['user_id','para_1','para_2','price','o_sku_num']]
	t.columns=['user_id','last_order_para1','last_order_para2','last_order_price','last_order_total']
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	'''
	3、用户第一次下单的一些特征(para_1、para_2、o_sku_num)
	'''
	t=train_set_order[['user_id','para_1']]
	t=t.groupby('user_id').nth(0).reset_index()
	t.columns=['user_id','first_order_para1']
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	'''
	4、最后下单天数间隔特征###########################################################################
	##################################################################################################
	'''
	t=train_set_order.drop_duplicates(['user_id','o_date',])
	t['order_gap']=t.groupby('user_id')['o_date'].diff().dt.days
	t1=t.groupby('user_id').nth(-1).reset_index()
	t1=t1[['user_id','order_gap']].rename(columns={'order_gap':'last_order_gap'})
	train_set=pd.merge(train_set,t1,on=['user_id'],how='left')

	t1=t.groupby('user_id')['order_gap'].agg({'order_gap_mean':'mean','order_gap_var':'var','order_gap_min':'min','order_gap_mid':'median',}).reset_index()
	train_set=pd.merge(train_set,t1,on=['user_id'],how='left')

	'''
	5、用户下单天数
	'''
	t=train_set_order[['user_id', 'o_date']]
	t=t.groupby(['user_id','o_date']).size().reset_index()
	t=t.groupby('user_id').size().reset_index().rename(columns={0: 'user_order_days'})
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	'''
	6、用户下单月数
	'''
	t=train_set_order[['user_id', 'year_month']]
	t=t.groupby(['user_id','year_month']).size().reset_index()
	t=t.groupby('user_id')[0].agg({'user_order_month_max':'max','user_order_month_total':'count'}).reset_index()
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	'''
	7、用户最后一次下单间隔的统计
	'''
	t=train_set_order[['user_id','o_date']]
	t=t.drop_duplicates(['user_id','o_date'])
	t['last_order_days'] = (pd.to_datetime(end_day)-t['o_date']).dt.days
	t=t.groupby('user_id')['last_order_days'].agg({'last_order_days_max':'max','last_order_days_min':'min','last_order_days_mean':'mean','last_order_days_mid':'median'}).reset_index()
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	'''
	8、用户下单的区域倾向、day_of_week倾向
	'''
	t=train_set_order.groupby('user_id')['o_area','day_of_week',].agg(lambda x:scipy.stats.mode(x)[0][0]).reset_index()
	t.columns=['user_id','o_area_like','day_of_week_like']
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	#提取所有品类有关的特征
	start_day="2017-02-02"
	end_day="2017-08-01"
	train_set_order=jd_user_order_all[(jd_user_order_all.o_date<end_day)&(jd_user_order_all.o_date>=start_day)]
	'''
	1、最后一次下单间隔
	'''
	t=train_set_order[['user_id','o_date',]]
	t=t.sort_values(by='o_date', ascending=False).drop_duplicates('user_id')
	t['o_date'] = (pd.to_datetime(end_day)-t['o_date']).dt.days
	t.columns=['user_id','all_last_order_gap',]
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	'''
	2、用户品类倾向
	'''
	t=train_set_order.groupby('user_id')['cate'].agg(lambda x: scipy.stats.mode(x)[0][0]).reset_index()
	t.columns=['user_id','cate_like']
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	#用户评论特征
	start_day="2017-06-02"
	end_day="2017-08-01"
	train_set_comment=jd_user_comment_score[(jd_user_comment_score.comment_create_tm<end_day)&(jd_user_comment_score.comment_create_tm>=start_day)]
	train_set_comment=pd.merge(train_set_comment,jd_user_order[['o_id','sku_id','o_date','o_area','o_sku_num']],'left','o_id')
	train_set_comment=pd.merge(train_set_comment,jd_sku_basic_info,'left','sku_id')
	'''
	#1、评论数量
	'''
	t=train_set_comment.groupby('user_id').size().reset_index().rename(columns={0:'comment_total'})
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')
	'''
	#2、最后一次评论的间隔
	'''
	t=train_set_comment[['user_id','comment_create_tm',]]
	t=t.sort_values(by='comment_create_tm', ascending=False).drop_duplicates('user_id')
	t['comment_create_tm']=(pd.to_datetime(end_day)-t['comment_create_tm']).dt.days
	t.columns = ['user_id','last_comment_gap',]
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	#交叉特征
	#提取跟目标品类有关的特征
	start_day="2017-07-27"
	end_day="2017-08-01"
	train_set_action=jd_user_action_all[(jd_user_action_all.a_date<end_day)&(jd_user_action_all.a_date>=start_day)]
	train_set_action=train_set_action[train_set_action.cate.isin([30,101])]
	train_set_order=jd_user_order_all[(jd_user_order_all.o_date<end_day)&(jd_user_order_all.o_date>=start_day)]
	train_set_order=train_set_order[train_set_order.cate.isin([30,101])]
	'''
	#用户有行为但是没有买
	'''
	train_set_order=train_set_order.drop_duplicates(['user_id','o_date','sku_id'])
	train_set_action=train_set_action.drop_duplicates(['user_id','a_date','sku_id'])
	train_set_order['date']=train_set_order['o_date']
	train_set_action['date']=train_set_action['a_date']
	train_set_action=train_set_action.merge(train_set_order,'left',['user_id','date','sku_id'])
	t=train_set_action.groupby('user_id')['o_date'].apply(lambda x: x.isnull().sum()).reset_index().rename(columns={'o_date':'action_no_order'})
	train_set=pd.merge(train_set,t,on=['user_id'],how='left')

	#划窗特征
	end_day="2017-08-01"
	win_length_list=[10,20,30,50,70]
	for gap in win_length_list:
		train_set=window_features(train_set,jd_user_order,jd_sku_basic_info,end_day,gap)

	#生成标签
	#构建标签
	#考虑目标品类
	start_day="2017-08-01"
	end_day="2017-08-31"
	train_label=jd_user_order_all[(jd_user_order_all.o_date<end_day)&(jd_user_order_all.o_date>=start_day)]
	train_label=train_label[train_label.cate.isin([30,101])]
	train_label=train_label[['user_id']].drop_duplicates()
	train_label['label']=1

	train_set=pd.merge(train_set,train_label,on=['user_id'],how='left')
	train_set=train_set.fillna({'label': 0})

	################################################################################################################
	################################################################################################################
	################################################################################################################
	#构造测试集
	#构建用户集合，考虑目标品类
	start_day="2017-06-18"
	end_day="2017-09-01"
	test_set=jd_user_order_all[(jd_user_order_all.o_date<end_day)&(jd_user_order_all.o_date>=start_day)]
	test_set=test_set[test_set.cate.isin([30,101])]
	test_set=test_set.drop_duplicates(subset='user_id')

	#提取特征
	#行为特征
	#提取跟目标品类有关的特征
	start_day="2017-02-03"
	end_day="2017-09-01"
	test_set_action=jd_user_action_all[(jd_user_action_all.a_date<end_day)&(jd_user_action_all.a_date>=start_day)]
	test_set_action=test_set_action[test_set_action.cate.isin([30,101])]
	a_type=pd.get_dummies(test_set_action['a_type'],prefix='a_type')
	test_set_action=pd.concat([test_set_action,a_type],axis=1)
	'''
	1、行为特征（最小、最大、均值、方差）
	'''
	t=test_set_action[['user_id','sku_id','a_type_1','a_type_2']]
	t=t.groupby(['user_id', 'sku_id']).agg('sum').reset_index()

	#最小
	t1=t[['user_id','a_type_1','a_type_2']]
	t1=t1.groupby(['user_id']).agg('min').reset_index()
	t1.rename(columns={'a_type_1':'a_type_1_min','a_type_2':'a_type_2_min'},inplace=True)
	test_set=pd.merge(test_set,t1,on=['user_id'],how='left')

	#最大
	t1=t[['user_id','a_type_1','a_type_2']]
	t1=t1.groupby(['user_id']).agg('max').reset_index()
	t1.rename(columns={'a_type_1':'a_type_1_max','a_type_2':'a_type_2_max'},inplace=True)
	test_set=pd.merge(test_set,t1,on=['user_id'],how='left')

	#均值
	t1=t[['user_id','a_type_1','a_type_2']]
	t1=t1.groupby(['user_id']).agg('mean').reset_index()
	t1.rename(columns={'a_type_1':'a_type_1_mean','a_type_2':'a_type_2_mean'},inplace=True)
	test_set=pd.merge(test_set,t1,on=['user_id'],how='left')

	#方差
	t1=t[['user_id','a_type_1','a_type_2']]
	t1=t1.groupby(['user_id']).agg('var').reset_index()
	t1.rename(columns={'a_type_1':'a_type_1_var','a_type_2':'a_type_2_var'},inplace=True)
	test_set=pd.merge(test_set,t1,on=['user_id'],how='left')

	#浏览总数
	t1=t[['user_id','a_type_1']]
	t1=t1.groupby(['user_id']).sum().reset_index()
	t1.rename(columns={'a_type_1':'a_type_1_total'},inplace=True)
	test_set=pd.merge(test_set,t1,on=['user_id'],how='left')

	'''
	2、活动天数特征
	'''
	'''
	t=test_set_action[['user_id','a_date']]
	t1=t.groupby(['user_id','a_date']).size().reset_index()
	t1.rename(columns={0: 'action_times_each_day'})
	test_set=pd.merge(test_set,t1,on=['user_id'],how='left')

	t1=t.groupby(['user_id']).size().reset_index()
	t1.rename(columns={'action_times_each_day': 'action_days'})
	test_set=pd.merge(test_set,t1,on=['user_id'],how='left')
	'''
	t=test_set_action[['user_id','a_date']]
	t1=t.groupby(['user_id','a_date']).size().reset_index()
	t1=t1.groupby(['user_id']).size().reset_index()
	t1.rename(columns={0: 'action_days'},inplace=True)
	test_set=pd.merge(test_set,t1,on=['user_id'],how='left')

	'''
	3、活动的最后一天
	'''
	t=test_set_action[['user_id','a_date']]
	t=t.groupby('user_id').nth(-1).reset_index()
	t['last_action_day']=(pd.to_datetime(end_day)-t['a_date']).dt.days
	t=t[['user_id','last_action_day']]
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	'''
	4、活动天数间隔（最小、均值、中值、方差等等）
	'''
	t=test_set_action[['user_id','a_date']]
	t=t.drop_duplicates(subset=['user_id','a_date'])
	t['action_gap']=t.groupby('user_id')['a_date'].diff().dt.days
	t=t.groupby('user_id')['action_gap'].agg({'action_gap_mean':'mean','action_gap_var':'var','action_gap_min':'min','action_gap_mid':'median',}).reset_index()
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	'''
	5、活动月数特征
	'''
	t=test_set_action[['user_id','year_month']]
	t=t.groupby(['user_id','year_month']).size().reset_index()
	t=t.groupby('user_id').size().reset_index().rename(columns={0: 'action_month_total'})
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	#提取所有品类有关的特征
	test_set_action=jd_user_action_all[(jd_user_action_all.a_date<end_day)&(jd_user_action_all.a_date>=start_day)]
	a_type=pd.get_dummies(test_set_action['a_type'],prefix='a_type')
	test_set_action=pd.concat([test_set_action,a_type],axis=1)
	'''
	1、行为特征（总数、方差）
	'''
	t=test_set_action[['user_id','sku_id','a_type_1','a_type_2']]
	t=t.groupby(['user_id', 'sku_id']).agg('sum').reset_index()

	#总数
	t1=t[['user_id','a_type_1','a_type_2']]
	t1=t1.groupby(['user_id']).agg('sum').reset_index()
	t1.rename(columns={'a_type_1':'all_a_type_1_total','a_type_2':'all_a_type_2_total'},inplace=True)
	test_set=pd.merge(test_set,t1,on=['user_id'],how='left')

	#方差
	t1=t[['user_id','a_type_1','a_type_2']]
	t1=t1.groupby(['user_id']).agg('var').reset_index()
	t1.rename(columns={'a_type_1':'all_a_type_1_var','a_type_2':'all_a_type_2_var'},inplace=True)
	test_set=pd.merge(test_set,t1,on=['user_id'],how='left')
	'''
	2、活动天数特征
	'''
	t=test_set_action[['user_id','a_date']]
	t1=t.groupby(['user_id','a_date']).size().reset_index()
	t1=t1.groupby(['user_id']).size().reset_index()
	t1.rename(columns={0: 'all_action_days'},inplace=True)
	test_set=pd.merge(test_set,t1,on=['user_id'],how='left')

	'''
	3、活动的最后一天
	'''
	t=test_set_action[['user_id','a_date']]
	t=t.groupby('user_id').nth(-1).reset_index()
	t['all_last_action_day']=(pd.to_datetime(end_day)-t['a_date']).dt.days
	t=t[['user_id','all_last_action_day']]
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')
	'''
	4、活动月数特征
	'''
	t=test_set_action[['user_id','year_month']]
	t=t.groupby(['user_id','year_month']).size().reset_index()
	t=t.groupby('user_id').size().reset_index().rename(columns={0: 'all_action_month_total'})
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	'''
	5、行为天数间隔的统计
	'''
	t=test_set_action.groupby('user_id')['dist_from_last_behav'].agg({'all_action_gap_mean':'mean','all_action_gap_var':'var',}).reset_index()
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	#下单特征
	#提取跟目标品类有关的特征
	test_set_order=jd_user_order_all[(jd_user_order_all.o_date<end_day)&(jd_user_order_all.o_date>=start_day)]
	test_set_order=test_set_order[test_set_order.cate.isin([30,101])]
	'''
	1、下单总数
	'''
	t=test_set_order.groupby('user_id')['o_sku_num'].sum().reset_index().rename(columns={'o_sku_num':'order_total'})
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')
	'''
	2、最近一次下单的一些特征(para_1、para_2、o_sku_num)
	'''
	t=test_set_order.sort_values(by='o_date', ascending=False).drop_duplicates('user_id')
	t=t[['user_id','para_1','para_2','price','o_sku_num']]
	t.columns=['user_id','last_order_para1','last_order_para2','last_order_price','last_order_total']
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	'''
	3、用户第一次下单的一些特征(para_1、para_2、o_sku_num)
	'''
	t=test_set_order[['user_id','para_1']]
	t=t.groupby('user_id').nth(0).reset_index()
	t.columns=['user_id','first_order_para1']
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	'''
	4、最后下单天数间隔特征###########################################################################
	##################################################################################################
	'''
	t=test_set_order.drop_duplicates(['user_id','o_date',])
	t['order_gap']=t.groupby('user_id')['o_date'].diff().dt.days
	t1=t.groupby('user_id').nth(-1).reset_index()
	t1=t1[['user_id','order_gap']].rename(columns={'order_gap':'last_order_gap'})
	test_set=pd.merge(test_set,t1,on=['user_id'],how='left')

	t1=t.groupby('user_id')['order_gap'].agg({'order_gap_mean':'mean','order_gap_var':'var','order_gap_min':'min','order_gap_mid':'median',}).reset_index()
	test_set=pd.merge(test_set,t1,on=['user_id'],how='left')

	'''
	5、用户下单天数
	'''
	t=test_set_order[['user_id', 'o_date']]
	t=t.groupby(['user_id','o_date']).size().reset_index()
	t=t.groupby('user_id').size().reset_index().rename(columns={0: 'user_order_days'})
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	'''
	6、用户下单月数
	'''
	t=test_set_order[['user_id', 'year_month']]
	t=t.groupby(['user_id','year_month']).size().reset_index()
	t=t.groupby('user_id')[0].agg({'user_order_month_max':'max','user_order_month_total':'count'}).reset_index()
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	'''
	7、用户最后一次下单间隔的统计
	'''
	t=test_set_order[['user_id','o_date']]
	t=t.drop_duplicates(['user_id','o_date'])
	t['last_order_days'] = (pd.to_datetime(end_day)-t['o_date']).dt.days
	t=t.groupby('user_id')['last_order_days'].agg({'last_order_days_max':'max','last_order_days_min':'min','last_order_days_mean':'mean','last_order_days_mid':'median'}).reset_index()
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	'''
	8、用户下单的区域倾向、day_of_week倾向
	'''
	t=test_set_order.groupby('user_id')['o_area','day_of_week',].agg(lambda x:scipy.stats.mode(x)[0][0]).reset_index()
	t.columns=['user_id','o_area_like','day_of_week_like']
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')


	#提取所有品类有关的特征
	start_day="2017-03-05"
	end_day="2017-09-01"
	test_set_order=jd_user_order_all[(jd_user_order_all.o_date<end_day)&(jd_user_order_all.o_date>=start_day)]
	'''
	1、最后一次下单间隔
	'''
	t=test_set_order[['user_id','o_date',]]
	t=t.sort_values(by='o_date', ascending=False).drop_duplicates('user_id')
	t['o_date'] = (pd.to_datetime(end_day)-t['o_date']).dt.days
	t.columns=['user_id','all_last_order_gap',]
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	'''
	2、用户品类倾向
	'''
	t=test_set_order.groupby('user_id')['cate'].agg(lambda x: scipy.stats.mode(x)[0][0]).reset_index()
	t.columns=['user_id','cate_like']
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	#用户评论特征
	start_day="2017-07-03"
	end_day="2017-09-01"
	test_set_comment=jd_user_comment_score[(jd_user_comment_score.comment_create_tm<end_day)&(jd_user_comment_score.comment_create_tm>=start_day)]
	test_set_comment=pd.merge(test_set_comment,jd_user_order[['o_id','sku_id','o_date','o_area','o_sku_num']],'left','o_id')
	test_set_comment=pd.merge(test_set_comment,jd_sku_basic_info,'left','sku_id')
	'''
	#1、评论数量
	'''
	t=test_set_comment.groupby('user_id').size().reset_index().rename(columns={0:'comment_total'})
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')
	'''
	#2、最后一次评论的间隔
	'''
	t=test_set_comment[['user_id','comment_create_tm',]]
	t=t.sort_values(by='comment_create_tm', ascending=False).drop_duplicates('user_id')
	t['comment_create_tm']=(pd.to_datetime(end_day)-t['comment_create_tm']).dt.days
	t.columns = ['user_id','last_comment_gap',]
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	#交叉特征
	#提取跟目标品类有关的特征
	start_day="2017-08-27"
	end_day="2017-09-01"
	test_set_action=jd_user_action_all[(jd_user_action_all.a_date<end_day)&(jd_user_action_all.a_date>=start_day)]
	test_set_action=test_set_action[test_set_action.cate.isin([30,101])]
	test_set_order=jd_user_order_all[(jd_user_order_all.o_date<end_day)&(jd_user_order_all.o_date>=start_day)]
	test_set_order=test_set_order[test_set_order.cate.isin([30,101])]
	'''
	#用户有行为但是没有买
	'''
	test_set_order=test_set_order.drop_duplicates(['user_id','o_date','sku_id'])
	test_set_action=test_set_action.drop_duplicates(['user_id','a_date','sku_id'])
	test_set_order['date']=test_set_order['o_date']
	test_set_action['date']=test_set_action['a_date']
	test_set_action=test_set_action.merge(test_set_order,'left',['user_id','date','sku_id'])
	t=test_set_action.groupby('user_id')['o_date'].apply(lambda x: x.isnull().sum()).reset_index().rename(columns={'o_date':'action_no_order'})
	test_set=pd.merge(test_set,t,on=['user_id'],how='left')

	#划窗特征
	end_day="2017-09-01"
	win_length_list=[10,20,30,50,70]
	for gap in win_length_list:
		test_set=window_features(test_set,jd_user_order,jd_sku_basic_info,end_day,gap)

	##################################################################################################################
	##################################################################################################################
	##################################################################################################################
	#格式化训练集和测试集
	X=train_set[features].values
	y=train_set['label'].values
	test_X=test_set[features].values
	test=test_set[['user_id']].copy()
	#超参数
	params = {
	          'metric': 'auc',
	          'objective': 'binary',
	          'learning_rate': 0.03,
	          }
	cv = 0
	kfold=5
	test['pre'] = 0.0
	skf = StratifiedKFold(n_splits=kfold, random_state=502)
	for i, (train_index, test_index) in enumerate(skf.split(X, y)):
		print(' lgb kfold: {}  of  {} : '.format(i+1, kfold))
		params['seed'] = i
		X_train,X_dev = X[train_index],X[test_index]
		y_train,y_dev = y[train_index],y[test_index]
		dtrain=lgb.Dataset(X_train, label=y_train)
		dval=lgb.Dataset(X_dev, label=y_dev)
		lgb_model = lgb.train(params, dtrain, 5000, dval, verbose_eval=True,early_stopping_rounds=50,)
		cv += roc_auc_score(y_dev, lgb_model.predict(X_dev))/kfold
		test['pre'] += lgb_model.predict(test_X)/kfold
	print('交叉验证结果：',cv)
	#######################################################################################################################
	test=test.sort_values('pre',ascending=False)
	test[['user_id','pre']].to_csv('../temp/yinhu_all_user_id.csv',index=False)
	test=test.head(50000)
	#######################################################################################################################
	test['pred_date'] = '2017-09-05'
	test[['user_id','pred_date']].to_csv('../temp/yinhu_user_id_50000.csv',index=False)

if __name__=="__main__":
	main()
