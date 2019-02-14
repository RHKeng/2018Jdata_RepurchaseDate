#! /usr/bin/env python
# -*- coding: utf-8 -*-
# 代码有问题别找我！虽然是我写的，并且我觉得它是没问题的，如果不是你的操作原因，或许是它自己长歪了！
from __future__ import absolute_import
import gc
import logging
import warnings
import pandas as pd
import lightgbm as lgb
from .utils import *
from datetime import datetime, timedelta
from dateutil.rrule import rrule, MONTHLY
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedKFold


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)


class FeatureSelection(object):
	def __init__(self, rebuild=False):
		"""
		原始数据处理以及特征工程
		:param rebuild: 是否重新执行特征处理
		"""
		self.inpath = os.path.abspath(RAW_PATH)
		self.outpath = os.path.abspath(TEMP_PATH)
		self.modelpath = os.path.abspath(MODEL_PATH)
		self.rebuild = rebuild
		self.__cache__ = dict()

	def input(self, fname):
		"""
		获取输入文件路径
		:param fname: 文件名
		"""
		fpath = os.path.join(self.inpath, fname)
		if not os.path.isfile(fpath):
			return
		return fpath

	def output(self, fname):
		"""
		获取输出文件路径
		:param fname: 文件名
		"""
		fpath = os.path.join(self.outpath, fname)
		fdir = os.path.dirname(fpath)
		if not os.path.isdir(fdir):
			os.makedirs(fdir)
		return fpath

	def model(self, fname):
		"""
		获取模型文件路径
		:param fname: 文件名
		"""
		fpath = os.path.join(self.modelpath, fname)
		fdir = os.path.dirname(fpath)
		if not os.path.isdir(fdir):
			os.makedirs(fdir)
		return fpath

	def date_process(self, date, check_cache=True, prefix=''):
		"""
		时间处理，datetime特征获得对应年月日周
		:param date: datetime特征
		:param check_cache: 是否从内存中得到结果
		:param prefix: 新特征前缀
		"""
		if check_cache and date in self.__cache__:
			tmp = {i: self.__cache__[date].get(i, -1) for i in
			       ('year', 'month', 'day', 'week_day')}
		else:
			tmp = {'year': date.year, 'month': date.month,
			       'day': date.day, 'week_day': date.weekday()}
			self.__cache__[date] = tmp
		if len(prefix):
			return {"{0}_{1}".format(prefix, i): j for i, j in tmp.items()}
		return tmp

	@property
	def raw_data(self):
		"""
		整合原始数据，用于挖掘特征
		"""
		logging.info("Start load data from {0}".format(self.outpath))
		f_action = self.output('jw.user_action.csv')
		f_order = self.output('jw.user_order.csv')

		if os.path.isfile(f_action) and os.path.isfile(f_order):
			return pd.read_csv(f_action), pd.read_csv(f_order)

		logging.info("Load data fail, try to rebuild !")
		user_info = pd.read_csv(self.input("jdata_user_basic_info.csv"))
		sku_info = pd.read_csv(self.input("jdata_sku_basic_info.csv"))
		user_action = pd.read_csv(self.input("jdata_user_action.csv"))
		user_order = pd.read_csv(self.input("jdata_user_order.csv"))
		user_comment = pd.read_csv(self.input("jdata_user_comment_score.csv"))
		user_action = user_action[user_action['sku_id'] != -1]

		user_order.loc[:, 'o_date'] = pd.to_datetime(user_order['o_date'])
		user_action.loc[:, 'a_date'] = pd.to_datetime(user_action['a_date'])
		user_comment.loc[:, 'c_date'] = pd.to_datetime(user_comment['comment_create_tm'])
		user_comment = user_comment.drop('comment_create_tm', axis=1)

		new_order_cols = pd.DataFrame(list(user_order['o_date'].apply(self.date_process, prefix='o')))
		new_action_cols = pd.DataFrame(list(user_action['a_date'].apply(self.date_process, prefix='a')))
		new_comment_cols = pd.DataFrame(list(user_comment['c_date'].apply(self.date_process, prefix='c')))

		user_action = pd.merge(user_action, new_action_cols, left_index=True, right_index=True, how='outer')
		user_order = pd.merge(user_order, new_order_cols, left_index=True, right_index=True, how='outer')
		user_comment = pd.merge(user_comment, new_comment_cols, left_index=True, right_index=True, how='outer')

		df_action = pd.merge(user_action, user_info, on='user_id', how='left')
		df_action = pd.merge(df_action, sku_info, on='sku_id', how='left')
		df_action.to_csv(f_action, index=False, header=True)

		df_order = pd.merge(user_order, user_comment, on=['user_id', 'o_id'], how='left')
		df_order = pd.merge(df_order, user_info, on='user_id', how='left')
		df_order = pd.merge(df_order, sku_info, on='sku_id', how='left')
		df_order.to_csv(f_order, index=False, header=True)

		return df_action, df_order

	def feature_encode(self, cal_start=None, cal_stop=None):
		"""
		获取目标时间段内的特征
		:param cal_start: 训练集起始时间，默认为全体用户最早购买记录时间
		:param cal_stop: 训练集结束时间，默认为全体用户最后购买记录时间
		"""
		train_file = self.output('jw.train_set.csv')
		test_file = self.output('jw.test_set.csv')
		if not self.rebuild and (os.path.isfile(train_file) and os.path.isfile(test_file)):
			return pd.read_csv(train_file), pd.read_csv(test_file)

		df_action, df_order = self.raw_data
		user_info = pd.read_csv(self.input("jdata_user_basic_info.csv"))
		logging.info("Load data complete !")

		df_action.loc[:, 'a_date'] = pd.to_datetime(df_action['a_date'])
		df_order.loc[:, 'o_date'] = pd.to_datetime(df_order['o_date'])
		df_order.loc[:, 'c_date'] = pd.to_datetime(df_order['c_date'])

		df_order.replace(['-1', -1, -1.0], np.nan, inplace=True)
		df_action.replace(['-1', -1, -1.0], np.nan, inplace=True)
		user_info.replace(['-1', -1, -1.0], np.nan, inplace=True)
		df_order[['score_level', 'o_sku_num']].fillna(1)

		df_action.fillna(0)
		df_order.fillna(0)

		df_order = df_order.sort_values('o_date')
		df_action = df_action.sort_values('a_date')

		cal_start = pd.to_datetime(cal_start) if cal_start else df_order['o_date'].min().replace(day=1)
		cal_stop = pd.to_datetime(cal_stop) if cal_stop else df_order['o_date'].max()
		max_month = cal_stop.month + 1
		cal_stop = cal_stop.replace(month=max_month, day=1)

		max_gap = (cal_stop - cal_start).days
		months = list(rrule(MONTHLY, dtstart=cal_start, until=cal_stop))
		month_list = list(zip(months[-2::-1], months[-1::-1]))
		logging.info("Start to create features !")

		test_start, test_stop = month_list.pop(0)

		train = list()
		for start, end in month_list:
			df_label = user_info.copy()
			p_s, p_e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
			logging.info("Start to Create Train Features between {0} - {1} !".format(p_s, p_e))

			label_month = end.month
			label_year = end.year

			df_label['tag'] = end.strftime("%y%m")
			order_label = df_order[(df_order['o_month'] == label_month) & (df_order['o_year'] == label_year) &
			                       ((df_order['cate'] == 30) | (df_order['cate'] == 101))]
			label = order_label.sort_values("o_date").drop_duplicates(["user_id"], keep="first")
			df_label = df_label.merge(label[['user_id', 'o_date']], on='user_id', how='left')
			tmp_fe = order_label.groupby('user_id')['o_id'].nunique().reset_index().rename(columns={'o_id': 's1'})
			df_label = df_label.merge(tmp_fe, on='user_id', how='left')
			df_label.loc[:, 's1'] = df_label['s1'].apply(lambda s: (3 if s > 3 else s))
			df_label.loc[:, 's2'] = df_label['o_date'].apply(lambda s: (s.day if (s and s >= start) else 0))
			del df_label['o_date']
			# 下月用户购买数量作为本月特征的s1，下月首次购买日作为s2

			order = df_order[df_order['o_date'] < end]
			action = df_action[df_action['a_date'] < end]
			order.loc[:, 'day_gap'] = order['o_date'].apply(lambda d: (end - d).days)
			action.loc[:, 'day_gap'] = action['a_date'].apply(lambda d: (end - d).days)

			train_tmp = self.create_feats(df_label, order, action, max_gap=max_gap)
			train.append(train_tmp)
			p_n = train_tmp.shape
			logging.info("Train Features in {0} - {1} generated, total {2} features!".format(p_s, p_e, p_n))

		train = pd.concat(train)
		train.to_csv(train_file, index=None, encoding='utf-8')

		df_label = user_info.copy()
		p_s, p_e = test_start.strftime("%Y-%m-%d"), test_stop.strftime("%Y-%m-%d")
		logging.info("Start to Create Test Features between {0} - {1} !".format(p_s, p_e))

		df_label['tag'] = test_stop.strftime("%y%m")
		df_label.loc[:, 's1'] = -1
		df_label.loc[:, 's2'] = -1

		order = df_order[df_order['o_date'] < test_stop]
		action = df_action[df_action['a_date'] < test_stop]
		order.loc[:, 'day_gap'] = order['o_date'].apply(lambda d: (test_stop - d).days)
		action.loc[:, 'day_gap'] = action['a_date'].apply(lambda d: (test_stop - d).days)

		test = self.create_feats(df_label, order, action, max_gap=max_gap)
		logging.info("Test Features in {0} - {1} generated!".format(p_s, p_e))
		test.to_csv(test_file, index=None, encoding='utf-8')
		return train, test

	@staticmethod
	def create_feats(df_label, order, action, max_gap=270):
		"""
		特征构造函数
		:param df_label: 客户信息
		:param order: 客户订单信息
		:param action: 客户行为信息
		:param max_gap: 统计最大窗口
		"""
		order_in_30_or_101 = order[order['cate'].isin([101, 30])]
		order_in_30 = order_in_30_or_101[order_in_30_or_101['cate'] == 30]
		order_in_101 = order_in_30_or_101[order_in_30_or_101['cate'] == 101]
		order_not_in_30_and_101 = order[(order['cate'] != 30) & (order['cate'] != 101)]

		action_in_30_or_101 = action[action['cate'].isin([101, 30])]
		action_in_30 = action_in_30_or_101[action_in_30_or_101['cate'] == 30]
		action_in_101 = action_in_30_or_101[action_in_30_or_101['cate'] == 101]

		# 价格特征
		tmp_fe = order_in_30_or_101.groupby('user_id')['price'].agg(
			{'price_sum': 'sum', 'price_mean': 'mean', 'price_min': 'min',
			 'price_max': 'max', 'price_median': 'median', 'price_std': 'std'}
		).reset_index()
		df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

		# 评价特征
		tmp_fe = order_in_30_or_101.groupby('user_id')['score_level'].mean().reset_index()\
			.rename(columns={'score_level': 'score_level_mean'})
		df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

		for sl in (1, 2, 3):
			tmp_fe = order_in_30_or_101[order_in_30_or_101['score_level'] == sl]
			tmp_fe = tmp_fe.groupby('user_id')['score_level'].sum().reset_index()\
				.rename(columns={'score_level': 'score_{0}_sum'.format(sl)})
			df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

		tmp_fe = order_in_30_or_101[order_in_30_or_101['c_month'] > 0]
		tmp_fe = tmp_fe.groupby('user_id').size().reset_index().rename(columns={0: 'comment_cnt'})
		df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

		for od in range(7):
			tmp_fe = order_in_30_or_101[order_in_30_or_101['o_week_day'] == od]
			tmp_fe = tmp_fe.groupby('user_id')['o_id'].count().reset_index() \
				.rename(columns={'o_id': 'user_week_{0}_buy'.format(od)})
			df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

		# 商城订单数量特征
		tmp_fe = order_in_30_or_101.groupby('user_id')['o_id'].nunique().reset_index()\
				.rename(columns={'o_id': 'o_id_nunique'})
		df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

		# 商品相关特征
		tmp_fe = order_in_30_or_101.groupby('user_id')['sku_id'].agg(
			{'sku_id_nunique': 'nunique', 'sku_id_count': 'count'}).reset_index()
		df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

		tmp_fe = order_in_30_or_101.groupby('user_id')['o_sku_num'].sum().reset_index() \
			.rename(columns={'o_sku_num': 'o_sku_num_sum'})
		df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

		tmp_fe = action_in_30_or_101.groupby('user_id')['sku_id'].count().reset_index() \
			.rename(columns={'sku_id': 'a_sku_id_count'})
		df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

		# 日期相关特征
		tmp_fe = order_in_30_or_101.groupby('user_id')['o_day'].mean().reset_index() \
			.rename(columns={'o_day': 'o_day_mean'})
		df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)
		for fe in ('o_date', 'o_month'):
			tmp_fe = order_in_30_or_101.groupby('user_id')[fe].nunique().reset_index() \
				.rename(columns={fe: '{0}_nunique'.format(fe)})
			df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)
		tmp_fe = action_in_30_or_101.groupby('user_id')['a_date'].nunique().reset_index() \
			.rename(columns={'a_date': 'a_date_nunique'})
		df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

		# 时间段内最后一次购买的相关特征
		order_last = order_in_30_or_101.sort_values(by='o_date', ascending=False).drop_duplicates('user_id')
		order_last = order_last[['user_id', 'para_1', 'para_2', 'price', 'o_sku_num', ]]
		order_last.columns = ['user_id', 'o_last_pa1', 'o_last_pa2', 'o_last_price', 'o_last_num']
		df_label = df_label.merge(order_last, how='left', on='user_id')

		# 时间段内第一次购买的相关特征
		order_first = order_in_30_or_101[['user_id', 'para_1']]
		order_first = order_first.groupby('user_id').nth(0).reset_index()
		order_first.columns = ['user_id', 'o_first_pa1']
		df_label = df_label.merge(order_first, how='left', on='user_id')

		# 分窗口统计特征，窗口为周、半月、月、季度、半年、270天
		gap_list = [i for i in [7, 14, 30, 90, 180, 270] if i <= int(max_gap)]
		for i in gap_list:
			order_tmp_1 = order_in_30_or_101[(order_in_30_or_101['day_gap'] <= i)]
			order_tmp_2_1 = order_in_30[(order_in_30['day_gap'] <= i)]
			order_tmp_2_2 = order_in_101[(order_in_101['day_gap'] <= i)]
			order_tmp_3 = order_not_in_30_and_101[(order_not_in_30_and_101['day_gap'] <= i)]

			action_tmp_1 = action_in_30_or_101[(action_in_30_or_101['day_gap'] <= i)]
			action_tmp_2_1 = action_in_30[(action_in_30['day_gap'] <= i)]
			action_tmp_2_2 = action_in_101[(action_in_101['day_gap'] <= i)]

			a = "action_win{0}_".format(i)
			o = "order_win{0}_".format(i)

			# 窗口内第一天购买
			df_label.loc[:, o + 'o_date_target_0day'] = order_tmp_1.drop_duplicates(['user_id'], keep='first')['o_day']

			# 窗口内的价格特征
			tmp_fe = order_tmp_1.groupby('user_id')['price'].agg(
				{o + 'price_sum': 'sum', o + 'price_mean': 'mean', o + 'price_min': 'min',
				 o + 'price_max': 'max', o + 'price_median': 'median', o + 'price_std': 'std'}
			).reset_index()
			df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

			for k, obj in enumerate([order_tmp_1, order_tmp_2_1, order_tmp_2_2, order_tmp_3]):
				# 窗口内的商城订单数量特征
				tmp_fe = obj.groupby('user_id')['o_id'].nunique().reset_index() \
					.rename(columns={'o_id': '{0}o_id_{1}_nunique'.format(o, k)})
				df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

				# 窗口内的商品特征
				tmp_fe = obj.groupby('user_id')['sku_id'].count().reset_index() \
					.rename(columns={'sku_id': '{0}sku_id_{1}_count'.format(o, k)})
				df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

				# 窗口内的商品数量特征
				tmp_fe = obj.groupby('user_id')['o_sku_num'].sum().reset_index() \
					.rename(columns={'o_sku_num': '{0}o_sku_num_{1}_sum'.format(o, k)})
				df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

				# 窗口内购买商品的参数特征
				for pa in ('para_1', 'para_2'):
					tmp_fe = obj.groupby('user_id')[pa].agg(
						{'{0}_{1}_{2}_count'.format(o, pa, k): 'sum'}
					).reset_index()
					df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

			for k, obj in enumerate([order_tmp_1, order_tmp_2_1, order_tmp_2_2]):
				# 窗口内购买商品的参数特征
				for pa in ('para_1', 'para_2'):
					tmp_fe = obj.groupby('user_id')[pa].agg(
						{'{0}_{1}_{2}_std'.format(o, pa, k): 'std',
						 '{0}_{1}_{2}_mean'.format(o, pa, k): 'mean'}
					).reset_index()
					df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)
				# 窗口内购买日期特征
				tmp_fe = obj.groupby('user_id')['o_day'].agg(
					{'{0}_{1}_day_max'.format(o, k): 'max',
					 '{0}_{1}_day_mean'.format(o, k): 'mean',
					 '{0}_{1}_day_min'.format(o, k): 'min',
					 '{0}_{1}_day_std'.format(o, k): 'std'}
				).reset_index()
				df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

				tmp_fe = obj.groupby('user_id')['o_date'].nunique().reset_index() \
					.rename(columns={'o_date': '{0}o_date_{1}_nunique'.format(o, k)})
				df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

				if i > 30:
					tmp_fe = obj.groupby('user_id')['o_month'].nunique().reset_index() \
						.rename(columns={'o_month': '{0}o_month_{1}_nunique'.format(o, k)})
					df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

			# 窗口内商品行为特征
			for k, obj in enumerate([action_tmp_1, action_tmp_2_1, action_tmp_2_2]):
				tmp_fe = obj.groupby('user_id')['sku_id'].agg(
					{'{0}_{1}_sku_id_count'.format(a, k): 'count',
					 '{0}_{1}_sku_id_nunique'.format(a, k): 'nunique'}
				).reset_index()
				df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

				for action in (1, 2):
					tmp_action_obj = obj[obj['a_type'] == action]

					tmp_fe = tmp_action_obj.groupby('user_id')['sku_id'].count().reset_index() \
						.rename(columns={'sku_id': '{0}sku_id_{1}_type{2}_count'.format(a, k, action)})
					df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

				tmp_fe = obj.groupby('user_id')['a_date'].nunique().reset_index() \
					.rename(columns={'a_date': '{0}a_date_{1}_nunique'.format(a, k)})
				df_label = df_label.merge(tmp_fe, 'left', 'user_id').fillna(0)

		return df_label


class LGBMPredict(FeatureSelection):
	def __init__(self, feat_rebuild=False, model_rerun=False):
		"""
		预测模型
		:param feat_rebuild: 是否重新执行特征处理
		:param model_rerun: 是否重新生成模型文件
		"""
		self.rerun = model_rerun
		self.stacking_num = 10
		self.bagging_num = 10
		super(LGBMPredict, self).__init__(rebuild=feat_rebuild)
		self.result = self.output('submission.jw.csv')

	def predict_o_num(self, train, test, params=None):
		_param = {
			'task': 'train', 'boosting_type': 'gbdt', 'objective': 'regression', 'metric': {'l2', 'auc'},
			'num_leaves': 64, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8,
			'bagging_freq': 5, 'verbose': 0, 'max_depth': -1
		}
		if params:
			_param.update(**params)
		logging.info('Start to predict order num by lgbm !')
		Pred = test.drop(['user_id', 's1', 's2', 'tag'], axis=1)
		result = None
		if not self.rerun:
			result = self.predict(Pred, label='s1')
		if result is None:
			X = train.drop(['user_id', 's1', 's2', 'tag'], axis=1)
			y = train['s1']
			result, feat_imp = self.fit_predict(X, y, Pred, _param, label='s1')
			logging.info('------- S1 features importance -------- !\n{0}'.format(feat_imp.to_string()))
		return result

	def predict_o_date(self, train, test, params=None):
		_param = {
			'task': 'train', 'boosting_type': 'gbdt', 'objective': 'regression', 'metric': {'l2', 'auc'},
			'num_leaves': 32, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8,
			'bagging_freq': 5, 'verbose': 0, 'max_depth': -1
		}
		if params:
			_param.update(**params)
		logging.info('Start to predict the first order date by lgbm !')

		Pred = test.drop(['user_id', 's1', 's2', 'tag'], axis=1)
		result = None
		if not self.rerun:
			result = self.predict(Pred, label='s2')
		if result is None:
			X = train.drop(['user_id', 's1', 's2', 'tag'], axis=1)
			y = train['s2']
			result, feat_imp = self.fit_predict(X, y, Pred, _param, label='s2')
			logging.info('------- S2 features importance -------- !\n{0}'.format(feat_imp.to_string()))
		return result

	def predict(self, X_pred, label):
		test_pred = np.zeros((X_pred.shape[0], self.stacking_num))
		for sn in range(self.stacking_num):
			model = self.model('jw.stacking.model.{0}.{1}.pkl'.format(sn, label))
			if not os.path.isfile(model):
				return None
			gbm = joblib.load(model)
			pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
			test_pred[:, sn] = pred
			X_pred = np.hstack((X_pred, test_pred.mean(axis=1).reshape((-1, 1))))

		pred_out = 0
		for bn in range(self.bagging_num):
			model = self.model('jw.bagging.model.{0}.{1}.pkl'.format(bn, label))
			if not os.path.isfile(model):
				return
			gbm = joblib.load(model)
			pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
			pred_out += pred
		return pred_out / self.bagging_num

	def fit_predict(self, X, y, X_pred, param, label):
		"""
		:param label: 预测对象
		:param X: 训练集
		:param y: 训练集结果
		:param X_pred: 预测集
		:param param: 模型参数
		"""
		predictors = [i for i in X.columns] + ['lgbm_result']
		X = X.values
		y = y.values
		layer_train = np.zeros((X.shape[0], 2))

		bagging_test_size = 1.0 / self.bagging_num
		num_boost_round = 2000
		early_stopping_rounds = 200

		stacking_model = list()
		bagging_model = list()
		l2_error = list()
		feat_imp = list()
		pred_out = 0

		SK = StratifiedKFold(n_splits=self.stacking_num, shuffle=True, random_state=1)
		for (train_index, test_index) in SK.split(X, y):
			lgb_train = lgb.Dataset(X[train_index], y[train_index])
			lgb_eval = lgb.Dataset(X[test_index], y[test_index])
			gbm = lgb.train(param, lgb_train, num_boost_round=num_boost_round,
			                valid_sets=lgb_eval, early_stopping_rounds=early_stopping_rounds)

			stacking_model.append(gbm)
		X = np.hstack((X, layer_train[:, 1].reshape((-1, 1))))

		for bn in range(self.bagging_num):
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=bagging_test_size, random_state=bn)

			lgb_train = lgb.Dataset(X_train, y_train)
			lgb_eval = lgb.Dataset(X_test, y_test)

			gbm = lgb.train(param, lgb_train, num_boost_round=10000,
			                valid_sets=lgb_eval, early_stopping_rounds=early_stopping_rounds)

			bagging_model.append(gbm)
			gbm_result = gbm.predict(X_test, num_iteration=gbm.best_iteration)
			l2_error.append(mean_squared_error(gbm_result, y_test))
			feat_imp.append(list(gbm.feature_importance()))

		feat_imp = np.array(feat_imp).mean(axis=0)
		feat_imp = pd.Series(feat_imp, predictors).sort_values(ascending=False)
		l2_s = sum(l2_error) / float(len(l2_error))
		logging.info('l2 : {0} !!'.format(l2_s))

		test_pred = np.zeros((X_pred.shape[0], self.stacking_num))
		for sn, gbm in enumerate(stacking_model):
			joblib.dump(gbm, self.model('jw.stacking.model.{0}.{1}.pkl'.format(sn, label)))
			pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
			test_pred[:, sn] = pred
			X_pred = np.hstack((X_pred, test_pred.mean(axis=1).reshape((-1, 1))))

		for bn, gbm in enumerate(bagging_model):
			joblib.dump(gbm, self.model('jw.bagging.model.{0}.{1}.pkl'.format(bn, label)))
			pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
			pred_out += pred
		return pred_out / self.bagging_num, feat_imp

	def run(self, run_model='both', test_month=None, s1_param=None, s2_param=None):
		"""
		模型执行函数
		:param run_model: 预测目标[s1, s2, both, test], s1表示只预测目标月购买数量，s2表示只预测购买日期，
						  both表示两者均预测，test表示线下预测， str
		:param test_month: 线下测试月，仅在run_model为test时可用， int
		:param s1_param: 根据CV得到的S1的超参，dict
		:param s2_param: 根据CV得到的S2的超参， dict
		"""
		train_set, predict = self.feature_encode()

		if run_model == 'test':
			if str(test_month).isdigit():
				logging.info('Offline Test : {0} !'.format(test_month))
				test_month = int(test_month)
				predict = train_set[train_set['tag'] == test_month]
				train_set = train_set[train_set['tag'] < test_month]
			else:
				test_shape = predict.shape[0]
				predict = train_set.iloc[:test_shape, :]
				train_set = train_set.iloc[test_shape:, :]
		result = predict[['user_id']]
		if run_model != 's2':
			result.loc[:, 'o_num'] = self.predict_o_num(train_set, predict, s1_param)
			gc.collect()
		if run_model != 's1':
			result.loc[:, 'pred_date'] = self.predict_o_date(train_set, predict, s2_param)
			gc.collect()
		if run_model == 'both':
			result = result.sort_values(by='o_num', ascending=False).reset_index(drop='index')
			result.loc[:, 'pred_date'] = result['pred_date'].apply(
				lambda day: datetime(2017, 9, 1) + timedelta(days=int(day + 0.49 - 1)))
			# result.loc[:49999, ['user_id', 'pred_date']].to_csv(self.result, index=None, encoding='utf-8')
			result.to_csv(self.result, index=None, encoding='utf-8')
			logging.info('All jobs done, result save to {0} !'.format(self.result))
			gc.collect()
		else:
			logging.info('All jobs done, result save to {0}.{1}.csv !'.format(self.result, run_model))
			result.to_csv('{0}.{1}.csv'.format(self.result, run_model), index=None, encoding='utf-8')
		if run_model == 'test':
			s1, s2 = score(result)
			logging.info('Test result: s1 score is %s,s2 score is %s, S is %s' % (s1, s2, 0.4 * s1 + 0.6 * s2))
			gc.collect()


if __name__ == "__main__":
	clf = LGBMPredict()
	clf.run()
