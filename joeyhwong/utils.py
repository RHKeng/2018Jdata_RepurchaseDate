#! /usr/bin/env python
# -*- coding: utf-8 -*-
# 代码有问题别找我！虽然是我写的，并且我觉得它是没问题的，如果不是你的操作原因，或许是它自己长歪了！

"""
一些配置信息
"""
import os
import numpy as np

RUN_PATH = os.path.abspath(os.path.dirname(__file__))
RAW_PATH = os.path.join(RUN_PATH, '../../data')  # 原始数据目录，位于与当前文件上级目录同级的data
OUT_PATH = os.path.join(RUN_PATH, '../../output')  # 结果目录，位于与当前文件上级目录同级的output
MODEL_PATH = os.path.join(RUN_PATH, '../../model')  # 模型存放目录，位于与当前文件上级目录同级的model
TEMP_PATH = os.path.join(RUN_PATH, '../../temp')  # 中间文件目录，位于与当前文件上级目录同级的temp


def score(df):
	sub = df.sort_values('o_num', ascending=False).reset_index(drop='index')
	sub['s1'] = [1 if x > 0 else 0 for x in sub['label_1']]
	sub = sub.loc[:49999, :]
	sub['weight'] = [1 / (1 + np.log(i)) for i in range(1, 50001)]

	s1 = sum(sub['s1'] * sub['weight']) / 4674.323

	s2 = 0.0
	df = df[df['s1'] > 0].reset_index(drop='index')
	for i in range(sub.shape[0]):
		if sub.loc[i, 's1'] > 0:
			s2 += 10.0 / ((sub.loc[i, 's2'] - np.round(sub.loc[i, 'pred_date'])) ** 2 + 10)

	s2 = s2 / df.shape[0]

	return s1, s2
