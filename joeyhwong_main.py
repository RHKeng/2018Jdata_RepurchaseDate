#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
主程序
"""
from __future__ import absolute_import
from joeyhwong.model import LGBMPredict


def main():
	"""
	主函数
	"""
	clf = LGBMPredict()
	clf.run(run_model='s1')


if __name__ == '__main__':
	main()
