#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: LAT_CN
@file: aTest.py
@site: https://github.com/latcn/mlInAction
@time: 2017/8/27 13:04
@desc:
"""

import kNN
# import numpy as np
# import matplotlib.pyplot as plt

# x = np.arange(0, 5, 0.1);
# y = np.sin(x)
# plt.plot(x, y)
# plt.show()


# datamat, labels = kNN.file2matrix('datingTestSet.txt')
# print datamat, labels
# print labels
#
# datamat, ranges, minvals = kNN.autonorm(datamat)
#
# plt.scatter(datamat[:, 1], datamat[:, 2])
# print datamat
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # fig.add_axes(datamat[:1])
# # fig.add_axobserver()
# # fig.add_subplot(111, datamat[:1], datamat[:2])
# # # ax.scatter(datamat[:1], datamat[:2])
# plt.show()

kNN.datingclass_test()
