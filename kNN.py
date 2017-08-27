#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: LAT_CN
@file: kNN.py
@site: https://github.com/latcn/mlInAction
@time: 2017/8/27 12:00
@desc: k-近邻算法
概述：
1）已知一个训练样本集（知道样本每个数据与所属分类/标签的对应关系）
输入新数据时，将新数据的所有特征组成一个向量，与样本集每条数据（向量）
进行差向量计算（欧氏距离）
2）选取与当前数据距离最近的k个样本数据
3）前k个样本数据所在类别的出现频率，返回频率最高者作为当前数据的预测分类
"""

from numpy import *
import operator


def create_dataset():
    """训练样本集 包含多条样本数据，每条数据对应的标签

    Args:
    Returns:
        group: 训练样本集数据
        labels: 数据对应的标签
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inx, dataset, labels, k):
    """分类器

    Args:
        inx: 输入数据
        dataset: 训练样本集
        labels: 标签
        k: 前k个
    Returns:
        输入数据属于哪个分类
    """
    dataset_size = dataset.shape[0]  # 样本训练集数据num
    diffmat = tile(inx, (dataset_size, 1)) - dataset  # tile:将inx行方向重复*次,列方向重复1次,计算与每个样本数据的差向量
    sq_diffmat = diffmat ** 2  # 每个数平方
    sq_distances = sq_diffmat.sum(axis=1)  # axis:0 同列的各行之和 1:同行的各列之和
    distance = sq_distances ** 0.5  # 每个数的平方根
    sorted_distindicies = distance.argsort()  # 返回从小到大的index
    classcount = {}
    for i in range(k):  # range(k): 生成从0到k-1的列表
        vote_ilabel = labels[sorted_distindicies[i]]  # 获取对应的分类标签
        classcount[vote_ilabel] = classcount.get(vote_ilabel, 0) + 1  # 字典get方法，存在返回对应的值，不存在返回0
    sorted_classcount = sorted(classcount.iteritems(),
                               key=operator.itemgetter(1), reverse=True)  # 按字典值降序排列

    return sorted_classcount[0][0]  # 返回符合的分类

