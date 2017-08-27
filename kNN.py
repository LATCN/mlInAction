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
        labels: 数据对应的标签（分类）
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inx, dataset, labels, k):
    """分类器

    Args:
        inx: 输入数据
        dataset: 训练样本集
        labels: 标签（分类）
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


def file2matrix(filename):
    """准备数据 将文本记录转换为array

    Args:
        filename: 文件名
    Returns:
        返回处理后的（样本）数据/标签（分类）
    """
    fr = open(filename)
    array_lines = fr.readlines()
    number_lines = len(array_lines)
    return_mat = zeros((number_lines, 3))  # 创建 n*3 数组
    classlabel_vector = []
    index = 0
    for line in array_lines:
        line = line.strip()  # 去除左右空格
        list_fromline = line.split('\t')
        return_mat[index, :] = list_fromline[0:3]  # 获取样本数据的各个特征值
        classlabel_vector.append(int(list_fromline[-1]))  # 获取样本对应的分类
        index += 1
    return return_mat, classlabel_vector


def autonorm(dataset):
    """归一化特征值 newvalue = (oldvalue-min)/(max-min)

    Args:
        dataset: 数据集
    Returns:
        返回归一化处理后的数据
    """
    minvals = dataset.min()
    maxvals = dataset.max()
    ranges = maxvals - minvals
    # norm_dataset = zeros(shape(dataset))
    m = dataset.shape[0]  # 返回行数
    norm_dataset = dataset - tile(minvals, (m, 1))  # minvals 为一行， 重复m行，1列
    norm_dataset = norm_dataset/tile(ranges, (m, 1))  # ranges 列数据间距（为一行），重复m行，1列，各元素相除，归一化
    return norm_dataset, ranges, minvals


def datingclass_test():
    """dating 分类器测试
    """
    ho_ratio = 0.10
    dating_datamat, dating_labels = file2matrix('datingTestSet2.txt')
    normmat, ranges, minvals = autonorm(dating_datamat)
    m = normmat.shape[0]  # 总数据量 90%的训练数据
    num_testvecs = int(m*ho_ratio)  # 10%测试数据量
    error_count = 0.0
    for i in range(num_testvecs):
        classifier_result = classify0(normmat[i, :], normmat[num_testvecs:m, :],
                                      dating_labels[num_testvecs:m], 3)
        print "he classifier came back with: %d, the real answer is: %d" \
            % (classifier_result, dating_labels[i])

        if classifier_result != dating_labels[i]:
            error_count += 1.0

    print "the total error rate is %f" %(error_count/float(num_testvecs))








