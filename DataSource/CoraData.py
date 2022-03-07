#!/usr/bin/env python
# coding: utf-8

import itertools
import os
import os.path as osp
import pickle
from collections import namedtuple

import numpy as np
import scipy.sparse as sp

Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


class CoraData(object):
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root=None, rebuild=False):
        """Cora数据，包括数据下载，处理，加载等功能
        当数据的缓存文件存在时，将使用缓存文件，否则将下载、进行处理，并缓存到磁盘

        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
            * y: 节点的标签，总共包括7个类别，类型为 np.ndarray
            * adjacency: 邻接矩阵，维度为 2708 * 2708，类型为 scipy.sparse.coo.coo_matrix
            * train_mask: 训练集掩码向量，维度为 2708，当节点属于训练集时，相应位置为True，否则False
            * val_mask: 验证集掩码向量，维度为 2708，当节点属于验证集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 2708，当节点属于测试集时，相应位置为True，否则False

        Args:
        -------
            data_root: string, optional
                存放数据的目录，原始数据路径: ../data/cora
                缓存数据路径: {data_root}/ch5_cached.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据

        """
        self.data_root = data_root if data_root else osp.join(osp.dirname(__file__), 'data/cora')

        # 数据集文件夹不存在则创建
        os.makedirs(self.data_root, exist_ok=True)

        save_file = osp.join(self.data_root, "ch5_cached.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)  # （待确认）pickle不能在类内存储该类的成员变量，可能导致循环引用的问题，所以要把Data定义到类外
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        引用自：https://github.com/rusty1s/pytorch_geometric
        """
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(
            osp.join(self.data_root, name)) for name in self.filenames]

        # y.shape = (140, 7)
        train_index = np.arange(y.shape[0])
        # train_index = [0,1,2,...,139]

        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        # val_index = [140,141,...,639]

        sorted_test_index = sorted(test_index)

        """
        numpy提供了numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接。其中a1,a2,...是数组类型的参数
        a=np.array([[1,2,3],[4,5,6]])
        b=np.array([[11,21,31],[7,8,9]])
        np.concatenate((a,b),axis=0)
        array([[ 1,  2,  3],
               [ 4,  5,  6],
               [11, 21, 31],
               [ 7,  8,  9]])

        np.concatenate((a,b),axis=1)  #axis=1表示对应行的数组进行拼接
        array([[ 1,  2,  3, 11, 21, 31],
               [ 4,  5,  6,  7,  8,  9]])
        """
        x = np.concatenate((allx, tx), axis=0)
        """
        argmax取出每个数组的最大值
        """
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        """
        a = np.array([[1, 2, 3], [4, 5, 6]])
        print(a[[0, 1]])  # [[1 2 3], [4 5 6]]
        a[[1, 0]] = a[[0, 1]]
        print(a[[1, 0]])  # [[1 2 3], [4 5 6]]
        print(a)  # [[4 5 6], [1 2 3]]
        下面将x数组的元素顺序调了下
        """
        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency=adjacency,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        """根据邻接表创建邻接矩阵"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # 去除重复的边
        """
        sorted(edge_index)将edge_index中相同的元素排在一起，这时itertools.groupby的k就是每一段相同元素的值
        print([k for k, g in itertools.groupby('AAAABBBCCDAABBB')])
        # ['A', 'B', 'C', 'D', 'A', 'B']
        print([list(g) for k, g in itertools.groupby('AAAABBBCCDAABBB')])
        # [['A', 'A', 'A', 'A'], ['B', 'B', 'B'], ['C', 'C'], ['D'], ['A', 'A'], ['B', 'B', 'B']]
        """
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))

        # 将输入数据（列表的列表，元组的元组，元组的列表等）转换为矩阵形式
        edge_index = np.asarray(edge_index)
        """
        row  = np.array([0, 3, 1, 0])
        col  = np.array([0, 3, 1, 2])
        data = np.array([4, 5, 7, 9])
        coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
        array([[4, 0, 9, 0],
               [0, 7, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 5]])
        """
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

    @staticmethod
    def normalization(adjacency):
        """计算 L=D^-0.5 * (A+I) * D^-0.5"""

        # eye 对角线为1的稀疏矩阵
        adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
        degree = np.array(adjacency.sum(1))

        # flatten 返回一个折叠成一维的数组
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat).tocoo()  # 返回稀疏矩阵的coo_matrix形式
