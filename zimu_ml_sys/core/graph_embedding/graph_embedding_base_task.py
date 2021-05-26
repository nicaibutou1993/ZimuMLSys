# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import pandas as pd

from zimu_ml_sys.constant import PREPROCESS_DATA_PATH, FEATURE_DATA_PATH
from zimu_ml_sys.core.graph_embedding.deep_walk import DeepWalk
from zimu_ml_sys.core.graph_embedding.graph import build_graph
from zimu_ml_sys.core.graph_embedding.node2vector import Node2Vec
from zimu_ml_sys.utils.data_util import df_group_parallel
from gensim.models import KeyedVectors


class GraphEmbeddingBaseTask():
    """
    根据用户历史点击行为，将 item 向量化处理

    1. node2vector: 有向图
        边与边的权重 即为：item 与 item 权重：
                使用用户点击 序列，两两相邻点击的item 作为边与边的关系。
                    权重： 取决于 相邻两个item：
                       1. 点击时间差： 两个item 点击时间越相近，表示两个item 关系越重要，权重值 越大
                       2. item1的热度 与item2 的热度： 这个主要是：
                            1. 如果你想要优先 热门item，那么冷门item 到热门item 的权重 你就设置 大一点
                            2. 如果你想要近可能 降低热门item，那么热门item 到冷门item 的权重 设置大一点

                        max(3, np.log(1, item1_click / item2_click))  ## 降低热门，使得在node2vector 随机游走的时候 能尽量不要总是选择热门item
                        / (1 + 时间差）     ## 时间差

    2. deepwalker：无向图
        仅仅根据 用户的点击序列：，item与item 连接边的权重都是1，
                        如果item1 与item2 共线次数多，item1 与item3 仅出现一次，他们在被item1 选择的概率是一样的。
    """

    """原始数据"""
    preprocess_root_data_path = ''
    """item embedding 存放根路径"""
    feature_root_data_path = ''

    """句子的长度"""
    walk_length = 20

    """循环步数   总的样本数： 80 * len(items)"""
    num_walks = 80

    """当前节点 返回上一次节点 惩罚项，值 越大，表示当前节点越不会返回上一个节点
        w * 1 / p
    """
    p = 2

    """当前节点 选择邻居节点， 且 该邻居节点 与 上一个节点 不是邻居，值越小，表示 当前节点 会去比较深的地方去
        w * 1 / q
    """
    q = 0.5

    """item向量维度"""
    embed_size = 128

    """窗口大小"""
    window_size = 10

    """word2vector 训练 并发数"""
    workers = 6

    """word2vector 模型 迭代次数"""
    epochs = 3

    user_field_name = 'user_id'

    item_field_name = 'item_id'

    time_field_name = 'time'

    train_df = None

    def read_data(self):
        """读取数据集"""

        # train_df = pd.read_csv(os.path.join(self.preprocess_root_data_path, 'train.csv'))
        # self.train_df = train_df.sort_values('time')

        raise NotImplementedError

    def train_node2vector(self):
        """
        构建有向图
        训练node2vector 模型
        边与边的权重 即为：item 与 item 权重：
            使用用户点击 序列，两两相邻点击的item 作为边与边的关系。
                权重： 取决于 相邻两个item：
                   1. 点击时间差： 两个item 点击时间越相近，表示两个item 关系越重要，权重值 越大
                   2. item1的热度 与item2 的热度： 这个主要是：
                        1. 如果你想要优先 热门item，那么冷门item 到热门item 的权重 你就设置 大一点
                        2. 如果你想要近可能 降低热门item，那么热门item 到冷门item 的权重 设置大一点

                    max(3, np.log(1, item1_click / item2_click))  ## 降低热门，使得在node2vector 随机游走的时候 能尽量不要总是选择热门item
                    / (1 + 时间差）     ## 时间差
        :return:
        """

        if self.train_df is None:
            self.train_df = self.read_data()
            self.train_df = self.train_df.sort_values(self.time_field_name)

        edges = self.build_edges()

        graph = build_graph(edges)

        print('load node2vector')
        node_model = Node2Vec(graph, walk_length=self.walk_length, num_walks=self.num_walks, p=self.p, q=self.q,
                              workers=1)

        print('train node2vector')
        node_model.train(embed_size=self.embed_size, window_size=self.window_size, workers=self.workers,
                         epochs=self.epochs)

        node_model.w2v_model.wv.save_word2vec_format(
            os.path.join(self.feature_root_data_path, "node2vec_embedding.bin"), binary=True)

    def train_deepwalker(self):
        """
        构建无向图
        训练deepwalker

        仅仅根据 用户的点击序列：，item与item 连接边的权重都是1，
                        如果item1 与item2 共线次数多，item1 与item3 仅出现一次，他们在被item1 选择的概率是一样的。

        :return:
        """
        direction = False

        if self.train_df is None:
            self.train_df = self.read_data()
            self.train_df = self.train_df.sort_values(self.time_field_name)

        edges = self.build_edges(direction=direction)
        graph = build_graph(edges, direction=direction)
        deep_model = DeepWalk(graph, walk_length=self.walk_length, num_walks=self.num_walks, workers=1)
        deep_model.train(embed_size=self.embed_size, window_size=self.window_size, workers=self.workers,
                         epochs=self.epochs)
        deep_model.w2v_model.wv.save_word2vec_format(
            os.path.join(self.feature_root_data_path, "deepwalk_embedding.bin"), binary=True)

    def run(self):
        """
        执行入口，训练node2vector 和 deepwalker 两个模型
        :return:
        """
        time_time = time.time()

        self.train_df = self.read_data()
        self.train_df = self.train_df.sort_values(self.time_field_name)

        self.train_node2vector()
        print(time.time() - time_time)
        self.train_deepwalker()
        print(time.time() - time_time)

    def build_edges(self, direction=True):
        """
        构建边与边
        :param direction: 是否有向图
        :return: 边的集合
        """
        item_value_counts = dict(self.train_df[self.item_field_name].value_counts())

        edgelist = []
        for _, user_group_data in self.train_df.groupby(self.user_field_name):

            items = user_group_data[self.item_field_name].values
            times = user_group_data[self.time_field_name].values
            for i in range(len(items) - 1):
                delta = abs(times[i] - items[i + 1]) / (60 * 60 * 12)

                ai, aj = item_value_counts[items[i]], item_value_counts[items[i + 1]]
                """热门与非热门：之比 在 20倍以上，体现当前3"""
                """1. delta_t: 时间差 时间差越大，表示两个节点 相关性越低
                   2. np.log(1 + ai / aj) 前者相比后者热度越高，则权重越大，使得node2vector 加大对冷门item 进行采样
                   3. 0.8：表示逆序情况下，需要降低 两item 之间的权重值
                   
                   有向有权图，热门商品-->冷门商品权重=热门商品个数/冷门商品个数
                """
                if direction:
                    edgelist.append([items[i], items[i + 1], max(3, np.log(1 + ai / aj)) * 1 / (1 + delta)])
                    edgelist.append([items[i + 1], items[i], max(3, np.log(1 + aj / ai)) * 0.8 * 1 / (1 + delta)])
                else:
                    edgelist.append([items[i], items[i + 1], 1.0])
        print('load edges success')
        return edgelist

    def load_node2vector_model(self):

        node_model = KeyedVectors.load_word2vec_format(
            os.path.join(self.feature_root_data_path, "node2vec_embedding.bin"), binary=True)

        return node_model

    def load_deepwalker_model(self):

        deep_model = KeyedVectors.load_word2vec_format(
            os.path.join(self.feature_root_data_path, "deepwalk_embedding.bin"), binary=True)

        return deep_model

    def explore_time_importance(self):
        """
            探索 用户相邻点击的item,时间相关性，一般来说 上一次点击 如果与 当前点击的 商品 点击距离越近 则与之 关系越近

            相邻item： 时间关系权重，时间越大，相邻item 权重越小：
                            delta  = abs( time(item1) - time(item2) ) / （60 * 60 * 12）  ## 60 * 60 * 12 时间 约束小一点
                            1 / ( 1 + delta )
            min           0.000000
            1%            4.955338
            5%           10.901743
            10%          19.821351
            25%          62.437254
            50%         291.373855
            75%        6101.011731
            90%       46305.648330
            95%       77996.023756
            99%      152713.001474
            max      327660.801257
        :param train_df:
        :return:
        """
        self.train_df = self.read_data()

        train_df = self.train_df.sort_values(self.time_field_name)

        def compute_interval(name, group_df):
            group_df['time_shift'] = group_df[self.time_field_name].shift(-1)

            group_df['time_interval'] = group_df['time_shift'] - group_df[self.time_field_name]

            return group_df

        train_df = df_group_parallel(train_df.groupby(self.user_field_name), compute_interval, n_jobs=1)

        train_df = train_df[train_df['time_interval'].notnull()]

        print(train_df['time_interval'].describe([.01, .05, .10, .25, .5, .75, .90, .95, .99]))


if __name__ == '__main__':
    GraphEmbeddingBaseTask().run()
