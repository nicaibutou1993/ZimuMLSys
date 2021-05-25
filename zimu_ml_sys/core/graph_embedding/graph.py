# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np


def build_graph(edgelist, direction=True, new_wei=False):
    """

    构建图：

    1. 获取每一个用户 所有点击的 item
    2. 循环每一个用户下，每一个点击的item，
        图的2个节点为：上一次点击的item，及当前点击的item，权重：w
        权重表示：节点与节点 之间的 weight
            使用deepwalk:表示不需要考虑 item与item之间的权重关系
            自定义权重：可以考虑 因素有：
                    1. 时间差：当前item 与 上一轮item 点击时间差，如果时间相差比较小，说明它们之间关系 比较紧密
                    2. 是否考虑降低流行商品权重：比如推荐系统 更会倾向 推荐热门item，
                            原因是模型训练的数据，很多都是用户点击的热门商品，所以导致模型最终推荐还是热门商品。
                            这里训练 node2vector,可以手动设值。
                                如果是热门商品 到 冷门商品 应该加大权重，曝光率低的冷门商品拥有优异表现
                                如果是冷门商品 到 热门商品，应该降低权重，因为很多 冷门商品下一步很多会去热门商品，node2vector 生成训练集中会出现大量的热门商品
                            同时热门冷门是相对的，np.log(1 + ai / aj)   ai：表示上一轮点击item 点击次数，aj:当前item点击次数，
                            如果上一轮是热门，当前是冷门，那么计算结果会高，相反，值就会很低

                    3. 次序问题： 正常是先点击某个item，在点击下一个item，默认权重是 1
                                反过来，计算 当前item 与上 一个item 权重值，是需要降低权重的，比如设置 0.8

    :param df: 输入数据，要求根据时间已经从小到大排序好
    :param user_col:
    :param item_col:
    :param time_col: 时间 field
    :param direction: 是否有向图 还是 无向图   针对 deepwalk 是无向图 node2vector 是有向图
    :param new_wei:
    :return:
    """
    # user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    # user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))
    # edgelist = []
    #
    # if time_col:
    #     user_time_ = df.groupby(user_col)['time'].agg(list).reset_index()  # 引入时间因素
    #     user_time_dict = dict(zip(user_time_[user_col], user_time_['time']))
    #
    # item_cnt = df[item_col].value_counts().to_dict()
    #
    # for user, items in user_item_dict.items():
    #     for i in range(len(items) - 1):
    #         if direction:
    #             t1 = user_time_dict[user][i]  # 点击时间提取
    #             t2 = user_time_dict[user][i + 1]
    #             delta_t = abs(t1 - t2) * 50000  # 中值 0.01 75%:0.02
    #             #             有向有权图，热门商品-->冷门商品权重=热门商品个数/冷门商品个数
    #             ai, aj = item_cnt[items[i]], item_cnt[items[i + 1]]
    #             """ 1. delta_t: 时间差 时间差越大，表示两个节点 相关性越低
    #                 2. np.log(1 + ai / aj) 前者相比后者热度越高，则权重越大，使得node2vector 加大对冷门item 进行采样
    #                 3. 0.8：表示逆序情况下，需要降低 两item 之间的权重值
    #             """
    #             edgelist.append([items[i], items[i + 1], max(3, np.log(1 + ai / aj)) * 1 / (1 + delta_t)])
    #             edgelist.append([items[i + 1], items[i], max(3, np.log(1 + aj / ai)) * 0.8 * 1 / (1 + delta_t)])
    #         else:
    #             edgelist.append([items[i], items[i + 1], 1])
    if direction:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    """导入节点及边权重"""
    for edge in edgelist:
        G.add_edge(str(edge[0]), str(edge[1]), weight=edge[2])
    if new_wei:
        for u, v, d in G.edges(data=True):
            deg = G.degree(u) / G.degree(v)
            if deg < 1:
                deg = max(0.1, deg)
            else:
                deg = min(3, deg)
            new_weight = d["weight"] * deg
            G[u][v].update({"weight": new_weight})
    return G
