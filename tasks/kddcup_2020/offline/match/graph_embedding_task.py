# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import pandas as pd

from tasks.kddcup_2020.constant import TASK_NAME
from zimu_ml_sys.core.graph_embedding.graph_embedding_base_task import GraphEmbeddingBaseTask
from zimu_ml_sys.constant import PREPROCESS_DATA_PATH, FEATURE_DATA_PATH


class GraphEmbeddingTask(GraphEmbeddingBaseTask):
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

    preprocess_root_data_path = os.path.join(PREPROCESS_DATA_PATH, TASK_NAME)
    feature_root_data_path = os.path.join(FEATURE_DATA_PATH, TASK_NAME)

    def read_data(self):
        """读取数据集"""

        train_df = pd.read_csv(os.path.join(self.preprocess_root_data_path, 'train.csv'))
        train_df = train_df.sort_values('time')
        return train_df

    def run(self):
        time_time = time.time()
        self.train_node2vector()
        print(time.time() - time_time)
        self.train_deepwalker()
        print(time.time() - time_time)


if __name__ == '__main__':
    GraphEmbeddingTask().run()
