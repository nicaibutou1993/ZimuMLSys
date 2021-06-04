# -*- coding: utf-8 -*-
import os.path
import pickle
from zimu_ml_sys.constant import *
import glob
import numpy as np

"""
1. read_encoding_mapping_data: 编码mapping
2. df_group_parallel: dataframe 分组 并行化
"""


def read_encoding_mapping_data(task_name=TASK_NAME, feature_fields=None):
    """
    根据类别变量名称 读取编码mapping
    :param feature_fields: list or str 表示 字段名称
    :param task_name: 任务名称
    :return: dict, key = 字段 ,value = (id2field, field2id)
    """
    if isinstance(feature_fields, str):
        feature_fields = [feature_fields]

    field_map = {}
    parent_data_path = os.path.join(ENCODING_DATA_PATH, task_name)

    if feature_fields is None:
        pkl_files = glob.glob(os.path.join(parent_data_path, '*.pkl'))

        feature_fields = list(map(lambda x: os.path.basename(x).replace('.pkl', ''), pkl_files))

    for field in feature_fields:

        id2field, field2id = {}, {}
        path = os.path.join(parent_data_path, field + '.pkl')
        if os.path.exists(path):
            id2field, field2id = pickle.load(open(path, mode='rb'))

        field_map[field] = (id2field, field2id)
    return field_map


class DataGenerator(object):
    """数据生成器模版
    """

    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self, random=True):
        while True:
            for d in self.__iter__(random):
                yield d


# def df_group_parallel(df_group_data, func, n_jobs=None):
#     """
#     针对 pandas dataframe 进行groupby 操作 进行并行操作，加快速度
#     :param df_group_data: df.grouby(field) 所得数据
#     :param func: 针对分组数据 进行处理的函数
#     :param n_jobs: 并行度
#     :return:
#     """
#     if n_jobs is None:
#         n_jobs = multiprocessing.cpu_count()
#
#     ret = Parallel(n_jobs=n_jobs)(delayed(func)(name, group) for name, group in df_group_data)
#     return pd.concat(ret)
#
#
# def list_data_parallel(datas,fn):
#
#     pool_size = 20
#     pool = ThreadPool(pool_size)  # 创建一个线程池
#     pool.map(fn, datas)  # 往线程池中填线程
#     pool.close()  # 关闭线程池，不再接受线程
#     pool.join()


if __name__ == '__main__':
    read_encoding_mapping_data('avazu')
