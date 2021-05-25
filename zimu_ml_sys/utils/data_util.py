# -*- coding: utf-8 -*-
import os.path
import pickle
from zimu_ml_sys.constant import *
import glob
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from multiprocessing.pool import ThreadPool

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


def df_group_parallel(df_group_data, func, n_jobs=None):
    """
    针对 pandas dataframe 进行groupby 操作 进行并行操作，加快速度
    :param df_group_data: df.grouby(field) 所得数据
    :param func: 针对分组数据 进行处理的函数
    :param n_jobs: 并行度
    :return:
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()

    ret = Parallel(n_jobs=n_jobs)(delayed(func)(name, group) for name, group in df_group_data)
    return pd.concat(ret)


def list_data_parallel(datas,fn):

    pool_size = 20
    pool = ThreadPool(pool_size)  # 创建一个线程池
    pool.map(fn, datas)  # 往线程池中填线程
    pool.close()  # 关闭线程池，不再接受线程
    pool.join()







if __name__ == '__main__':
    read_encoding_mapping_data('avazu')
