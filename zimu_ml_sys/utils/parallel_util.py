# -*- coding: utf-8 -*-
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

"""
线程 并发
进程 并发
针对 pandas 分组操作 并发

"""


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


def thread_parallel(datas, func, has_return=True, n_jobs=5):
    """
    使用线程的方式 处理数据
    :param datas:
    :param fn:
    :param has_return:
    :param n_jobs:
    :return:
    """

    pool = ThreadPool(n_jobs)
    results = []

    if has_return:
        results = pool.map(func, datas)
    else:
        pool.map(func, datas)

    pool.close()
    pool.join()

    return results


def func(name):
    return name


def process_parallel(datas, func, has_return=True, n_jobs=5):
    """
    使用进程的方式 处理数据
    读取批量文件，进程速度 由于线程
    :param datas:
    :param fn:
    :param has_return:
    :param n_jobs:
    :return:
    """
    pool = Pool(n_jobs)

    results = []
    if has_return:
        results = pool.map(func, datas)
    else:
        pool.map(func, datas)

    pool.close()
    pool.join()
    return results


if __name__ == '__main__':
    path = 'E:\project\ZimuMLSys\data/feature_data\kddcup_2020/'

    res = process_parallel([path + 'itemcf_sim_item_11.pkl', path + 'itemcf_sim_item_10.pkl'], func, n_jobs=2)
    print(len(res))
