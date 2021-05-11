# -*- coding: utf-8 -*-
import os.path
import pickle
from zimu_ml_sys.constant import *
import glob


def read_encoding_mapping_data(task_name, feature_fields=None):
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





if __name__ == '__main__':
    read_encoding_mapping_data('avazu')
