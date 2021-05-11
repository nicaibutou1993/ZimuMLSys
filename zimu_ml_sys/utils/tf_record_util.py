# -*- coding: utf-8 -*-
import tensorflow as tf
from zimu_ml_sys.core.feature_columns import VarLenSparseFeat
from collections import namedtuple

SHUFFLE_DEFAULT = 10000


class TFRecordBean(namedtuple('TFRecordBean', ['name', 'shape', 'dtype', 'is_label'])):
    """tf-record-bean 暂时 只是支持 FixedLenFeature：
        外部需要 创建 bean，传入解析tf-record 数据
        name： 字段名称
        shape：数据形状 去除batch_size
        dtype: 数据类型
        is_label: 如果是 true,表示 label 字段
    """

    def __new__(cls, name, shape=[], dtype=tf.int64, is_label=False):
        return super(TFRecordBean, cls).__new__(cls, name, shape, dtype, is_label)


def get_tf_record_beans(feature_columns, label_name=None):
    """
    根据字段信息 获取 TFRecordBean
    :param feature_columns: 字段
    :param label_name:
    :return:
    """
    beans = []
    for feature_column in feature_columns:
        dtype = tf.as_dtype(feature_column.dtype)
        if isinstance(feature_column, VarLenSparseFeat):
            beans.append(TFRecordBean(feature_column.name, [feature_column.maxlen], dtype=dtype))

        beans.append(TFRecordBean(feature_column.name, dtype=dtype))

    if label_name:
        beans.append(TFRecordBean(label_name, is_label=True))
    return beans


def tf_record_to_dataset(files_list,
                         tf_record_beans=None,
                         is_train=True,
                         batch_size=128,
                         is_shuffle=True,
                         epochs=None):
    """
    从tf-record 文件中读取，转换为 dataset 数据类型
    :param files_list: tf-record 文件集
    :param tf_record_beans: 定义tf-record 数据解析格式
    :param is_train: 是否是训练集
    :param batch_size:
    :param is_shuffle:
    :param epochs: 训练轮次
    :return:
    """
    dataset = tf.data.TFRecordDataset(files_list)
    if isinstance(tf_record_beans, list):

        def parse_dataset(serial_exmp):
            fields_mapping = {}

            label_name = None
            for bean in tf_record_beans:
                fields_mapping[bean.name] = tf.io.FixedLenFeature(bean.shape, bean.dtype)
                if bean.is_label:
                    label_name = bean.name

            features = tf.io.parse_example(serial_exmp, features=fields_mapping)

            if label_name:
                label = features.pop(label_name)
                return features, label
            return features

        dataset = dataset.map(parse_dataset)

        if is_train:
            if is_shuffle:
                dataset = dataset.shuffle(SHUFFLE_DEFAULT, seed=39)
            if epochs:
                dataset = dataset.repeat(epochs)

        dataset = dataset.batch(batch_size)

    return dataset


if __name__ == '__main__':
    bean = TFRecordBean('user_id', 1, is_label=False)
    print(bean)
