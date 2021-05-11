# -*- coding: utf-8 -*-
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Binarizer
import os
import pickle
from zimu_ml_sys.constant import *


class FeaturePreProcess(object):
    """
        特征预处理：
            1. sampling_by_label: 样本采样，根据label 进行采样

    """
    DEFAULT_SAMPLING_NUM = 10000

    def __init__(self, data_frame):

        self.data_frame = data_frame

    def sampling_by_label(self, label_field=None, sampling_rate=None):
        """
        根据label 分布 进行采样， 保证 真实样本label分布 与 采样后保持一致 分层采样
        :param label_field: 传入标签字段名称
        :return:
        """
        if label_field:

            if sampling_rate is None:
                sampling_rate = self.DEFAULT_SAMPLING_NUM / float(len(self.data_frame))
                sampling_rate = 1.0 if sampling_rate > 1.0 else sampling_rate

            split = StratifiedShuffleSplit(n_splits=1, train_size=sampling_rate, random_state=39)

            for sample_index, _ in split.split(self.data_frame, self.data_frame[label_field]):
                self.data_frame = self.data_frame.loc[sample_index]
                self.data_frame.index = range(len(self.data_frame))

        return self

    def binary_fields(self, feature_fields, by_thresholds=None, by_counts=None):
        """
        二值化 某一个字段的值，将值 变成 0,1
        方式一：by_threshold：通过给定阈值，将数据直接转换成0,1
        方式二：by_count： 最有可能针对类别变量，不管类别变量是 数字还是字符串都可以 可能需要统计每一个值 计数，如果通过小于某个 分为0，大于count计数 1

        支持 传入多个 字段，by_thresholds 与 by_counts 只能使用一种
        :param feature_fields: 某些字段名称
        :param by_threshold: 使用 阈值 直接转换
        :param by_count: 先统计该字段下 没有值的 计数，根据设定的 计数 进行转换
        :return: cls
        """

        field_columns = self.data_frame.columns

        if field_columns.__contains__(feature_fields):

            if isinstance(feature_fields, str):
                feature_fields = [feature_fields]
                by_thresholds = [by_thresholds]
                by_counts = [by_counts]

            for feature_field, by_threshold, by_count in zip(feature_fields, by_thresholds, by_counts):

                if by_threshold:
                    binarizer = Binarizer(threshold=by_threshold)

                    self.data_frame[feature_field] = binarizer.transform(self.data_frame[feature_field])

                if by_count:
                    field_count = self.data_frame[feature_field].value_counts()

                    field_count_dict = dict(field_count)

                    self.data_frame[feature_field] = self.data_frame[feature_field] \
                        .apply(lambda x: 1 if field_count_dict.get(x) >= by_count else 0)

        return self

    def encoding_fields(self, feature_fields, task_name, save_file=True, mode='append', is_padding=True):
        """
        针对类别 编码，这里编码默认下标 以 1 开始，这里预留 index 0 为了以后模型需要 padding，padding 的index为 0
        :param feature_fields: 需要编码的特征
        :param task_name: 任务名称
        :param save_file: 是否 将 编码映射文件 保持
        :param mode: 默认 append模式，属于更新模型，会在原有的编码基础上，不停的更新迭代
        :param is_padding: True 下标1 开始 ，False 则下标 以 0 开始
        :return:cls
        """

        if isinstance(feature_fields, str):
            feature_fields = [feature_fields]

        parent_path = os.path.join(ENCODING_DATA_PATH, task_name)
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)

        for field in feature_fields:

            id2field, field2id = {}, {}

            path = os.path.join(parent_path, field + '.pkl')

            if mode == 'append':
                if os.path.exists(path):
                    id2field, field2id = pickle.load(open(path, mode='rb'))

            field_values = self.data_frame[field].unique()

            field_ids = set(field_values) - set(field2id.keys())

            if len(field_ids) > 0:

                field_num = len(field2id) + 1 if is_padding else len(field2id)

                _id2field = {i + field_num: field_id for i, field_id in enumerate(field_ids)}
                _field2id = {field_id: id for id, field_id in _id2field.items()}

                id2field.update(_id2field)
                field2id.update(_field2id)

                if save_file:
                    pickle.dump((id2field, field2id), open(path, mode='wb'))

            self.data_frame[field] = self.data_frame[field].apply(lambda x: field2id.get(x))

        return self
