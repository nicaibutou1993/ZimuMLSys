# -*- coding: utf-8 -*-
from sklearn.model_selection import StratifiedShuffleSplit


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
        :param data_frame:
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
