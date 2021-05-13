# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Binarizer
import pickle
import sys, os
from zimu_ml_sys.constant import *


class FeatureENG(object):
    """特征工程  可以保存 相应的模型数据，方便测试集 处理数据
        1. 创建对象，并是否加载 之前处理的模型及数据 相关数据
        2. 设置 数据集：并指定 训练阶段还是测试阶段
        3. 执行相关操作 ....
        4. 返回处理后的数据集
        5. 全部处理完成：是否将相关的模型及数据 写入到文件

        操作：
        1.fill_missing_value_by_model: 使用 随机森林 预测 缺失值
        2.get_dummies: 将类别特征 one_hot处理
        3.standard_scaler_data: 标准化数据
        4.sampling_by_label: 根据标签 进行数据采样，保证采样的正负样本与 原始的正负样本 保存均衡
        5.binary_fields： 将字段值 进行二值化处理，支持 阈值 及 计数 两种方式
        6.encoding_fields： 针对 id 类型的变量，进行编码，支持padding

    """
    DEFAULT_SAMPLING_NUM = 10000

    feature_operation_save_path = ''
    feature_operation_mapping = {}

    def __init__(self, task_name=TASK_NAME, is_load_file=False):
        """
        :param save_file: 是否加载 已经保存的操作模型及数据
        :param task_name: 当前执行任务名称
        """
        self.task_name = task_name

        if bool(task_name) & is_load_file:
            parent_path = os.path.join(FEATURE_DATA_PATH, task_name)
            self.feature_operation_save_path = os.path.join(parent_path, 'feature_eng_operation.pkl')

        if bool(self.feature_operation_save_path) & os.path.exists(self.feature_operation_save_path):
            self.feature_operation_mapping = pickle.load(open(self.feature_operation_save_path, mode='rb'))

        self.is_load_file = False
        if is_load_file and len(self.feature_operation_mapping) > 0:
            self.is_load_file = True

    def save_file(self):
        """
        将特征工程 操作模型 保存至文件
        :return:
        """
        if len(self.feature_operation_mapping) > 0:
            if self.feature_operation_save_path:
                parent_path = os.path.dirname(self.feature_operation_save_path)
                if not os.path.exists(parent_path):
                    os.makedirs(parent_path)
                pickle.dump(self.feature_operation_mapping, open(self.feature_operation_save_path, mode='wb'))

    def set_data_frame(self, data_frame, is_train=True):
        """设置数据集，并制定 是训练阶段 还是 测试 阶段"""

        self.data_frame = data_frame
        self.is_train = is_train

    @property
    def get_data_frame(self):
        return self.data_frame

    def fill_missing_value_by_model(self, miss_field_name, feature_fields, mode='regression'):
        """
        使用随机森林 方式 进行对缺失填充： 支持回归与分类
        :param miss_field_name: 缺失值字段
        :param feature_fields: 训练特征字段
        :param mode:
        :return:
        """

        operate_name = sys._getframe().f_code.co_name + '_' + miss_field_name

        if self.is_train and not self.is_load_file:
            train_data = self.data_frame[self.data_frame[miss_field_name].notnull()]
            train_x = train_data.loc[:, feature_fields].fillna(0)
            train_y = train_data.loc[:, miss_field_name]

            if mode == 'regression':
                estimator = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=39)
            else:
                estimator = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=39)

            estimator.fit(train_x, train_y)

            self.feature_operation_mapping[operate_name] = estimator
        else:

            estimator = self.feature_operation_mapping.get(operate_name)

        predict_data = self.data_frame[self.data_frame[miss_field_name].isnull()]
        predict_x = predict_data.loc[:, feature_fields].fillna(0)

        self.data_frame.loc[(self.data_frame[miss_field_name].isnull()), miss_field_name] = estimator.predict(
            predict_x)

        return self

    def get_dummies(self, field_names, is_drop_o_fields=True):
        """
        将字段里面的值 进行onehot 处理
        1. 训练集 value 出现，而测试集value没有出现的数据，进行增加字段并 填补0
        2. 训练集 value 没有出现，而测试集value出现的数据，直接删除该字段
        :param field_names:
        :param is_drop_o_fields:
        :return:
        """
        operate_name_prefix = sys._getframe().f_code.co_name + '_'

        if isinstance(field_names, str):
            field_names = [field_names]

        fields_dummies = []
        fields_dummies.append(self.data_frame)

        for field in field_names:

            operate_name = operate_name_prefix + field

            field_dummies = pd.get_dummies(self.data_frame[field], prefix=field)

            """
                1. 训练集 value 出现，而测试集value没有出现的数据，进行增加字段并 填补0
                2. 训练集 value 没有出现，而测试集value出现的数据，直接删除该字段
            """
            c_field_names = field_dummies.columns
            if not self.is_train:
                train_fields = self.feature_operation_mapping.get(operate_name)
                fill_fields = set(train_fields) - set(c_field_names)
                if len(fill_fields) > 0:
                    field_dummies[list(fill_fields)] = 0
                    field_dummies = field_dummies[train_fields]
            else:
                if not self.is_load_file:
                    self.feature_operation_mapping[operate_name] = c_field_names
            fields_dummies.append(field_dummies)

        self.data_frame = pd.concat(fields_dummies, axis=1)

        if is_drop_o_fields:
            self.data_frame.drop(columns=field_names, inplace=True)
        return self

    def standard_scaler_data(self, field_names):
        """
        对数据 进行标准化处理
        :param field_names: 针对处理的字段名称， 类型 str or list
        :return:
        """
        operate_name_prefix = sys._getframe().f_code.co_name + '_'
        if isinstance(field_names, str):
            field_names = [field_names]

        scaler = StandardScaler()
        for field_name in field_names:
            operate_name = operate_name_prefix + field_name
            if self.is_train and not self.is_load_file:
                model = scaler.fit(self.data_frame[[field_name]])
                self.feature_operation_mapping[operate_name] = model
            else:
                model = self.feature_operation_mapping.get(operate_name)
            self.data_frame[field_name] = model.transform(self.data_frame[[field_name]])
        return self

    def sampling_by_label(self, label_field_name=None, sampling_rate=None):
        """
        根据label 分布 进行采样， 保证 真实样本label分布 与 采样后保持一致 分层采样
        :param label_field_name: 传入标签字段名称
        :return:
        """
        if label_field_name:
            if sampling_rate is None:
                sampling_rate = self.DEFAULT_SAMPLING_NUM / float(len(self.data_frame))
                sampling_rate = 1.0 if sampling_rate > 1.0 else sampling_rate

            split = StratifiedShuffleSplit(n_splits=1, train_size=sampling_rate, random_state=39)
            for sample_index, _ in split.split(self.data_frame, self.data_frame[label_field_name]):
                self.data_frame = self.data_frame.loc[sample_index]
                self.data_frame.index = range(len(self.data_frame))
        return self

    def binary_fields(self, field_names, by_thresholds=None, by_counts=None):
        """
        二值化 某一个字段的值，将值 变成 0,1
        方式一：by_threshold：通过给定阈值，将数据直接转换成0,1
        方式二：by_count： 最有可能针对类别变量，不管类别变量是 数字还是字符串都可以 可能需要统计每一个值 计数，如果通过小于某个 分为0，大于count计数 1

        支持 传入多个 字段，by_thresholds 与 by_counts 只能使用一种
        :param field_names: 某些字段名称
        :param by_threshold: 使用 阈值 直接转换
        :param by_count: 先统计该字段下 没有值的 计数，根据设定的 计数 进行转换
        :return: cls
        """
        operate_name_prefix = sys._getframe().f_code.co_name + '_'
        field_columns = self.data_frame.columns

        if field_columns.__contains__(field_names):

            if isinstance(field_names, str):
                field_names = [field_names]
                by_thresholds = [by_thresholds]
                by_counts = [by_counts]

            for feature_field, by_threshold, by_count in zip(field_names, by_thresholds, by_counts):
                if by_threshold:
                    binarizer = Binarizer(threshold=by_threshold)
                    self.data_frame[feature_field] = binarizer.transform(self.data_frame[feature_field])

                if by_count:
                    operate_name = operate_name_prefix + feature_field
                    if self.is_train and not self.is_load_file:
                        field_count = self.data_frame[feature_field].value_counts()
                        field_count_dict = dict(field_count)
                        self.feature_operation_mapping[operate_name] = field_count_dict
                    else:
                        field_count_dict = self.feature_operation_mapping.get(operate_name)

                    self.data_frame[feature_field] = self.data_frame[feature_field] \
                        .apply(lambda x: 1 if field_count_dict.get(x) >= by_count else 0)
        return self

    def encoding_fields(self, field_names, save_file=True, mode='append', is_padding=True):
        """
        padding：0
            1. 作用一： 针对序列，可以为0
            2. 作用二： 针对unk, 还是设置 该id 为0，即表示padding 备注： 在SequencePoolingLayer 针对unk 会进行处理成unk 向量

        1. 训练阶段：
            1. 第一次训练模型 需要 index 需要从1 开始，预留0 0：padding
            2. 第二次训练模型 在原有的index 增加一些 未知的index 使用append 方式

        2. 预测阶段：
            1. 如果出现新的id，应该为 index 0 : padding 方式

        :param field_names: 需要编码的特征
        :param save_file: 是否 将 编码映射文件 保持
        :param mode: 默认 append模式，属于更新模型，会在原有的编码基础上，不停的更新迭代
        :param is_padding: True 下标1 开始 ，False 则下标 以 0 开始
        :return:cls
        """
        if isinstance(field_names, str):
            field_names = [field_names]

        parent_path = os.path.join(ENCODING_DATA_PATH, self.task_name)
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)

        for field in field_names:

            id2field, field2id = {}, {}
            path = os.path.join(parent_path, field + '.pkl')
            if mode == 'append' or not self.is_train:
                if os.path.exists(path):
                    id2field, field2id = pickle.load(open(path, mode='rb'))

            if self.is_train and not self.is_load_file:
                field_values = self.data_frame[field].unique()
                field_ids = set(field_values) - set(field2id.keys())
                if len(field_ids) > 0:
                    field_num = len(field2id) + sum([is_padding])
                    _id2field = {i + field_num: field_id for i, field_id in enumerate(field_ids)}
                    _field2id = {field_id: id for id, field_id in _id2field.items()}

                    id2field.update(_id2field)
                    field2id.update(_field2id)

                    if save_file:
                        pickle.dump((id2field, field2id), open(path, mode='wb'))

            self.data_frame[field] = self.data_frame[field].apply(lambda x: field2id.get(x, 0))

        return self


if __name__ == '__main__':
    FeatureENG().fill_missing_value_by_model('', '')
