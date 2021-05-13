# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import sys


class FeatureENG(object):
    """特征工程
        1. 创建对象，并是否加载 之前处理的模型及数据 相关数据
        2. 设置 数据集：并指定 训练阶段还是测试阶段
        3. 执行相关操作 ....
        4. 返回处理后的数据集
        5. 全部处理完成：是否将相关的模型及数据 写入到文件
    """

    def __init__(self, save_file=False):

        self.operation_mapping = {}

    @property
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

        if self.is_train:
            train_data = self.data_frame[self.data_frame[miss_field_name].notnull()]
            train_x = train_data.loc[:, feature_fields].fillna(0)
            train_y = train_data.loc[:, miss_field_name]

            if mode == 'regression':
                estimator = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=39)
            else:
                estimator = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=39)

            estimator.fit(train_x, train_y)

            self.operation_mapping[operate_name] = operate_name
        else:

            estimator = self.operation_mapping.get(operate_name)

        predict_data = self.data_frame[self.data_frame[miss_field_name].isnull()]
        predict_x = predict_data.loc[:, feature_fields].fillna(0)

        self.data_frame.loc[(self.data_frame[miss_field_name].isnull()), miss_field_name] = estimator.predict(predict_x)

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
            field_names = field_dummies.columns
            if not self.is_train:
                train_fields = self.operation_mapping.get(operate_name)

                fill_fields = set(train_fields) - set(field_names)
                field_dummies[fill_fields] = 0
                field_dummies = field_dummies[train_fields]
            else:
                self.operation_mapping[operate_name] = field_names
            fields_dummies.append(field_dummies)

        self.data_frame = pd.concat(fields_dummies, axis=1)
        if is_drop_o_fields:
            self.data_frame.drop(columns=field_names, inplace=True)

        return self

    def standard_scaler_data(self, field_names):

        if isinstance(field_names, str):
            field_names = [field_names]

        scaler = StandardScaler()
        for field_name in field_names:
            scaler.fit(self.data_frame[field_name])


if __name__ == '__main__':
    FeatureENG().fill_missing_value_by_model('', '')
