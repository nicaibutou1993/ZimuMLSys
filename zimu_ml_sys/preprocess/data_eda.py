# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import copy


class DataEDA(object):
    """
        数据分析：
        1. fields_describe： 数据基本描述
        2. fields_count_distribution：图表展示 每一个字段 分布情况
        3. fields_count_with_label_distribution：图表展示 每一个字段 与 label 字段 分布情况
        4. fields_heatmap： 字段与字段之间热力图
    """

    def __init__(self, data_frame):

        self.data_frame = data_frame

    def fields_describe(self, percentiles=[.01, .05, .10, .25, .5, .75, .90, .95, .99]):
        """
        数据 简单的描述：字段，是否趋势之，字段分布情况
        :param self.data_frame:
        :return:
        """

        if isinstance(self.data_frame, pd.DataFrame):
            print('----数据字段----')
            print(self.data_frame.columns)
            print()

            print('----数据类型相关说明----')
            print(self.data_frame.info())
            print()

            print('----数据分布情况----')
            print(self.data_frame.describe(percentiles))

        return self

    def fields_count_distribution(self, fields=None, is_print_count_value=True):
        """
        图表展示 每一个字段 count分布情况
        :param field: 默认None,所有的字段都会统计，支持 list，str
        :return:
        """

        def display(field):

            sns.countplot(self.data_frame[field])
            plt.show()

            if is_print_count_value:
                counter_data = Counter(self.data_frame[field])
                print("field name : ", field, " value_count: ", len(counter_data))
                print(counter_data)

        if isinstance(self.data_frame, pd.DataFrame):

            if fields is None:

                columns = self.data_frame.columns
                for field in columns:
                    display(field)

            elif isinstance(fields, str):
                display(fields)
            elif isinstance(fields, list):
                for field in fields:
                    display(field)

        return self

    def fields_count_with_label_distribution(self, label_field, fields=None, kind='bar'):
        """
        柱状图：
        图表展示 每一个字段 与 label 字段 count分布情况
        :param self.data_frame:
        :param label_field:
        :param fields:
        :return:
        """
        if isinstance(self.data_frame, pd.DataFrame):
            label_values = self.data_frame[label_field].unique()

            def display(field):

                if field == label_field:
                    return

                fig = plt.figure()
                fig.set(alpha=0.2)  # 设定图表颜色alpha参数

                field_label_mapping = {}
                for label_value in label_values:
                    field_label_mapping[label_field + '_' + str(label_value)] = self.data_frame[field][
                        self.data_frame[label_field] == label_value].value_counts()

                df = pd.DataFrame(field_label_mapping)
                df.plot(kind=kind, stacked=True)

                plt.xlabel(field)
                plt.ylabel("count")
                plt.show()

            if fields is None:

                columns = self.data_frame.columns

                for field in columns:
                    display(field)

            elif isinstance(fields, str):
                display(fields)
            elif isinstance(fields, list):
                for field in fields:
                    display(field)

        return self

    def fields_count_with_cross_feature_distribution(self, main_cross_feature_field, fields=None):
        """
        柱状图：
        表示 选择两个特征，查看在某个特征情况下
        :param main_cross_feature_field:
        :param fields:
        :param kind:
        :return:
        """
        self.fields_count_with_label_distribution(label_field=main_cross_feature_field, fields=fields)

        return self

    def fields_cross_feature_kde_distribution(self, main_cross_feature_field, fields=None):
        """
        曲线图：
        交叉特征展示：一般针对 连续变量 在 某个类别变量 表现的密度情况
        比如： 年龄：连续变量， 性别：类别变量， 分析 在每一个年龄值 查看 性别男女的比例 情况
        :param main_cross_feature_field: 类别变量，比如 性别
        :param fields: 可以是 连续变量 也可以是 类别变量，主要查看 密度
        :return:
        """

        if isinstance(self.data_frame, pd.DataFrame):
            values = self.data_frame[main_cross_feature_field].unique()

            def display(field):

                fig = plt.figure()
                fig.set(alpha=0.2)  # 设定图表颜色alpha参数

                for value in values:
                    self.data_frame[field][self.data_frame[main_cross_feature_field] == value].plot(kind='kde')

                plt.xlabel(field)  # plots an axis lable
                plt.ylabel(u"密度")
                plt.title(main_cross_feature_field)
                plt.legend(values, loc='best')
                plt.show()

            if fields is None:

                columns = self.data_frame.columns

                for field in columns:
                    display(field)

            elif isinstance(fields, str):
                display(fields)
            elif isinstance(fields, list):
                for field in fields:
                    display(field)

        return self

    def fields_heatmap(self, fields=None, is_encoder=True):
        """
        字段与字段之间的 协方差，
        观测 字段与字段 之间 线性关系
        :param fields:
        :return:
        """
        if isinstance(self.data_frame, pd.DataFrame):
            if fields is None:
                fields = self.data_frame.columns

            if isinstance(fields, list):
                df = copy.deepcopy(self.data_frame)
                if is_encoder:

                    for field in fields:
                        enc = LabelEncoder()
                        df[field] = enc.fit_transform(df[field])

                df = df.loc[:, fields]
                plt.figure(figsize=(len(fields) + 2, len(fields)))
                sns.heatmap(df.corr().abs(), annot=True)
                plt.show()

        return self
