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

    def fields_count_distribution(self, fields=None, is_print_count_value=True):
        """
        图表展示 每一个字段 分布情况
        :param self.data_frame:
        :param field:
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

    def fields_count_with_label_distribution(self, label_field, fields=None):
        """
        图表展示 每一个字段 与 label 字段 分布情况
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
                df.plot(kind='bar', stacked=True)

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
