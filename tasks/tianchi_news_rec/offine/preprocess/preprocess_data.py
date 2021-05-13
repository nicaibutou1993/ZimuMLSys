from tasks.tianchi_news_rec.utils.spark_client import SparkBase
from pyspark.sql.functions import *
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import logging
import os
import pickle
from pyspark.sql.types import *
import numpy as np
from pyspark.storagelevel import StorageLevel
import pandas as pd
from tasks.tianchi_news_rec.constant import *

'''
排序模型：spark特征提取，最后写入到 tfrecord 文件中
'''


class PreprocessData(SparkBase):
    APP_NAME = 'preprocess_data'

    MASTER = 'local[10]'

    tf_record_path = 'E:/data/tianchi_news_rec/tf_recored/'

    # tf_record_path = '/root/zimu_news_recs/offline/tianchi_news_rec/tf_recored/'

    log_trace = logging.getLogger('pyspark')
    log_trace.setLevel(logging.WARN)

    def __init__(self):
        """
        设置spark 写入 tf-record 需要加载的jar
        """
        config = [("spark.jars", 'hdfs://zimu:9000/jars/spark-tensorflow-connector_2.11-1.15.0.jar')]
        self.spark = self._create_spark_session(config)

    def run(self):
        """
        代码执行入口
        :return:
        """
        self.user_logs_df = self.read_data()

        self.user_logs_df.cache()

        encode_mapping = self.encodding(self.user_logs_df)

        """针对日志出现的类别id，使用上面的新编码 进行替换"""
        self.decoding(encode_mapping)

        self.user_logs_df.cache()

        train_df, test_df = self.process_data()

        test_df = test_df.withColumn('label', F.lit(1))

        feature_df = self.neg_sample(train_df)

        feature_df = feature_df.select(CTR_FIELDS)

        test_df = test_df.select(CTR_FIELDS)

        """将测试集 训练集 写入到 tf-record 文件中"""

        feature_df.write.mode("overwrite").format("tfrecords").option("recordType", "Example").save(
            self.tf_record_path + "train_data")

        test_df.write.mode("overwrite").format("tfrecords").option("recordType", "Example").save(
            self.tf_record_path + "test_data")

        feature_df.show(1000, truncate=False)
        test_df.show(1000, truncate=False)

    def encodding(self, user_logs_df=None):
        """
        类别变量 编码，并存储，
        下一次 出现新的编码时：在后面 继续添加id
        考虑了动态 出现新的id，如何编码问题
        :param user_logs_df:
        :return:
        """

        field_map = {}
        for field in ENCODDING_FIELDS:

            id2field, field2id = {}, {}
            path = '../data/' + field + '.pkl'
            if os.path.exists(path):
                id2field, field2id = pickle.load(open(path, mode='rb'))

            if user_logs_df:

                field_ids = []

                for row in user_logs_df.select(field).distinct().collect():
                    field_ids.append(row[field])

                field_ids = set(field_ids) - set(field2id.keys())

                if len(field_ids) > 0:
                    field_num = len(field2id)
                    _id2field = {i + 1 + field_num: field_id for i, field_id in enumerate(field_ids)}
                    _field2id = {field_id: id for id, field_id in _id2field.items()}

                    id2field.update(_id2field)
                    field2id.update(_field2id)

                    pickle.dump((id2field, field2id), open(path, mode='wb'))

            field_map[field] = (id2field, field2id)

        return field_map

    def decoding(self, encode_mapping):

        for field, (_, field2id) in encode_mapping.items():
            func = udf(lambda x: field2id.get(x, 0), IntegerType())
            self.user_logs_df = self.user_logs_df.withColumn(field, func(F.col(field)))

    def read_data(self):
        """
        加载资讯数据 及 用户点击日志
        1. 去除用户重复点击 数据，以最后一次点击时间
        :return:
        """

        # where dt='20171003'
        user_logs_df = self.spark.sql("""select * from tianchi_news_rec.user_logs """)

        article_df = self.spark.sql("""select * from tianchi_news_rec.article_data""")

        user_logs_df = user_logs_df.join(article_df, on='article_id', how='left')

        user_logs_df = user_logs_df.withColumn('rn',
                                               F.row_number().over(
                                                   Window.partitionBy('user_id', 'article_id').
                                                       orderBy(F.col("click_timestamp").desc()))) \
            .where(F.col('rn') == 1) \
            .drop(F.col('rn'))

        return user_logs_df

    def process_data(self):
        """特征提取"""

        def padding_zero(inputs):
            """
            增加padding操作，针对用户历史点击行为，可能不足 20 条，会补充为0

            这里 针对如果用户一次都没有点击的话，也会训练一个 历史行为全是 0 这个向量特征
            :param inputs:
            :return:
            """
            if len(inputs) >= 20:

                return inputs
            else:

                inputs.extend([0 for _ in range(20 - len(inputs))])

            return inputs

        padding_zero_fn = udf(padding_zero, ArrayType(IntegerType()))

        """获取用户历史点击的资讯及资讯类别"""
        user_feature_df = self.user_logs_df.withColumn('history_click_articles',
                                                       F.collect_list(F.col('article_id'))
                                                       .over(Window.partitionBy('user_id')
                                                             .orderBy(F.col('click_timestamp').asc())
                                                             .rowsBetween(-20, -1))) \
            .withColumn('history_click_categories',
                        F.collect_list(F.col('category_id'))
                        .over(Window.partitionBy('user_id')
                              .orderBy(F.col('click_timestamp').asc())
                              .rowsBetween(-20, -1)))

        user_feature_df.persist(StorageLevel.MEMORY_AND_DISK)

        user_feature_df = user_feature_df.withColumn('history_click_articles',
                                                     padding_zero_fn(F.col('history_click_articles'))) \
            .withColumn('history_click_categories', padding_zero_fn(F.col('history_click_categories')))

        """拆分训练集 及测试集： 具体就是最后 一次用户点击 为测试集"""
        user_feature_df = user_feature_df.withColumn('rn', F.row_number()
                                                     .over(Window.partitionBy('user_id')
                                                           .orderBy(F.col('click_timestamp').desc())))

        user_feature_df.persist(StorageLevel.MEMORY_AND_DISK)

        test_df = user_feature_df.filter(F.col('rn') == 1)

        train_df = user_feature_df.filter(F.col('rn') > 1)

        return train_df, test_df

    def neg_sample(self, train_df):
        """
        负采样，针对ctr 问题，正样本 与负样本  1 : 4

        采集过程，是去除 用户已经点击的资讯，然后在全库中 随机采样
        :param train_df:
        :return:
        """
        user_click_articles_df = self.user_logs_df.select('user_id', 'article_id') \
            .groupby('user_id').agg(F.collect_list(F.col('article_id')).alias('click_articles'))

        article_ids_df = self.user_logs_df.select('article_id', 'category_id').distinct()

        article_ids_dict = {int(row.article_id): int(row.category_id) for row in article_ids_df.collect()}

        broadcast = self.spark.sparkContext.broadcast(article_ids_dict)

        train_df = train_df.join(user_click_articles_df, on='user_id', how='left')

        train_df.persist(StorageLevel.MEMORY_AND_DISK)

        def map_fun(partition):
            article_ids_dict = broadcast.value

            for row in partition:
                click_articles = row.click_articles
                sample_article_ids = set(article_ids_dict.keys()) - set(click_articles)

                data = []
                article_id = row.article_id
                category_id = row.category_id
                label = 1

                data.append((article_id, category_id, label))

                neg_article_ids = np.random.choice(list(sample_article_ids), 4)

                data.extend(
                    [(neg_article_id, article_ids_dict[neg_article_id], 0) for neg_article_id in neg_article_ids])

                for article_id, category_id, label in data:
                    yield row.user_id, int(article_id), row.click_environment, \
                          row.click_deviceGroup, row.click_os, row.click_country, \
                          row.click_region, row.click_referrer_type, category_id, \
                          row.history_click_articles, row.history_click_categories, label

        feature_df = train_df.rdd.mapPartitions(map_fun).toDF(CTR_FIELDS)

        return feature_df

    def read_tf_record(self):

        df = self.spark.read.format("tfrecords").option("recordType", "Example").load(
            self.tf_record_path + 'test_data')

        count = df.count()
        print(count)

    def get_item_feature(self):
        """
        获取item特征：主要是article_id category_id 编码后的ID
        这里返回的是dataframe
        :return:
        """
        data_path = PROJECT_ROOT_PATH + 'data/item_feature.csv'
        if os.path.exists(data_path):
            data_df = pd.read_csv(data_path, index_col=0)
        else:
            test_df = self.spark.read.format("tfrecords").option("recordType", "Example").load(
                self.tf_record_path + 'test_data')

            train_df = self.spark.read.format("tfrecords").option("recordType", "Example").load(
                self.tf_record_path + 'train_data')

            test_df = test_df.select("article_id", "category_id").distinct()

            train_df = train_df.select("article_id", "category_id").distinct()

            df = train_df.union(test_df)

            df = df.select("article_id", "category_id").distinct()

            data_df = df.toPandas()
            data_df.to_csv("../data/item_feature.csv")

        data_df = data_df[['article_id', 'category_id']]

        return data_df

    def get_user_history_items(self):
        """用户历史点击过的item，写入文件，用于出重"""

        data_path = PROJECT_ROOT_PATH + 'data/user_history_click_items.csv'
        if os.path.exists(data_path):
            data_df = pd.read_csv(data_path, index_col=0)
        else:

            test_df = self.spark.read.format("tfrecords").option("recordType", "Example").load(
                self.tf_record_path + 'test_data')

            train_df = self.spark.read.format("tfrecords").option("recordType", "Example").load(
                self.tf_record_path + 'train_data')

            train_df = train_df.filter(F.col("label") == 1).select("user_id", "article_id").distinct()
            test_df = test_df.filter(F.col("label") == 1).select("user_id", "article_id").distinct()

            df = train_df.union(test_df).select("user_id", "article_id").distinct()

            data_df = df.groupby("user_id").agg(F.collect_list("article_id").alias("history_click_items")). \
                select("user_id", "history_click_items").toPandas()

            data_df.to_csv("../data/user_history_click_items.csv")

        return data_df

    def get_test_user_features(self):
        """用户历史点击过的item，写入文件，用于出重"""

        data_path = PROJECT_ROOT_PATH + 'data/test_user_features.csv'
        if os.path.exists(data_path):
            test_user_df = pd.read_csv(data_path, index_col=0)
        else:
            id2user, _ = pickle.load(open(PROJECT_ROOT_PATH + 'data/user_id.pkl', mode='rb'))

            test_df = self.spark.read.format("tfrecords").option("recordType", "Example").load(
                self.tf_record_path + 'test_data')

            test_df.printSchema()

            print(test_df.rdd.getNumPartitions())

            def filter_test_user(encoding_user):

                decoder_user_id = id2user.get(encoding_user)

                if decoder_user_id >= 200000:
                    return 1
                else:
                    return 0

            fn = udf(filter_test_user, returnType=IntegerType())

            test_user_df = test_df.withColumn('rn', fn(F.col('user_id'))). \
                filter(F.col('rn') == 1).drop('article_id', 'category_id')

            test_user_df.toPandas().to_csv(data_path)

        return test_user_df


if __name__ == '__main__':
    # PreprocessData().read_tf_record()
    PreprocessData().get_test_user_features()
