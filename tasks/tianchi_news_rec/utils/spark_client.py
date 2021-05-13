# -*- coding: utf-8 -*-
from tasks.tianchi_news_rec.constant import *

import os

os.environ['SPARK_HOME'] = 'F:\spark\spark-2.4.3-bin-hadoop2.7'
os.environ['HADOOP_USER_NAME'] = 'root'

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession


class SparkBase(object):
    APP_NAME = None

    MASTER = None

    def _create_spark_session(self, config=None):
        _config = [
            ("spark.app.name", self.APP_NAME),
            ("spark.sql.warehouse.dir", SPARK_SQL_WAREHOUSE_DIR),
            ("hive.metastore.uris", HIVE_METASTORE_URIS),
            ("dfs.replication", "1")
        ]
        conf = SparkConf()

        if self.MASTER:
            conf.setMaster(self.MASTER)

        if config:
            for k, v in config:
                _config.append((k, v))

        conf.setAll(_config)

        spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

        return spark
