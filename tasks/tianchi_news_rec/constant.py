# -*- coding: utf-8 -*-

TASK_NAME = 'tianchi_news_rec'

"""ctr 字段"""
CTR_FIELDS = ['user_id', 'article_id', 'click_environment',
              'click_deviceGroup', 'click_os', 'click_country',
              'click_region',
              'click_referrer_type', 'category_id',
              'history_click_articles',
              'history_click_categories', 'label']

"""需要编码的字段"""
ENCODDING_FIELDS = ['user_id', 'article_id', 'click_environment', 'click_deviceGroup', 'click_os',
                    'click_country', 'click_region', 'click_referrer_type', 'category_id']


"""模型数据集 存放位置"""
CTR_TF_RECORD_DATA_PATH = 'E:/data/tianchi_news_rec/tf_recored/'

"""双塔模型最后一层输出维度"""
TOWER_OUTPUT_DIM = 128

"""双塔模型 user 侧 tf-serving-url"""
TF_SERVING_USER_TOWER_URL = 'http://zimu:8501/v1/models/user_tower:predict'

"""双塔模型 item 侧 faiss-url"""
FAISS_ITEM_TOWER_URL = 'http://zimu:10001/tower/rec_items'

HIVE_METASTORE_URIS = 'thrift://192.168.18.99:9083'
SPARK_SQL_WAREHOUSE_DIR = "/user/hive/warehouse"

