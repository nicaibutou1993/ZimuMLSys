# -*- coding: utf-8 -*-

TASK_NAME = 'tianchi_news_rec'

"""ctr 字段"""
CTR_FIELDS = ['user_id', 'article_id', 'click_environment',
              'click_deviceGroup', 'click_os', 'click_country',
              'click_region',
              'click_referrer_type', 'category_id',
              'history_click_articles',
              'history_click_categories', 'label']

"""模型数据集 存放位置"""
CTR_TF_RECORD_DATA_PATH = 'E:/data/tianchi_news_rec/tf_recored/'

"""双塔模型最后一层输出维度"""
TOWER_OUTPUT_DIM = 128

