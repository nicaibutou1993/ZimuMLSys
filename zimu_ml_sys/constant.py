# -*- coding: utf-8 -*-
import os

MODULE_NAME = 'zimu_ml_sys'

TASK_NAME = 'zimu_ml'

CURRENT_PROJECT_PATH = os.path.abspath(__file__).split(MODULE_NAME)[0]

"""编码 数据路径"""
ENCODING_DATA_PATH = os.path.join(CURRENT_PROJECT_PATH, os.path.join('data', 'encoding_data'))

"""模型保存路径"""
CHECKPOINT_DATA_PATH = os.path.join(CURRENT_PROJECT_PATH, os.path.join('data', 'checkpoint'))

"""特征数据保存路径"""
FEATURE_DATA_PATH = os.path.join(CURRENT_PROJECT_PATH, os.path.join('data', 'feature_data'))

"""预处理数据"""
PREPROCESS_DATA_PATH = os.path.join(CURRENT_PROJECT_PATH, os.path.join('data', 'preprocess_data'))