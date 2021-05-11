# -*- coding: utf-8 -*-
import os

MODULE_NAME = 'zimu_ml_sys'

CURRENT_PROJECT_PATH = os.path.abspath(__file__).split(MODULE_NAME)[0]

ENCODING_DATA_PATH = os.path.join(CURRENT_PROJECT_PATH, os.path.join('data', 'encoding_data'))

CHECKPOINT_DATA_PATH = os.path.join(CURRENT_PROJECT_PATH, os.path.join('data', 'checkpoint'))
