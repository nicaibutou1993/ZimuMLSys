# -*- coding: utf-8 -*-
import pandas as pd
from zimu_ml_sys.eda.data_eda import DataEDA
from zimu_ml_sys.eda.feature_preprocess import FeaturePreProcess
from tasks.tianchi_news_rec.constant import *

pd.set_option('display.max_column', None)
pd.set_option('display.max_row', None)
data_df = pd.read_csv(AVAZU_DATA_PATH)
data_eda = DataEDA(data_df)

feature_process = FeaturePreProcess(data_df)
feature_process.encoding_fields('banner_pos','avazu')
#data_eda.fields_count_distribution()
