# -*- coding: utf-8 -*-
import pandas as pd
from tasks.constant import *
from zimu_ml_sys.preprocess.data_eda import DataEDA
from zimu_ml_sys.preprocess.feature_preprocess import FeaturePreProcess

data_df = pd.read_csv(AVAZU_DATA_PATH)

data_eda = DataEDA(data_df)

# ,label_field='click',is_sampling=True,sampling_rate=0.2
data_df = FeaturePreProcess(data_df). \
    sampling_by_label(label_field='click', sampling_rate=0.2)

pd.set_option('display.max_column', None)
pd.set_option('display.max_row', None)

# print(data_df.columns)

# data_eda.fields_heatmap(['C1', 'site_id', 'site_domain',
#                          'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
#                          'device_ip', 'device_model'])
