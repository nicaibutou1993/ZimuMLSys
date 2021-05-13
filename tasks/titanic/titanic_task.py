# -*- coding: utf-8 -*-
from tasks.titanic.constant import *
from zimu_ml_sys.preprocess.data_eda import DataEDA
from zimu_ml_sys.feature.feature_eng import FeatureENG

import pandas as pd

pd.set_option('display.max_column', None)
pd.set_option('display.max_row', None)

train_df = pd.read_csv(Titanic_DATA_PATH + 'train.csv')
train_df.pop('PassengerId')

test_df = pd.read_csv(Titanic_DATA_PATH + 'test.csv')
test_df.pop('PassengerId')

eda = DataEDA(train_df)
# eda.fields_describe().fields_count_distribution(fields=['Survived','Pclass']) \
#     .fields_count_with_cross_feature_distribution(main_cross_feature_field='Survived',fields=['Pclass','Sex','SibSp'])\
#     .fields_cross_feature_kde_distribution(main_cross_feature_field='Pclass',fields=['Age']) \
#     .fields_cross_feature_kde_distribution(main_cross_feature_field='Survived',fields=['Age'])

eng = FeatureENG(task_name=TASK_NAME, is_load_file=True)


def feature_eng(data_frame, is_train=True):
    eng.set_data_frame(data_frame, is_train=is_train)
    frame = eng.fill_missing_value_by_model('Age', feature_fields=['Fare', 'Parch', 'SibSp', 'Pclass']).data_frame
    frame.loc[frame['Cabin'].notnull(), 'Cabin'] = "Yes"
    frame.loc[frame['Cabin'].isnull(), 'Cabin'] = "No"

    data_frame = eng.get_dummies(['Cabin', 'Embarked', 'Sex', 'Pclass']).data_frame

    data_frame.drop(columns=['Name', 'Ticket'], inplace=True)
    data_frame.fillna(0, inplace=True)
    data_frame = eng.standard_scaler_data(['Age', 'Fare']).data_frame
    return data_frame


train_df = feature_eng(train_df, is_train=True)

test_df = feature_eng(test_df, is_train=False)

eng.save_file()
