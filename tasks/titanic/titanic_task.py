# -*- coding: utf-8 -*-
from tasks.titanic.constant import *
from zimu_ml_sys.preprocess.data_eda import DataEDA
from zimu_ml_sys.feature.feature_eng import FeatureENG


import pandas as pd
pd.set_option('display.max_column',None)
pd.set_option('display.max_row',None)

train_df = pd.read_csv(Titanic_DATA_PATH + 'train.csv')

eda = DataEDA(train_df)

eng = FeatureENG()

eng.set_data_frame(train_df)


frame = eng.fill_missing_value_by_model('Age', feature_fields=['Fare', 'Parch', 'SibSp', 'Pclass']).data_frame
frame.loc[ frame['Cabin'].notnull(),'Cabin'] = "Yes"
frame.loc[ frame['Cabin'].isnull(),'Cabin'] = "No"

data_frame = eng.get_dummies(['Cabin', 'Embarked', 'Sex', 'Pclass']).data_frame

data_frame.drop(columns=['Name','Ticket'],inplace=True)


print(data_frame.head())

# eda.fields_describe().fields_count_distribution(fields=['Survived','Pclass']) \
#     .fields_count_with_cross_feature_distribution(main_cross_feature_field='Survived',fields=['Pclass','Sex','SibSp'])\
#     .fields_cross_feature_kde_distribution(main_cross_feature_field='Pclass',fields=['Age']) \
#     .fields_cross_feature_kde_distribution(main_cross_feature_field='Survived',fields=['Age'])





