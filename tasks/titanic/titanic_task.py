# -*- coding: utf-8 -*-
from tasks.titanic.constant import *
from zimu_ml_sys.eda.data_eda import DataEDA
from zimu_ml_sys.feature.feature_eng import FeatureENG
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import BaggingRegressor
import numpy as np

pd.set_option('display.max_column', None)
pd.set_option('display.max_row', None)

train_df = pd.read_csv(Titanic_DATA_PATH + 'train.csv')
train_df.pop('PassengerId')

test_df = pd.read_csv(Titanic_DATA_PATH + 'test.csv')
test_df.pop('PassengerId')

"""特征探索"""
eda = DataEDA(train_df)
# eda.fields_describe().fields_count_distribution(fields=['Survived','Pclass']) \
#     .fields_count_with_cross_feature_distribution(main_cross_feature_field='Survived',fields=['Pclass','Sex','SibSp'])\
#     .fields_cross_feature_kde_distribution(main_cross_feature_field='Pclass',fields=['Age']) \
#     .fields_cross_feature_kde_distribution(main_cross_feature_field='Survived',fields=['Age'])

eng = FeatureENG(task_name=TASK_NAME, is_load_file=True)


def feature_eng(data_frame, is_train=True):
    """特征工程"""
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

train_y = train_df.pop('Survived')

"""单个模型，并查看模型特征重要性，正值越大，表明 该特征成正相关，负值越大，表明该特征成负相关，正负相关都是模型重要的特征，
如果特征重要性为0，表示该特征对模型不起作用"""
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_df, train_y)
frame = pd.DataFrame({"columns": list(train_df.columns), "coef": list(clf.coef_.T)})
validate = cross_validate(clf, train_df, train_y, cv=5)
print(frame)
print(validate)

"""当模型融合，单个模型，针对特征及样本采样，形成新的模型，集成所有新的模型，一般会比单模型 提升2%左右的效果"""
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
                               bootstrap_features=False, n_jobs=1)
bagging_clf.fit(train_df, train_y)
predict = bagging_clf.predict(test_df).astype(np.int32)
print(predict)
