# -*- coding: utf-8 -*-

import glob

import tensorflow as tf
import tensorflow.keras as keras

from tasks.tianchi_news_rec.constant import *
from zimu_ml_sys.core.feature_columns import SparseFeat, VarLenSparseFeat
from zimu_ml_sys.core.models.dcn import DCN
from zimu_ml_sys.utils.data_util import read_encoding_mapping_data
from zimu_ml_sys.utils.tf_record_util import get_tf_record_beans, tf_record_to_dataset

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class DCNTask(object):
    embedding_size = 16

    batch_size = 512

    epochs = 10

    label_name = CTR_FIELDS.pop(-1)

    def get_feature_columns(self):

        sparse_features = [field for field in CTR_FIELDS if not field.startswith('history')]

        encoding_fields_mapping = read_encoding_mapping_data(TASK_NAME)

        dnn_feature_colums = [SparseFeat(name=field,
                                         vocabulary_size=len(encoding_fields_mapping[field][0]) + 1,
                                         embedding_dim=self.embedding_size) for field in sparse_features]

        dnn_feature_colums += [VarLenSparseFeat(SparseFeat(name='history_click_articles',
                                                           vocabulary_size=len(
                                                               encoding_fields_mapping['article_id'][0]) + 1,
                                                           embedding_dim=self.embedding_size,
                                                           embedding_name='article_id'),
                                                maxlen=20),
                               VarLenSparseFeat(SparseFeat(name='history_click_categories',
                                                           vocabulary_size=len(
                                                               encoding_fields_mapping['category_id'][0]) + 1,
                                                           embedding_dim=self.embedding_size,
                                                           embedding_name='category_id'),
                                                maxlen=20)
                               ]

        return dnn_feature_colums

    def get_dataset(self, feature_columns, is_train=True):
        """
        获取dataset 数据集
        :param feature_columns:
        :param is_train:
        :return:
        """

        tf_record_beans = get_tf_record_beans(feature_columns, label_name=self.label_name)

        if is_train:
            path = CTR_TF_RECORD_DATA_PATH + 'train_data/' + "part*"
        else:
            path = CTR_TF_RECORD_DATA_PATH + 'test_data/' + "part*"

        files_list = glob.glob(path)

        dataset = tf_record_to_dataset(files_list,
                                       tf_record_beans=tf_record_beans,
                                       is_train=is_train,
                                       batch_size=self.batch_size,
                                       is_shuffle=True,
                                       epochs=self.epochs)

        return dataset

    def train(self):
        feature_columns = self.get_feature_columns()

        train_dataset = self.get_dataset(feature_columns, is_train=True)
        test_dataset = self.get_dataset(feature_columns, is_train=False)

        model = DCN(feature_columns)

        model.summary()

        model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

        model.fit(train_dataset, epochs=self.epochs, steps_per_epoch=13358,
                  validation_data=test_dataset,
                  validation_steps=489)


if __name__ == '__main__':
    DCNTask().train()
