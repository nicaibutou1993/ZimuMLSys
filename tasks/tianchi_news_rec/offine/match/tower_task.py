# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from zimu_ml_sys.constant import CHECKPOINT_DATA_PATH
from zimu_ml_sys.core.feature_columns import SparseFeat, VarLenSparseFeat
from zimu_ml_sys.utils.data_util import read_encoding_mapping_data
from zimu_ml_sys.utils.tf_record_util import get_tf_record_beans, tf_record_to_dataset
from zimu_ml_sys.core.models.tower import Tower

import glob
from tasks.tianchi_news_rec.constant import *
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

"""
26ms/step - loss: 0.0978 - accuracy: 0.9643 - val_loss: 0.4616 - val_accuracy: 0.8564
"""


class TowerTask(object):
    embedding_size = 16

    batch_size = 512

    epochs = 10

    model_path = os.path.join(CHECKPOINT_DATA_PATH, 'tower/whole_tower/model')

    label_name = CTR_FIELDS.pop(-1)

    def get_fields(self):
        """
        特征
        :return:
        """

        sparse_features = [field for field in CTR_FIELDS if not field.startswith('history')]

        item_features = ['article_id', 'category_id']

        user_features = filter(lambda x: x not in item_features, sparse_features)

        encoding_fields_mapping = read_encoding_mapping_data('tianchi_news_rec')

        user_feature_colums = [SparseFeat(name=field,
                                          vocabulary_size=len(encoding_fields_mapping[field][0]) + 1,
                                          embedding_dim=self.embedding_size) for field in user_features]

        user_feature_colums += [VarLenSparseFeat(SparseFeat(name='history_click_articles',
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

        item_feature_colums = [SparseFeat(name=field,
                                          vocabulary_size=len(encoding_fields_mapping[field][0]) + 1,
                                          embedding_dim=self.embedding_size) for field in item_features]

        return user_feature_colums, item_feature_colums

    def get_dataset(self, feature_columns, is_train=True):

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
        """
        训练模型
        :return:
        """
        user_feature_colums, item_feature_colums = self.get_fields()

        feature_columns = list(user_feature_colums) + list(item_feature_colums)

        train_dataset = self.get_dataset(feature_columns,is_train=True)
        test_dataset = self.get_dataset(feature_columns,is_train=False)

        tower_model, user_model, item_model = Tower(user_feature_colums, item_feature_colums)

        tower_model.summary()

        tower_model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'],
                            experimental_run_tf_function=False)

        checkpoint = ModelCheckpoint(self.model_path, monitor='val_accuracy', save_best_only=True,
                                     save_weights_only=True)

        tower_model.fit(train_dataset, epochs=self.epochs, steps_per_epoch=13358,
                        validation_data=test_dataset,
                        validation_steps=489,
                        callbacks=[checkpoint]
                        )


if __name__ == '__main__':
    TowerTask().train()
