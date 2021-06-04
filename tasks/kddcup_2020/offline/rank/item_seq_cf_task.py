# -*- coding: utf-8 -*-
from zimu_ml_sys.utils.data_util import read_encoding_mapping_data
from tasks.kddcup_2020.constant import TASK_NAME
from zimu_ml_sys.core.feature_columns import SparseFeat, VarLenSparseFeat, ConditionFeat
from tasks.kddcup_2020.offline.preprocess.preprocess_data import PreprocessData
from zimu_ml_sys.core.ctr_models.item_seq_cf import ItemSeqCF

CTR_FIELDS = ['user_id', 'history_items']

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


class ItemSeqCFTask(object):

    def __init__(self):
        print()

    embedding_size = 256
    maxlen = 50
    neg_sample_num = 10

    def get_feature_columns(self):
        sparse_features = [field for field in CTR_FIELDS if not field.startswith('history')]

        encoding_fields_mapping = read_encoding_mapping_data(TASK_NAME)

        user_feature_colums = [SparseFeat(name=field,
                                          vocabulary_size=len(encoding_fields_mapping[field][0]) + 2,
                                          embedding_dim=self.embedding_size) for field in sparse_features]

        item_feature_columns = [VarLenSparseFeat(SparseFeat(name='history_items',
                                                            vocabulary_size=len(
                                                                encoding_fields_mapping['item_id'][0]) + 2,
                                                            embedding_dim=self.embedding_size,
                                                            embedding_name='item_id'),
                                                 maxlen=self.maxlen,
                                                 is_hist_mean_pooling=False)
                                ]

        neg_item_columns = [ConditionFeat(name='neg_items', dimension=(self.maxlen, self.neg_sample_num),embedding_name='item_id')]

        return user_feature_colums, item_feature_columns, neg_item_columns

    def train(self):
        user_feature_colums, item_feature_columns,neg_item_columns = self.get_feature_columns()

        # data_df = PreprocessData().generate_rank_data_frame()

        ItemSeqCF(user_feature_colums, item_feature_columns,neg_item_columns, att_embedding_size=self.embedding_size)


if __name__ == '__main__':
    ItemSeqCFTask().train()
