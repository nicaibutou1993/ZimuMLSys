import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from collections import OrderedDict
import copy
from tensorflow.keras.initializers import GlorotUniform, Zeros
from .layer import SequencePoolingLayer


class SparseFeat:
    """
    针对类别特征
    """

    def __init__(self, name, vocabulary_size, embedding_dim=4, dtype="int32",
                 embedding_name=None,
                 embeddings_initializer=None,
                 is_hist_mean_pooling=True
                 ):
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        if embedding_name is None:
            embedding_name = name
        self.embedding_name = embedding_name

        if embeddings_initializer is None:
            embeddings_initializer = GlorotUniform(seed=2020)

        self.embeddings_initializer = embeddings_initializer

        self.is_hist_mean_pooling = is_hist_mean_pooling


class DenseFeat:
    """
    针对连续变量
    """

    def __init__(self, name, dimension, dtype='float32'):
        self.name = name
        self.dimension = dimension
        self.dtype = dtype


class VarLenSparseFeat:
    """
    针对seq 序列变量

    is_hist_pooling: 针对 历史 相关特征  在进行 Embedding，是否进行 pooling (mean)

    """

    def __init__(self, sparsefeat, maxlen, is_hist_mean_pooling=True):
        self.sparsefeat = sparsefeat
        self.maxlen = maxlen

        self.is_hist_mean_pooling = is_hist_mean_pooling

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def embeddings_initializer(self):
        return self.sparsefeat.embeddings_initializer


def build_input_layers(feature_columns):
    """
    创建 模型输入
    :param feature_columns:
    :return:
    """
    features_input_layers = OrderedDict()

    for fc in feature_columns:

        if isinstance(fc, SparseFeat):
            features_input_layers[fc.name] = Input(name=fc.name, shape=(1,), dtype=fc.dtype)

        elif isinstance(fc, VarLenSparseFeat):

            features_input_layers[fc.name] = Input(name=fc.name, shape=(fc.maxlen,), dtype=fc.dtype)

        elif isinstance(fc, DenseFeat):
            features_input_layers[fc.name] = Input(name=fc.name, shape=(1,), dtype=fc.dtype)

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return features_input_layers


def build_embedding_outputs(input_layers, feature_columns, mask_zero=True, prefix='', is_concat=True):
    """
    构建 类别变量 Embedding 合并输出， 及返回 Dense 特征输出

    类别变量：
            1. 类别变量
            2. 序列类别变量： 这里针对序列类别变量 使用了 mean 取平均方式
            3. 序列类别变量 与 类别变量 共用 Embedding
    :param input_layers:
    :param feature_columns:
    :param mask_zero:
    :return:
    """
    embedding_layers = OrderedDict()

    sparse_embedding_outputs_dict = OrderedDict()

    var_embedding_outputs_dict = OrderedDict()

    dense_outputs_dict = OrderedDict()

    sparse_feature_columns = {feature_column.name: feature_column for feature_column in feature_columns if
                              isinstance(feature_column, SparseFeat)}

    var_feature_columns = {feature_column.name: feature_column for feature_column in feature_columns if
                           isinstance(feature_column, VarLenSparseFeat)}

    dense_feature_columns = {feature_column.name: feature_column for feature_column in feature_columns if
                             isinstance(feature_column, DenseFeat)}

    for name, feature_column in dense_feature_columns:
        dense_outputs_dict[feature_column.name] = input_layers[name]

    for name, feature_column in sparse_feature_columns.items():
        embedding_layers[feature_column.embedding_name] = Embedding(input_dim=feature_column.vocabulary_size,
                                                                    output_dim=feature_column.embedding_dim,
                                                                    embeddings_initializer=feature_column.embeddings_initializer,
                                                                    name=prefix + 'sparse' + '_emb_' + feature_column.embedding_name
                                                                    )

    for name, feature_column in var_feature_columns.items():
        embedding_layers[feature_column.embedding_name] = Embedding(input_dim=feature_column.vocabulary_size,
                                                                    output_dim=feature_column.embedding_dim,
                                                                    embeddings_initializer=feature_column.embeddings_initializer,
                                                                    mask_zero=mask_zero,
                                                                    name=prefix + 'var' + '_emb_' + name
                                                                    )

    for name, feature_column in sparse_feature_columns.items():
        embedding_layer = embedding_layers[name]
        sparse_embedding_outputs_dict[name] = embedding_layer(input_layers[name])

    for name, feature_column in var_feature_columns.items():
        embedding_name = feature_column.embedding_name

        embedding_layer = embedding_layers[embedding_name]

        embedding_output = embedding_layer(input_layers[name])

        if feature_column.is_hist_mean_pooling:
            pooling_output = SequencePoolingLayer(mode='mean')(embedding_output)
        else:
            pooling_output = embedding_output

        var_embedding_outputs_dict[name] = pooling_output

    # sparse_embedding_outputs_dict.extend(var_embedding_outputs_dict)

    sparse_embedding_outputs_dict.update(var_embedding_outputs_dict)

    # concat_embedding_outputs = Concatenate(axis=1)(sparse_embedding_outputs_dict)

    # concat_embedding_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(sparse_embedding_outputs_dict)
    if is_concat:
        embedding_outputs_list = list(sparse_embedding_outputs_dict.values())
        embedding_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(embedding_outputs_list)

        dense_outputs_list = list(dense_outputs_dict.values())
        if len(dense_outputs_list) > 0:
            dense_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(dense_outputs_list)
        else:
            dense_outputs = []

    else:
        embedding_outputs = sparse_embedding_outputs_dict
        dense_outputs = dense_outputs_dict

    return embedding_outputs, dense_outputs


def get_linear_logit(feature_input_layers, feature_columns, prefix='linear_'):
    """
    线性端：
    1. 针对 类别变量
    2. 所有类别变量 Embedding 到 1维
    3. 求和 输出
    :param feature_input_layers:
    :param feature_columns:
    :return:
    """
    linear_feature_columns = copy.deepcopy(feature_columns)

    for i in range(len(linear_feature_columns)):
        feature_column = linear_feature_columns[i]
        if isinstance(feature_column, SparseFeat):
            feature_column.embedding_dim = 1
            feature_column.embedding_initializer = Zeros()

        if isinstance(feature_column, VarLenSparseFeat):
            feature_column.sparsefeat.embedding_dim = 1
            feature_column.sparsefeat.embedding_initializer = Zeros()

    concat_embedding_outputs, _ = build_embedding_outputs(feature_input_layers, linear_feature_columns, prefix=prefix)

    output = tf.squeeze(concat_embedding_outputs, axis=-1)

    linear_logit = tf.reduce_sum(output, axis=1, keepdims=True)

    return linear_logit
