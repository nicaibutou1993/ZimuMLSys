import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from collections import OrderedDict
import copy
from tensorflow.keras.initializers import GlorotUniform, Zeros
from zimu_ml_sys.core.layers import SequencePoolingLayer
from zimu_ml_sys.core.layers import Hash
from zimu_ml_sys.core.layers import PositionEmbedding


class SparseFeat:
    """
    针对类别特征
    """

    def __init__(self, name, vocabulary_size, embedding_dim=4, dtype="int64",
                 embedding_name=None,
                 embeddings_initializer=None,
                 is_hist_mean_pooling=True,
                 use_hash=False
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

        self.use_hash = use_hash


class DenseFeat:
    """
    针对连续变量
    """

    def __init__(self, name, dimension, dtype='float32'):
        self.name = name
        self.dimension = dimension
        self.dtype = dtype


class ConditionFeat:
    """外部条件输入模型，比如true_label 输入模型"""

    def __init__(self, name,
                 dimension=(1,),
                 embedding_name=None,
                 is_pred=False,
                 dtype='int64'):
        """
        :param name:
        :param dimension: 维度 (50,20)
        :param embedding_name: 默认 None，表示不需要进行 Embedding操作。 指定Embedding名称
        :param is_pred: 如果设置为 true,针对额外输入，输入的是否 参与 预测阶段，比如输入一些items,用于 预测这些items 概率
        :param dtype:
        """
        self.name = name
        self.dimension = dimension
        self.dtype = dtype
        self.embedding_name = embedding_name
        self.is_pred = is_pred


class VarLenSparseFeat:
    """
    针对seq 序列变量

    is_hist_pooling: 针对 历史 相关特征  在进行 Embedding，是否进行 pooling (mean)
    is_position: 针对 历史序列特征，是否 考虑位置信息
    """

    def __init__(self, sparsefeat, maxlen, is_hist_mean_pooling=True, is_position=False):
        self.sparsefeat = sparsefeat
        self.maxlen = maxlen

        self.is_hist_mean_pooling = is_hist_mean_pooling
        self.is_position = is_position

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

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash


def build_input_layers(feature_columns):
    """
    创建 模型输入
    :param feature_columns: 字段
    :return: 输入层 input_x
    """
    features_input_layers = OrderedDict()

    for fc in feature_columns:

        if isinstance(fc, SparseFeat):
            features_input_layers[fc.name] = Input(name=fc.name, shape=(1,), dtype=fc.dtype)

        elif isinstance(fc, VarLenSparseFeat):

            features_input_layers[fc.name] = Input(name=fc.name, shape=(fc.maxlen,), dtype=fc.dtype)

        elif isinstance(fc, DenseFeat):
            features_input_layers[fc.name] = Input(name=fc.name, shape=(1,), dtype=fc.dtype)

        elif isinstance(fc, ConditionFeat):
            features_input_layers[fc.name] = Input(name=fc.name, shape=fc.dimension, dtype=fc.dtype)

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
            4. 提供 针对输入是字符串类别变量，进行Hash 分桶 （可选）
    :param input_layers: 输入层
    :param feature_columns:  字段
    :param mask_zero: 针对 Hash 或者 mean pooling 操作， 针对输入是0 相当于是 padding 部分
    :param is_concat: 是否将所有的Embedding 全部进行 concat 操作
    :param prefix: 针对多次使用相同 field Embedding 使用 prefix 加以区分，否则会因为 层的名称相同 导致创建模型识别
    :return: embedding dict 和 dense dict 可能是 dict 也可能是 tensor
    """
    embedding_layers = OrderedDict()
    sparse_embedding_outputs_dict = OrderedDict()
    var_embedding_outputs_dict = OrderedDict()
    dense_outputs_dict = OrderedDict()
    condition_outputs_dict = OrderedDict()

    sparse_feature_columns = {feature_column.name: feature_column for feature_column in feature_columns if
                              isinstance(feature_column, SparseFeat)}
    var_feature_columns = {feature_column.name: feature_column for feature_column in feature_columns if
                           isinstance(feature_column, VarLenSparseFeat)}
    dense_feature_columns = {feature_column.name: feature_column for feature_column in feature_columns if
                             isinstance(feature_column, DenseFeat)}
    condition_feature_columns = {feature_column.name: feature_column for feature_column in feature_columns if
                                 isinstance(feature_column, ConditionFeat)}

    """create embedding layer"""
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

        # embedding = Embedding(input_dim=feature_column.vocabulary_size, output_dim=feature_column.embedding_dim,
        #                       embeddings_initializer=feature_column.embeddings_initializer, mask_zero=mask_zero,
        #                       name=prefix + 'var' + '_emb_' + name)

    """set dense output"""
    for name, feature_column in dense_feature_columns.items():
        dense_outputs_dict[feature_column.name] = input_layers[name]

    """set condition output"""
    for name, feature_column in condition_feature_columns.items():
        embedding_name = feature_column.embedding_name
        input_idx = input_layers[name]

        if embedding_name:
            embedding_layer = embedding_layers[embedding_name]
            embedding_output = embedding_layer(input_idx)
            condition_outputs_dict[feature_column.name] = embedding_output
        else:
            condition_outputs_dict[feature_column.name] = input_idx

    """set sparse output"""
    for name, feature_column in sparse_feature_columns.items():
        embedding_layer = embedding_layers[name]

        input_idx = input_layers[name]
        use_hash = feature_column.use_hash
        if use_hash:
            input_idx = Hash(feature_column.vocabulary_size)(input_idx)

        sparse_embedding_outputs_dict[name] = embedding_layer(input_idx)

    '''针对 序列变量 进行 Embedding，针对输入是 字符串。
        1. hash 分桶（可选），
        2. 针对序列 进行平均池化 （可选）
        3. 是否添加 位置 信息（可选）
        '''
    for name, feature_column in var_feature_columns.items():
        embedding_name = feature_column.embedding_name

        embedding_layer = embedding_layers[embedding_name]

        input_idx = input_layers[name]

        use_hash = feature_column.use_hash
        if use_hash:
            input_idx = Hash(feature_column.vocabulary_size)(input_idx)

        embedding_output = embedding_layer(input_idx)

        if feature_column.is_position:
            embedding_output = PositionEmbedding(feature_column.maxlen)(embedding_output)

        if feature_column.is_hist_mean_pooling:
            pooling_output = SequencePoolingLayer(mode='mean')(embedding_output)
        else:
            pooling_output = embedding_output

        var_embedding_outputs_dict[name] = pooling_output

    sparse_embedding_outputs_dict.update(var_embedding_outputs_dict)

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

    if len(condition_outputs_dict) > 0:
        return embedding_outputs, dense_outputs, condition_outputs_dict
    else:
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
