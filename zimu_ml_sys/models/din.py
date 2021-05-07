from .feature_column import *
from .layer import AttentionSequencePoolingLayer, DNNLayer
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model


def seq_attention_pooling_output(feature_columns, embedding_outputs_dict, att_hidden_units, is_gru):
    """
    针对历史浏览记录 与 当前的item， 求 当前item 对 历史行为 的关注度，输出 历史序列 加权向量。
    添加了padding mask 处理
    :param feature_columns:
    :param embedding_outputs_dict:
    :param att_hidden_units:
    :return:
    """
    hist_feature_columns = [feature_column for feature_column in feature_columns if
                            not feature_column.is_hist_mean_pooling]
    keys_emb_list = [embedding_outputs_dict.get(feature_column.name) for feature_column in hist_feature_columns]

    querys_emb_list = [embedding_outputs_dict.get(feature_column.embedding_name) for feature_column in
                       hist_feature_columns]
    querys_emb = Concatenate()(querys_emb_list)
    keys_emb = Concatenate()(keys_emb_list)

    if is_gru:
        embedding_size = 64
        for feature_column in hist_feature_columns:
            embedding_size = feature_column.embedding_dim
            break

        keys_emb = GRU(embedding_size * len(hist_feature_columns), return_sequences=True)(keys_emb)

    att_output = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_units)([querys_emb, keys_emb])

    att_output = Flatten()(att_output)

    return att_output


"""
339s 25ms/step - loss: 0.0802 - accuracy: 0.9712 - val_loss: 0.3022 - val_accuracy: 0.9042
"""


def DIN(feature_columns, att_hidden_units=(80, 40), hidden_units=(128, 64),
        activation='relu', output_activation='', is_gru=False):
    """
    din 模型

    is_gru: 表示历史记录 使用GRU 提取用户兴趣层，这里并没有实现 Dien 用户兴趣演变，只是简单针对用户历史记录 做GRU 提取

    """
    feature_input_layers = build_input_layers(feature_columns)

    inputs_list = feature_input_layers.values()

    embedding_outputs_dict, dense_outputs_dict = build_embedding_outputs(feature_input_layers, feature_columns,
                                                                         is_concat=False)

    """seq attention """

    att_output = seq_attention_pooling_output(feature_columns, embedding_outputs_dict, att_hidden_units, is_gru)

    deep_feature_columns = [feature_column for feature_column in feature_columns if feature_column.is_hist_mean_pooling]
    sparse_embedding_layers = [embedding_outputs_dict.get(feature_column.name) for feature_column in
                               deep_feature_columns if isinstance(feature_column, SparseFeat)]

    """dnn 端"""
    sparse_output = Flatten()(Concatenate()(sparse_embedding_layers))

    if len(dense_outputs_dict) > 0:
        dense_output = Concatenate()(dense_outputs_dict.values())
        dnn_input = Concatenate()([att_output, sparse_output, dense_output])
    else:
        dnn_input = Concatenate()([att_output, sparse_output])

    dnn_output = DNNLayer(dnn_hidden_units=hidden_units,
                          activation=activation,
                          output_activation=output_activation)(dnn_input)

    output = Dense(1, activation='sigmoid')(dnn_output)

    model = Model(inputs=inputs_list, outputs=output)

    return model
