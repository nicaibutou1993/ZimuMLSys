from zimu_ml_sys.core.layers import DNNLayer, CrossLayer
from tensorflow.keras.models import Model
from zimu_ml_sys.core.snipets import combined_dnn_input
from zimu_ml_sys.core.feature_columns import *
from tensorflow.keras.layers import *


def DCN(feature_columns, hidden_units=(128, 128), activation='relu', output_activation=''):
    """
        deep & cross
    :param feature_columns:
    :param hidden_units:
    :param activation:
    :param output_activation:
    :return:
    """
    feature_input_layers = build_input_layers(feature_columns)

    inputs_list = feature_input_layers.values()

    """线性端"""
    linear_logit = get_linear_logit(feature_input_layers, feature_columns)

    embedding_outputs, dense_outputs = build_embedding_outputs(feature_input_layers, feature_columns)

    dnn_input = combined_dnn_input(embedding_outputs, dense_outputs)

    """deep 端"""
    dnn_output = DNNLayer(dnn_hidden_units=hidden_units,
                          activation=activation,
                          output_activation=output_activation)(dnn_input)

    '''特征交叉层，多项式交叉，cross_num=2 表示特征与特征 多项式 3次方'''
    cross_output = CrossLayer(cross_num=2)(dnn_input)

    """合并 deep 端 与 特征交叉端 """
    cross_output = Concatenate(axis=-1)([dnn_output, cross_output])

    cross_logit = Dense(1, use_bias=False)(cross_output)

    logit = Add()([linear_logit, cross_logit])

    output = Dense(1, activation='sigmoid')(logit)

    model = Model(inputs=inputs_list, outputs=output)

    return model
