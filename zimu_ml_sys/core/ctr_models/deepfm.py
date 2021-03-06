from zimu_ml_sys.core.layers import FMLayer, DNNLayer
from tensorflow.keras.models import Model
from zimu_ml_sys.core.snipets import combined_dnn_input
from zimu_ml_sys.core.feature_columns import *
from tensorflow.keras.layers import *

def DeepFM(feature_columns, hidden_units=(128, 128,), activation='relu', output_activation=''):
    """
    DEEPFM 模型
    """
    feature_input_layers = build_input_layers(feature_columns)

    inputs_list = feature_input_layers.values()

    linear_logit = get_linear_logit(feature_input_layers, feature_columns)

    embedding_outputs, dense_outputs = build_embedding_outputs(feature_input_layers, feature_columns)

    """FM 端"""
    fm_logit = FMLayer()(embedding_outputs)

    """deep 端"""

    dnn_input = combined_dnn_input(embedding_outputs, dense_outputs)

    dnn_output = DNNLayer(dnn_hidden_units=hidden_units,
                          activation=activation,
                          output_activation=output_activation)(dnn_input)

    dnn_logit = Dense(1, use_bias=False)(dnn_output)

    logit = Add()([linear_logit, fm_logit, dnn_logit])

    output = Dense(1, activation='sigmoid')(logit)

    model = Model(inputs=inputs_list, outputs=output)

    return model
