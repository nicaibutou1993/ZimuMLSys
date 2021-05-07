from .feature_column import *
from .layer import FMLayer
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model


def FM(feature_columns):
    """
    FM 模型
    """
    feature_input_layers = build_input_layers(feature_columns)

    inputs_list = feature_input_layers.values()

    linear_logit = get_linear_logit(feature_input_layers, feature_columns)

    embedding_outputs, dense_outputs = build_embedding_outputs(feature_input_layers, feature_columns)

    fm_logit = FMLayer()(embedding_outputs)

    logit = Add()([linear_logit, fm_logit])

    output = Dense(1, activation='sigmoid')(logit)

    model = Model(inputs=inputs_list, outputs=output)

    return model
