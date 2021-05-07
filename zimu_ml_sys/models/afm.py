
from .feature_column import *
from .layer import AFMLayer
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model


def AFM(feature_columns):


    """
    AFM 模型
    """
    feature_input_layers = build_input_layers(feature_columns)

    inputs_list = feature_input_layers.values()

    linear_logit = get_linear_logit(feature_input_layers, feature_columns)

    embedding_outputs, dense_outputs = build_embedding_outputs(feature_input_layers, feature_columns)

    afm_logit = AFMLayer()(embedding_outputs)

    logit = Add()([linear_logit, afm_logit])

    output = Dense(1, activation='sigmoid')(logit)

    model = Model(inputs=inputs_list, outputs=output)

    return model
