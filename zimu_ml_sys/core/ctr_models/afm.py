from zimu_ml_sys.core.layers import AFMLayer
from tensorflow.keras.models import Model
from zimu_ml_sys.core.feature_columns import *
from tensorflow.keras.layers import *


def AFM(feature_columns):
    """
    AFM 模型  attention + FM
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
