from zimu_ml_sys.core.layers import FMLayer
from tensorflow.keras.models import Model
from zimu_ml_sys.core.feature_columns import *
from tensorflow.keras.layers import *

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
