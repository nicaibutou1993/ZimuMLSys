from .feature_column import *
from .layer import MutiHeadSelfAttention, DNNLayer
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model
from .snipets import combined_dnn_input


"""
autoint模型：添加了 mutiheadselfattention 效果没有 FM DeepFM 好
30ms/step - loss: 0.0462 - accuracy: 0.9835 - val_loss: 0.3911 - val_accuracy: 0.8885
Total params: 4,907,990
"""

def AutoInt(feature_columns, hidden_units=(128, 128),
            activation='relu', output_activation='', att_layer_num=3,
            att_embedding_size=16, head_num=2, use_res=True):
    """
    AutoInt 模型
    """
    feature_input_layers = build_input_layers(feature_columns)

    inputs_list = feature_input_layers.values()

    linear_logit = get_linear_logit(feature_input_layers, feature_columns)

    embedding_outputs, dense_outputs = build_embedding_outputs(feature_input_layers, feature_columns)

    att_input = embedding_outputs
    for i in range(att_layer_num):
        att_input = MutiHeadSelfAttention(att_embedding_size, head_num, use_res=use_res)(att_input)

    att_output = Flatten()(att_input)

    dnn_input = combined_dnn_input(embedding_outputs, dense_outputs)

    dnn_output = DNNLayer(dnn_hidden_units=hidden_units,
                          activation=activation,
                          output_activation=output_activation)(dnn_input)

    stack_input = Concatenate(axis=-1)([att_output, dnn_output])

    final_logit = Dense(1, use_bias=False)(stack_input)

    logit = Add()([linear_logit, final_logit])

    output = Dense(1, activation='sigmoid')(logit)

    model = Model(inputs=inputs_list, outputs=output)

    return model
