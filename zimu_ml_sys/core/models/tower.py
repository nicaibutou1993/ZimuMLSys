import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from zimu_ml_sys.core.feature_columns import *
from zimu_ml_sys.core.layers import DNNLayer
from tensorflow.keras.layers import *


def Tower(user_columns, item_columns, dnn_hidden_units=(128,)):
    '''
    双塔模型：
        最终向量训练：
        必须要进行归一化处理：1,2 选择一个
            1. 那么在训练最后 进行sigmoid 处理训练
            2. 在向量output，进行 归一化处理，l2_normalize   即 对应元素 除以 根号下 所有元素平方和

    '''
    feature_columns = list(user_columns) + list(item_columns)

    input_layers_dict = build_input_layers(feature_columns)

    user_input_layers = [input_layers_dict[feature_column.name] for feature_column in user_columns]
    item_input_layers = [input_layers_dict[feature_column.name] for feature_column in item_columns]
    input_layers = input_layers_dict.values()

    '''返回Embedding 向量'''
    embedding_outputs_dict, dense_outputs_dict = build_embedding_outputs(input_layers_dict, feature_columns,
                                                                         is_concat=False)

    '''用户塔'''
    user_output = get_tower_output_layer(dense_outputs_dict, embedding_outputs_dict, user_columns, dnn_hidden_units)

    '''item塔'''
    item_output = get_tower_output_layer(dense_outputs_dict, embedding_outputs_dict, item_columns, dnn_hidden_units)

    output = Dot(axes=1)([user_output, item_output])

    user_model = Model(inputs=user_input_layers, outputs=user_output)
    item_model = Model(inputs=item_input_layers, outputs=item_output)

    tower_model = Model(inputs=input_layers, outputs=output)

    return tower_model, user_model, item_model


def get_tower_output_layer(dense_outputs_dict, embedding_outputs_dict, columns, dnn_hidden_units):
    '''
    单侧塔
        1. 合并
        2. dense
        3. 归一化
    :param dense_outputs_dict:
    :param embedding_outputs_dict:
    :param columns:
    :param dnn_hidden_units:
    :return:
    '''
    dense_outputs = [dense_outputs_dict[feature_column.name] for feature_column in columns if
                     feature_column.name in dense_outputs_dict]
    embedding_outputs = [embedding_outputs_dict[feature_column.name] for feature_column in columns if
                         feature_column.name in embedding_outputs_dict]
    embedding_output = Lambda(lambda x: K.squeeze(K.concatenate(x, axis=-1), 1))(embedding_outputs)
    concat_output = Lambda(lambda x: K.concatenate(x, axis=-1))(list(dense_outputs) + list([embedding_output]))

    x = concat_output
    # for unit in dnn_hidden_units:
    #     x = Dense(unit, activation='relu')(x)

    x = DNNLayer(dnn_hidden_units=dnn_hidden_units)(x)

    """针对向量做归一化，方差为1, 然后 两向量相乘在 【-1,1】 之间 """
    output = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)

    return output


def get_item_tower_output_vectors(item_model, item_df, item_field_name=None, batch_size=512):
    """
    获取 item端  输出向量
    :param item_model: item 模型
    :param item_df: item 相关的dataframe
    :param item_field_name: 返回 item_ids 数据  默认None，表示第一列 作为 item_id
    :param batch_size:
    :return:
    """

    batch_item_output_vectors = []
    batch_item_ids = []

    if isinstance(item_df, pd.DataFrame):

        field_columns = item_df.columns
        if item_field_name is None:
            item_field_name = field_columns[0]

        dataset = tf.data.Dataset.from_tensor_slices(dict(item_df))
        dataset = dataset.batch(batch_size)

        for x in dataset:
            item_ids = x.get(item_field_name)
            x = [tf.reshape(x.get(field),(-1,1)).numpy().tolist() for field in field_columns]

            item_output_vectors = item_model(x, training=False)
            batch_item_output_vectors.extend(item_output_vectors.numpy().tolist())
            batch_item_ids.extend(item_ids.numpy())

    return batch_item_output_vectors,batch_item_ids
