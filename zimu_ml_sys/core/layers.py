import pandas as pd
from tensorflow.keras.layers import *
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import glorot_normal, Zeros
import itertools
from zimu_ml_sys.core.snipets import sequence_masking, lm_masking, unlim_masking
from tensorflow.keras import initializers


class Layer(Layer):
    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.supports_masking = True


class FMLayer(Layer):
    """
    FM：完成的是特征2阶交叉
    FM: loss function 实现过程 0.5 * sum(和的平方 - 平方的和)

    实现 特征两两 交叉，方式直接相乘，其每一个输出的权重都是1，而AFM 中，加入了attention，使得所有交叉特征权重总和为1
    """

    def __init__(self, **kwargs):
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        square_of_sum = K.square(K.sum(inputs, axis=1, keepdims=True))

        sum_of_square = K.sum(inputs * inputs, axis=1, keepdims=True)

        cross_term = square_of_sum - sum_of_square

        logit = 0.5 * K.sum(cross_term, axis=2, keepdims=False)

        return logit

    def compute_output_shape(self, input_shape):
        return (None, 1)


class DNNLayer(Layer):
    """
    全连接层：
    dnn_hidden_units:(128,128,1)
    output_activation:sigmoid,softmax,None,relu

    dnn_hidden_units: 为 None： 表示同 输入的 x最后一维，是相同维度, repeat_num 为次数

    """

    def __init__(self, dnn_hidden_units=None, activation='relu', output_activation='', repeat_num=1, **kwargs):
        """

        :param dnn_hidden_units: 默认 None，即 输入值 （N,10,128）, 那么 权重为 （128,128）  与 repeat_num 共用
        :param activation: 设置 统一激活层
        :param output_activation: 设置 最后一层 激活层 默认同其 activation 激活一致，如果 output_activation 设置 None 表示没有激活
        :param repeat_num: 同dnn_hidden_units 一起使用
        :param kwargs:
        """
        self.dnn_hidden_units = dnn_hidden_units
        self.activation = activation
        self.output_activation = output_activation
        self.repeat_num = repeat_num

        super(DNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.dnn_hidden_units is None:
            """与上一轮输出值，保持 相同的dim"""
            unit = input_shape[-1]
            hidden_units = list([unit for _ in range(self.repeat_num)])
        else:
            """自定义输入 """
            hidden_units = list(self.dnn_hidden_units)

        last_unit = hidden_units.pop(-1)
        if self.output_activation == '':
            self.output_activation = self.activation

        self.denses = []
        for unit in hidden_units:
            dense = Dense(unit, activation=self.activation, )
            self.denses.append(dense)

        dense = Dense(last_unit, activation=self.output_activation, )
        self.denses.append(dense)

        super(DNNLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        x = inputs
        for dense_layer in self.denses:
            x = dense_layer(x)

        output = x
        return output

    def compute_output_shape(self, input_shape):

        return (-1, list(self.dnn_hidden_units)[-1])

    def get_config(self):
        config = {'dnn_hidden_units': self.dnn_hidden_units,
                  'activation': self.activation,
                  'output_activation': self.output_activation,
                  }
        base_config = super(DNNLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class CrossLayer(Layer):
    """
    用于 deep & cross， cross 特征交叉 使用的是 多项式方式、
    公式 ：x(l+1) = x0 * (xl * wl) + xl
    deep cross 可以提供特征 3阶 4阶 特征交叉，而 FM等其他都是 特征2阶交叉
    """

    def __init__(self, cross_num=2, **kwargs):
        self.cross_num = cross_num

        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.kernels = [self.add_weight(shape=(dim, 1), name='kernel_' + str(i),
                                        initializer=glorot_normal(),
                                        trainable=True,
                                        ) for i in range(self.cross_num)]

        self.biases = [self.add_weight(shape=(dim, 1), name='bias' + str(i),
                                       initializer=Zeros(),
                                       trainable=True,
                                       ) for i in range(self.cross_num)]

        super(CrossLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        x = K.expand_dims(x, 2)

        x0 = x
        xl = x

        for i in range(self.cross_num):
            _xl = tf.tensordot(xl, self.kernels[i], axes=(1, 0))
            _xl = tf.matmul(x0, _xl)
            xl = _xl + self.biases[i] + xl
        return tf.squeeze(xl, 2)

    def get_config(self):
        config = {'cross_num': self.cross_num,
                  }
        base_config = super(CrossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AFMLayer(Layer):
    """
    AFM ,attention FM ,在特征与特征两两组合 交叉，接入dense，然后计算 softmax 权重，
    """

    def __init__(self, attention_factor=32, **kwargs):
        self.attention_factor = attention_factor

        super(AFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense1 = Dense(self.attention_factor, activation='relu', )
        self.dense2 = Dense(1, use_bias=False, )
        self.dense3 = Dense(1, use_bias=False, )

        super(AFMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        split_num = K.int_shape(x)[1]
        x_list = tf.split(x, split_num, axis=1)

        row = []
        col = []
        for r, c in itertools.combinations(x_list, 2):
            row.append(r)
            col.append(c)

        p = K.concatenate(row, axis=1)
        q = K.concatenate(col, axis=1)
        inner_product = p * q

        x = self.dense1(inner_product)
        x = self.dense2(x)
        a = K.softmax(x, axis=1)

        o = K.sum(a * inner_product, axis=1)

        afm_logit = self.dense3(o)

        return afm_logit

    def get_config(self):
        config = {"attention_factor": self.attention_factor,
                  }

        base_config = super(AFMLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (None, 1)


class MutiHeadSelfAttention(Layer):
    """多头self attention"""

    def __init__(self, head_num,
                 att_embedding_size=None,
                 use_padding_masking=True,
                 use_lm_masking=False,
                 use_unlim_masking=False,
                 use_res=True,
                 **kwargs):
        """

        :param head_num: 多头
        :param att_embedding_size: head_size
        :param use_padding_masking: 默认开启：是否 针对padding 0 进行mask
        :param use_lm_masking: 默认关闭：是否启用 下三角形 masking, 针对 未来信息 不可见，1. 可用于生成任务解码阶段，2. 推荐，用户点击序列
        :param use_res: 默认开启： 是否使用残差设计，针对原始输入
        :param kwargs:
        """
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.use_padding_masking = use_padding_masking
        self.use_lm_masking = use_lm_masking
        self.use_unlim_masking = use_unlim_masking
        super(MutiHeadSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.att_embedding_size is None:
            out_dim = input_shape[-1]
            self.att_embedding_size = out_dim // self.head_num
        else:
            self.att_embedding_size = self.att_embedding_size // self.head_num

        self.q_dense = Dense(self.att_embedding_size * self.head_num)
        self.k_dense = Dense(self.att_embedding_size * self.head_num)
        self.v_dense = Dense(self.att_embedding_size * self.head_num)

        if self.use_res:
            self.res_dense = Dense(self.att_embedding_size * self.head_num)

        super(MutiHeadSelfAttention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):

        x = inputs
        qw = self.q_dense(x)
        kw = self.k_dense(x)
        vw = self.v_dense(x)

        seq_len = K.int_shape(qw)[1]
        qw = K.reshape(qw, (-1, seq_len, self.head_num, self.att_embedding_size))
        kw = K.reshape(kw, (-1, seq_len, self.head_num, self.att_embedding_size))
        vw = K.reshape(vw, (-1, seq_len, self.head_num, self.att_embedding_size))

        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))

        a = tf.matmul(qw, kw, transpose_b=True)

        a = a / (K.int_shape(kw)[-1] ** 0.5)

        """padding mask"""

        if self.use_padding_masking:
            a = sequence_masking(a, mask, '-inf', -1)

        """下三角 masking ,未来信息 隔绝"""
        if self.use_lm_masking:
            a += lm_masking(x)

        """现有数据结构内部 attention， 未来 信息为 下三角形 """
        if self.use_unlim_masking:
            a += unlim_masking(x)

        a = K.softmax(a, axis=-1)

        o = tf.matmul(a, vw)

        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, seq_len, self.head_num * self.att_embedding_size))

        """再次 padding mask， 如果出现 lm_masking 与 padding mask 同时使用的话，需要在次mask,确保 不会泄密"""
        if self.use_padding_masking:
            o = sequence_masking(o, mask, value=0)

        if self.use_res:
            o += self.res_dense(x)
        return o

    def compute_output_shape(self, input_shape):
        return (-1, input_shape[1], self.head_num * self.att_embedding_size)

    def get_config(self):
        config = {'att_embedding_size': self.att_embedding_size,
                  'head_num': self.head_num,
                  'use_padding_masking': self.use_padding_masking,
                  'use_lm_masking': self.use_lm_masking,
                  'use_unlim_masking': self.use_unlim_masking,
                  'use_res': self.use_res
                  }
        base_config = super(MutiHeadSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionSequencePoolingLayer(Layer):
    """
    针对 query 与  key 做减乘操作
    concat([ query, key, query -  key, query * key ],axis = -1)    [N , 4, 12 * 4]
    经过 三层 dense  （【80,40,1】） 最终输出是  【N,4,1】
    对应当前商品ID与当前广告ID 针对 历史商品集合 和 历史广告集合 的关注程度，
    """

    def __init__(self, att_hidden_units=(80, 40), **kwargs):
        self.att_hidden_units = att_hidden_units

        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):

        self.denses = []
        for unit in list(self.att_hidden_units):
            self.denses.append(Dense(unit, activation='relu', ))

        last_dense = Dense(1, )
        self.denses.append(last_dense)

        super(AttentionSequencePoolingLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):

        k_mask = mask[1]

        q, k = inputs

        seq_len = K.int_shape(k)[1]
        q = K.repeat_elements(q, seq_len, axis=1)

        a = K.concatenate([q, k, q - k, q * k], axis=-1)

        for dense_layer in self.denses:
            a = dense_layer(a)

        """针对padding，填充 -inf """
        a = sequence_masking(a, k_mask, value='-inf')

        a = K.softmax(a, axis=1)

        o = tf.matmul(a, k, transpose_a=True)

        return o

    def compute_output_shape(self, input_shape):

        return (-1, 1, input_shape[0][-1])

    def get_config(self):
        config = {'att_hidden_units': self.att_hidden_units,
                  }
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SequencePoolingLayer(Layer):
    '''

    mask:
     【【1,1,1,0,0,0,0,0】
    【0,0,0,0,0,0,0,0】】
    这里针对  一条点击记录都没有,
    首先 从Embedding 获取 0 代表的向量，其实我们这里针对用户如果一次都没有点击，
    也能 做相关的推荐，这里其实就是学习 0 所代表的向量，
    针对 mean_pooling 的时候，如果有行为的记录，我们只需要将行为记录对应的Embedding 相加求平均，而 去掉mask 为0的。
    如果针对 没有行为记录的，我们 需要将mask 全部转为 1，这样可以学习 Embedding 为0 所在的向量特征。
    '''

    def __init__(self, mode='mean', **kwargs):
        self.mode = mode

        super(SequencePoolingLayer, self).__init__(**kwargs)

        self.supports_masking = True

    def build(self, input_shape):
        super(SequencePoolingLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        mask = K.cast(mask, dtype=tf.float32)

        seq_num = K.int_shape(mask)[-1]

        user_behavior_length = K.sum(mask, axis=-1, keepdims=True)

        greater = K.greater(user_behavior_length, 0)

        seq_length = tf.where(greater, user_behavior_length, K.ones_like(user_behavior_length) * seq_num)

        final_mask = tf.where(greater, mask, K.ones_like(mask))

        final_mask = K.expand_dims(final_mask, -1)

        new_inputs = inputs * final_mask
        final_inputs = K.sum(new_inputs, axis=1) / seq_length

        final_inputs = K.expand_dims(final_inputs, 1)

        return final_inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[-1])

    def get_config(self):
        config = {'mode': self.mode}
        base_config = super(SequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Hash(Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):

        if x.dtype != tf.string:
            zero = tf.as_string(tf.zeros([1], dtype=x.dtype))
            x = tf.as_string(x, )
        else:
            zero = tf.as_string(tf.zeros([1], dtype='int32'))

        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, num_buckets,
                                                   name=None)  # weak hash
        except:

            hash_x = tf.strings.to_hash_bucket(x, num_buckets,
                                               name=None)  # weak hash
        if self.mask_zero:
            mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            hash_x = (hash_x + 1) * mask

        return hash_x

    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, }
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionEmbedding(Layer):
    """
    针对 序列特征，如果需要引入 位置特征
    """

    def __init__(self, input_dim, merge_mode='add',
                 embeddings_initializer='zeros', **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.merge_mode = merge_mode
        self.embeddings_initializer = initializers.get(embeddings_initializer)

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.embeddings = self.add_weight(name='embeddings',
                                          shape=(self.input_dim, output_dim),
                                          initializer=self.embeddings_initializer)

        super(PositionEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_shape = K.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        embeddings = self.embeddings[None, :seq_len]
        return inputs + embeddings

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'merge_mode': self.merge_mode,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer)
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Loss(Layer):
    """特殊的层，用来定义复杂loss
    """

    def __init__(self, output_axis=None, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self.output_axis = output_axis

    def build(self, input_shape):
        super(Loss, self).build(input_shape)

    def call(self, inputs, mask=None):
        loss = self.compute_loss(inputs, mask)
        self.add_loss(loss, inputs=inputs)
        if self.output_axis is None:
            return inputs
        elif isinstance(self.output_axis, list):
            return [inputs[i] for i in self.output_axis]
        else:
            return inputs[self.output_axis]

    def compute_loss(self, inputs, mask=None):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        if self.output_axis is None:
            return input_shape
        elif isinstance(self.output_axis, list):
            return [input_shape[i] for i in self.output_axis]
        else:
            return input_shape[self.output_axis]

    def compute_mask(self, inputs, mask):
        if mask is not None:
            if self.output_axis is None:
                return mask
            elif isinstance(self.output_axis, list):
                return [mask[i] for i in self.output_axis]
            else:
                return mask[self.output_axis]

    def get_config(self):
        config = {
            'output_axis': self.output_axis,
        }
        base_config = super(Loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    # csv = pd.read_csv('E:\data/avazu/mini_train.csv')
    #
    # values = list(set(csv['device_id'].values))
    #
    # num = 10000
    # k = tf.constant([values])
    #
    # hash_value = Hash(20000, mask_zero=False)(k)
    # data = pd.DataFrame(list(hash_value.numpy()[0])).value_counts()
    #
    # print(data)
    # print(len(values), len(data))

    data = tf.constant([[[0, 0, 0, 0, 0, 5, 4, 1, 2, 3], [0, 0, 0, 0, 0, 0, 7, 1, 2, 3]],
                        [[0, 0, 0, 0, 0, 0, 0, 1, 2, 3], [0, 0, 0, 0, 0, 0, 0, 0, 2, 3]]], dtype=tf.float32)

    data2 = tf.constant([[[0, 0, 0, 0, 0, 5, 4, 1, 2, 3], [0, 0, 0, 0, 0, 0, 7, 1, 2, 3]],
                         [[0, 0, 0, 0, 0, 0, 0, 1, 2, 3], [0, 0, 0, 0, 0, 0, 0, 0, 2, 3]]], dtype=tf.float32)

    pos_logits = Lambda(lambda x: K.batch_dot(x[0], x[1]))([data, data2])

    x = Embedding(8, 16, mask_zero=True)(data)

    output = MutiHeadSelfAttention(head_num=2, use_lm_masking=True)(x)

    print(output)

    # x = Input(shape=(6,), dtype='int32')
