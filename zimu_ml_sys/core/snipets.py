import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K


def combined_dnn_input(embedding_outputs, dense_outputs):
    output = []
    if isinstance(embedding_outputs, tf.Tensor):
        output = Flatten()(embedding_outputs)

    if isinstance(dense_outputs, tf.Tensor):
        output = Concatenate()([output, dense_outputs])

    return output


def sequence_masking(x, mask, value=0.0, axis=None):
    """
    padding mask
    为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    value: mask部分要被替换成的值，可以是'-inf'或'inf'；
    axis: 序列所在轴，默认为1；
    """
    if mask is None:
        return x
    else:
        if K.dtype(mask) != K.dtype(x):
            mask = K.cast(mask, K.dtype(x))
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        if axis is None:
            axis = 1
        elif axis < 0:
            axis = K.ndim(x) + axis
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        return x * mask + value * (1 - mask)


def lm_masking(s):
    """
    language masking: 为下三角 masking, 表示当前点 只能看到之前的点，不能看到 未来的序列

    1. 可以用来 做 生成任务解码阶段，只能看到 前面的输入，不能看到未来信息
    2. 针对 推荐，用户点击序列，只能看到 用户之前点击的序列，不能看到用户历史序列
    :param x: 输入原始的 三维向量， 【N, seq, dim】
    :return: 仅仅返回 lm_mask,
    """
    seq_len = K.shape(s)[1]  # 获取句子最大长度
    idxs = K.arange(0, seq_len)
    mask = idxs[None, :] <= idxs[:, None]  # 这里形成下三角  None 表示添加新的一维
    mask = K.cast(mask, K.floatx())
    return - (1 - mask[None, None]) * 1e12  # mask[None, None] 表示新增第一、二列


def unlim_masking(s):
    """
    类如： 针对阅读理解  给定 篇章，生成 答案及问题 任务
    篇章 与 篇章 之间可以 相互attention
    在答案与问题 阶段 采用的是 seq2seq, 即当前字 只能 attention 它之前的字
    :param x: 输入原始的 三维向量， 【N, seq, dim】
    :return: 仅仅返回 unlim_mask,
    """
    idxs = K.cumsum(s, axis=1)
    mask = idxs[:, None, :] <= idxs[:, :, None]
    mask = K.cast(mask, K.floatx())
    return - (1 - mask[:, None]) * 1e12
