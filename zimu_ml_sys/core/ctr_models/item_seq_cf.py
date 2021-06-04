# -*- coding: utf-8 -*-
from tensorflow.keras.models import Model
from zimu_ml_sys.core.feature_columns import *
from tensorflow.keras.layers import *
from zimu_ml_sys.core.layers import MutiHeadSelfAttention, DNNLayer, Loss
import tensorflow.keras.backend as K

"""
模型训练过程：

    1. 针对用户点击item序列：[N,50]  ==> seq_item_embedding [N,50,256]
    2. seq 位置Embedding 位置index(0-50) loc_embedding [N,50,256]
    3. x = seq_item_embedding + loc_embedding
    4. 针对 x 进行transformer (mutiheadattention + forward)
    5. user_embedding: [N,1,256]
    6. x = x + user_embedding    [N,50,256]
    7. 最终这里的x: 表示混合输出端：混合了 当前点击item信息、当前点击item 位置信息、历史用户点击item信息、用户id 信息
    8. x：[N,50,256] reshape: ==> [N * 50 , 256]
       正样本： true_embedding移动一位的 true_item_embedding [N * 50, 256]  与 之前 用户点击item 序列 错开一位 作为预测标签
       true_logits = dot(x , true_embedding) ==> [N * 50,1]  ==> 在 [0,1] 之间

       负样本： neg_embedding [N * 50, 20, 256]  针对每一个正样本 进行随机采样20个负样本
       neg_logits = dot(x , neg_embedding) ==> [N * 50,20,1]  ==> 在 [0,1] 之间

       logits = true_logits + neg_logits


    正样本：为下一次点击的item, 负样本为 随机采样 20个

    针对一条用户点击序列：
        【112,11,145,22,0,0,0,0,0....0】  ## 用户点击item 序列
        【 11,145,220,0,0,0,0....0】      ## 标签
         【【43,56,765,768，43...,6575】,  ## 随机采样的负样本 一个正样本对应20个负样本
            [...],
            [...],
            ...
            [...]
         】

    标签 过 item_embedding ==> [N,50,256]
    负样本 过 item_embedding ==> [N,50,20,256]

    seq_item_embedding: reshape ==> [N * 50,256]
    item_embedding: reshape ==>  [N * 50,256]
    dot( seq_item_embedding, item_embedding) == > [N * 50,1] 结果是否点击 在【0,1】 之间

    细说 混合输出端：
        在用户点击序列中：【112,11,145,22,0,0,0,0,0....0】
        预测标签为       【 11,145,220,0,0,0,0....0】

        针对最终点击序列中： 我们选择 145：最终会产生一个 output_vector:【256】向量 ：
                            这个向量包含的信息有哪些？
                            1. 含有该 145 item_embedding 信息
                            2. 含有 145 对应的位置信息：当前位置是 3
                            3. 使用了attention 还有 之前用户点击 112,11 两个item 信息，当前针对未来点击的做了mask处理
                            4. 还有了 用户的信息

    预测阶段：
        输入： 用户ID【1】，用户点击序列 【1,50】，及所有带推荐的items ：【2000】
        输出： 【1,50,2000】 用户每一个点击时间点，都有2000 推荐的概率值：取出 每一个序列最大值 【1,50】 作为推荐

    备注： 这里针对 item Embedding 参数： 是通过外部输入，item：通过文字及图片，通过nlp 及cv 参数 【256】向量，
          这里作为加载预训练Embedding，当然 这里的外部Embedding，可以适当进行 整体缩小一下

"""


def ItemSeqCF(user_columns, item_columns, neg_item_columns, att_layer_num=1, head_num=2, att_embedding_size=None,
              use_res=True):
    """
    :param user_columns: 用户user_id feature
    :param item_columns: 用户点击序列 items
    :param neg_item_columns: 负采样标签，针对正样本 对应多少个负样本
    :param att_layer_num: attention 层数
    :param head_num: self_attention 头数目
    :param att_embedding_size: 默认 同 item及user dim 一致
    :param use_res: 针对attention 阶段，是否使用残差
    :return:
    """

    feature_columns = list(user_columns) + list(item_columns) + list(neg_item_columns)

    feature_input_layers = build_input_layers(feature_columns)
    inputs_list = feature_input_layers.values()

    embedding_outputs_dict, dense_outputs_dict, neg_outputs_dict = build_embedding_outputs(feature_input_layers,
                                                                                           feature_columns,
                                                                                           is_concat=False)

    """根据 历史点击序列， 做muti-self-attention"""
    hist_feature_columns = [feature_column for feature_column in feature_columns if
                            isinstance(feature_column, VarLenSparseFeat)]
    history_item_emb_list = [embedding_outputs_dict.get(feature_column.name) for feature_column in hist_feature_columns]

    if len(history_item_emb_list) > 1:
        history_item_emb = Concatenate()(history_item_emb_list)
    else:
        history_item_emb = history_item_emb_list[0]

    att_output = history_item_emb

    """ 用户点击序列 进行 attention"""
    for i in range(att_layer_num):
        att_output = MutiHeadSelfAttention(head_num,
                                           att_embedding_size=att_embedding_size,
                                           use_lm_masking=True,
                                           use_res=use_res)(att_output)

        att_output = DNNLayer(repeat_num=2)(att_output)

    user_emb_output = [embedding_outputs_dict.get(user_column.name) for user_column in user_columns][0]

    output = user_emb_output + att_output

    neg_output = [v for k, v in neg_outputs_dict.items()][0]

    pos_output = history_item_emb

    mask = tf.cast(tf.not_equal(feature_input_layers['history_items'], 0), dtype=tf.float32)


    pred_pos_output = pos_output[:,-1,:]
    pred_output = output[:,-1,:]
    pred = tf.sigmoid(tf.multiply(pred_output, pred_pos_output))

    loss = compute_loss(mask, output, pos_output, neg_output)

    train_model = Model(inputs=inputs_list, outputs=[])
    train_model.add_loss(loss)



    return train_model


def compute_loss(mask, output, pos_output, neg_output):
    """
    计算loss = 正样本loss + 负样本loss
    :param mask:
    :param output:
    :param true_ouput:
    :param neg_output:
    :return:
    """

    mask = mask[:, :-1]
    output = output[:, :-1, :]
    pos_output = pos_output[:, 1:, :]
    neg_output = neg_output[:, 1:, ...]

    pos_logits = tf.reduce_sum(tf.multiply(output, pos_output), axis=-1, keepdims=True)
    pos_trues = K.ones_like(pos_logits)
    pos_loss = tf.squeeze(K.binary_crossentropy(pos_trues, pos_logits, from_logits=True), -1)
    pos_loss = tf.reduce_sum(pos_loss * mask, axis=-1) / tf.reduce_sum(mask, axis=-1)

    ouput = Lambda(lambda x: K.expand_dims(x, axis=-2))(output)
    neg_logits = tf.matmul(neg_output, ouput, transpose_b=True)
    neg_trues = K.zeros_like(neg_logits)
    neg_loss = tf.squeeze(K.binary_crossentropy(neg_trues, neg_logits, from_logits=True), -1)
    neg_loss = tf.reduce_sum(neg_loss, axis=-1)
    neg_loss = tf.reduce_sum(neg_loss * mask, axis=-1) / tf.reduce_sum(mask, axis=-1)

    loss = pos_loss + neg_loss

    return loss
