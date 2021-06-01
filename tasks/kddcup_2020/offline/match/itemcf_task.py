# -*- coding: utf-8 -*-
import os
import time

import pandas as pd

from tasks.kddcup_2020.constant import TASK_NAME
from zimu_ml_sys.constant import PREPROCESS_DATA_PATH, FEATURE_DATA_PATH
from zimu_ml_sys.utils.data_util import df_group_parallel
import joblib
from tasks.kddcup_2020.offline.match.graph_embedding_task import GraphEmbeddingTask
from gensim.models.keyedvectors import KeyedVectors
import math
import numpy as np
from tqdm import tqdm
import pickle
import gc
from zimu_ml_sys.utils.parallel_util import process_parallel


class ItemCFTask(object):
    """
        1. compute_sim_item： 计算全局 item 与 item 两两之间的相似度
            根据用户点击序列，具体因素：（时间差，位置差，item 热门程度，用户个人的活跃程度，item与item 模型相似度：（node,deep,text,image等等））

        2. recommend： 召回阶段，实现一个用户 召回 topN 个item
            根据用户点击序列：根据用户序列中每一个item：
                    1. 找到该item 相似的topN items(备注：从全局中查找)
                    2. 用户点击item 距离 当前的 （时间差，位置差，推荐的item 最早出现时间，推荐item 热门程度，）
    """

    """
    针对数据量大：
        1. 写入多个文件，针对文件的加载，使用多进程方式 并发加载
        2. 使用joblib，同时 数据中如果有数组，尽量使用numpy.array 方式，joblib 会将numpy 数据 压缩成很小的大小
    """

    """
    相关度计算：
        1. 通过模型，训练item Embedding：  粗粒度 (这里有一定随机游走，没有更 精细的体现 位置差 时间差 等等其他更细节的特征加入)
           根据用户点击序列，通过node2vector 或者 deepwalker 模型 通过随机游走的方式 生成样本，并训练 item Embedding
           这里的item Embedding 代表全局 中 item的向量表示
           两个item 相关性：可以通过向量点乘的方式

        2. compute_sim_item:
            根据 用户序列 重新计算 item与item 相似度： 细粒度  （这里仅仅表示的是： item 与 item 之间的相似度情况）

            循环用户点击序列，针对用户自己点击的item之间进行 item与item相似度，首先这里针对随机游走的方式 更加精确了。
            考虑 时间差，位置差，node_sim,deep_sim,txt_sim,是否逆序,item 热度，用户活跃情况
            这里融入的模型训练出的 相似度，同时加入了 用户及item 本身的一些 属性，使得 item与item相似度 更加细粒度话

        3. recommend:
            针对用户推荐的话，需要遍历用户点击序列中的item，找出用户点击的item，根据它找出相似的items,
            在考虑用户的 点击的item 针对最新的点击的item，考虑时间差，位置差等等因素加入到相似的items,
            最终进行累积，取top1000 作为用户召回的item
    """

    preprocess_root_data_path = os.path.join(PREPROCESS_DATA_PATH, TASK_NAME)
    feature_root_data_path = os.path.join(FEATURE_DATA_PATH, TASK_NAME)

    time_field_name = 'time'
    user_field_name = 'user_id'
    item_field_name = 'item_id'

    """使用node2vector模型 存放两两item 计算的相似度,避免重复计算"""
    node2vector_sim_mapping = {}

    """使用deepwalker模型 存放两两item 计算的相似度,避免重复计算"""
    deepwalker_sim_mapping = {}

    """使用提供的文本模型 存放两两item 计算的相似度,避免重复计算"""
    txt_sim_mapping = {}

    RELATE_ITEM_NUM = 100

    is_evaluate = True

    def __init__(self):

        graph_embedding_task = GraphEmbeddingTask()
        self.node2vector_model = graph_embedding_task.load_node2vector_model()
        self.deepwalker_model = graph_embedding_task.load_deepwalker_model()
        self.txt_model = KeyedVectors.load_word2vec_format(os.path.join(self.feature_root_data_path, 'w2v_txt_vec.txt'))

    def recommend(self, user_ids, sim_item_corr,
                  user_items_dict, user_times_dict,
                  item_value_counts_dict, item_time_dict,
                  top_k=100, item_num=200
                  ):

        """
        推荐逻辑：

            1. items[::-1]获取当前用户：历史点击序列items,倒序处理：
            2. recent_item = items[0]  #最近点击item
            2. for item in items:
                    for sim_item in sim_item_corr(item)[:500]:  ## 取出当前点击item，sim_item_corr 取出top 500个与当前点击item 相似的item
                        # 相似item 不能再 用户历史点击列表中
                        1. 时间特征：
                          t0:表示最近点击时间
                          t1: 表示该用户点击 item时间
                          t2: 推荐相似的item 在所有用户中 最早点击时间

                          delta_t1 = abs(t0 - t1) * 650000   ##表示用户点击item 距离当前时间，如果时间越长 表示该item寻找的相似推荐 应该权重越小
                          delta_t2 = abs(t0 - t2) * 650000   ## 表示 推荐item 最早出现的时间，最早出现时间最早，说明该item 不新鲜 权重越小

                        2. 热度特征：
                            alpha = max(0.2, 1 / (1 + item_dict[j])) ##推荐的item，热度越高，其权重 越低

                        3. 位置特征：
                            beta = max(0.5, (0.9 ** loc)) ## 距离最近位置越远，权重越低


                        该推荐的sim_item 计算总体 推荐值：

                            wij[0]  ## 表示item 与 sim_item 的相似度情况
                        * (alpha ** 2)  ## sim_item热度
                        * (beta)        ## sim_item位置
                        * (max(0.5, 1 / (1 + delta_t1)) ** 2)  ## 表示用户点击item 距离当前时间
                        * max(0.5, 1 / (1 + delta_t2))   ## 表示 推荐item 最早出现的时间

                        计算累积值：用户针对该sim_item 累积推荐值

        input:item_sim_list, user_item, uid, 500, 50
        # 用户历史序列中的所有商品均有关联商品,整合这些关联商品,进行相似性排序
        :param sim_item_corr: {item:{relate_item:[sim_score,....]}}
        :param user_item_dict: 每一个用户点击 item序列
        :param user_id: 用户id
        :param times: 该用户 点击每一个item 的时间序列
        :param item_dict: 每一个item 点击的次数
        :param item_time_dict: 每一个item 最早点击时间
        :param top_k: 推荐个数
        :param item_num: item个数
        :return:
        """

        ranks = {}

        file_nums = 10
        total_num = len(user_ids)
        batch_size = int(total_num / file_nums)

        for i, user_id in enumerate(user_ids):
            if user_id in user_items_dict:
                items = user_items_dict[user_id][::-1]
                times = user_times_dict[user_id][::-1]
                t0 = times[0]
                rank = {}
                for loc, item in enumerate(items):

                    for sim_item, wij in sorted(sim_item_corr[item].items(),
                                                key=lambda x: x[1][0],
                                                reverse=True)[:top_k]:

                        '''
                        wij
                        The meaning of each columns:
                        {'sim': 0,------------------------0
                          'item_cf': 0,-------------------1
                          'item_cf_weighted': 0,----------2
                          'time_diff': np.inf,------------3
                          'loc_diff': np.inf,-------------4
                          'node_sim': -1e8,-----------5
                          'node_sim_sum':0,---------------6
                          'deep_sim': -1e8,-----------7
                          'deep_sim_sum':0----------------8
                                                  }
                        '''

                        if sim_item not in items:
                            '''
                            RANK
                            {'sim': 0,---------------------------------0
                            'item_cf': 0,------------------------------1
                            'item_cf_weighted': 0,---------------------2
                            'time_diff': np.inf,-----------------------3
                            'loc_diff': np.inf,------------------------4
                            # Some feature generated by recall
                            'time_diff_recall': np.inf,----------------5
                            'time_diff_recall_1': np.inf,--------------6
                            'loc_diff_recall': np.inf,-----------------7
                            # Nodesim and Deepsim
                              'node_sim': -1e8,--------------------8
                              'node_sim_sum':0,------------------------9
                              'deep_sim': -1e8,--------------------10
                              'deep_sim_sum':0,------------------------11
                                                      }
                            '''

                            rank.setdefault(sim_item,
                                            np.array(
                                                [0, 0, 0, np.inf, np.inf, np.inf, np.inf, np.inf, -1e8, 0, -1e8, 0]))

                            """最新点击时间 与 当前点击的item 时间差"""
                            delta_t1 = abs(t0 - times[loc]) / (60 * 60)
                            """相似的item 最早什么时候出现点击"""
                            delta_t2 = abs(t0 - item_time_dict[sim_item]) / (60 * 60)

                            """当前item 点击的次数，次数越小，权重越高，相当于 热门降权"""
                            alpha = max(0.2, 1 / (1 + item_value_counts_dict[sim_item]))

                            """当前位置"""
                            beta = max(0.5, (0.9 ** loc))

                            """时间差 权重"""
                            theta = max(0.5, 1 / (1 + delta_t1))

                            """item出现的最早时间，出现越早，权重越小"""
                            gamma = max(0.5, 1 / (1 + delta_t2))

                            """1. 针对用户下，对sim_item进行 累积 得分
                                这个累积体现在：
                                    序列中：每一个item 找到其推荐的500items,这里会有重复sim_item 会被多次推荐出来，
                                    被推荐次数越多，说明 sim_item 与用户 更加相关
                            """

                            rank[sim_item][0] += self.myround(wij[0] * (alpha ** 2) * (beta) * (theta ** 2) * gamma, 4)

                            """2. 计算该用户下的sim_item 累积出现的次数"""
                            rank[sim_item][1] += wij[1]

                            """3. 全局 item 与 sim_item 根据时间差，位置差，用户活跃程度 计算出来的相似度。
                                   这里针对该用户下的 sim_item,累积权重值"""
                            rank[sim_item][2] += wij[2]

                            """4. 全局来说：针对用户点击的item 与 sim_item ，wij[3]为该两个item点击的最短时间差
                                针对用户来说，其他点击的item 与 sim_item,
                                设置 用户，最小的 sim_item 点击时间差
                            """
                            if wij[3] < rank[sim_item][3]:
                                rank[sim_item][3] = wij[3]

                            """5. 全局来说：针对用户点击的item 与 sim_item ，wij[3]为该两个item点击的最短位置差
                                针对用户来说，其他点击的item 与 sim_item,
                                设置 用户，最小的 sim_item 位置差
                            """
                            if wij[4] < rank[sim_item][4]:
                                rank[sim_item][4] = wij[4]

                            """6. 设置 距离当前用户最新点击的 与 序列中item 推荐的sim_item,最短时间差 """
                            if delta_t1 < rank[sim_item][5]:
                                rank[sim_item][5] = self.myround(delta_t1, 4)

                            """7. 设置sim_item,全局最早出现 与当前 用户最近点击的 时间差 """
                            if delta_t2 < rank[sim_item][6]:
                                rank[sim_item][6] = self.myround(delta_t2, 4)

                            """8. 针对 用户点击items,那些出现 sim_item,距离最新点击 设置最短位置差"""
                            if loc < rank[sim_item][7]:
                                rank[sim_item][7] = loc

                            """9. 在用户历史点击中，与sim_item,设置最大node_sim 相似度"""
                            if wij[5] > rank[sim_item][8]:
                                rank[sim_item][8] = wij[5]

                            """10. wij[6] / wij[1] 相当于 node_sim,wij[1] 是次数，wij[6] 累积值
                                这里是针对用户下，历史点击items,出现推荐sim_item, 累积值
                            """
                            rank[sim_item][9] += wij[6] / wij[1]

                            """11. 在用户历史点击中，与sim_item,设置最大deep_sim 相似度"""
                            if wij[7] > rank[sim_item][10]:
                                rank[sim_item][10] = wij[7]

                            """10. wij[8] / wij[1] 相当于 deep_sim,wij[1] 是次数，wij[8] 累积值
                               这里是针对用户下，历史点击items,出现推荐sim_item, 累积值
                           """
                            rank[sim_item][11] += wij[8] / wij[1]

                rank = sorted(rank.items(), key=lambda x: x[1][0], reverse=True)[:item_num]

                ranks[user_id] = rank

            if i % batch_size == 0 or i == total_num - 1:
                if i != 0:
                    part = int(i / batch_size) + 1 if i == total_num - 1 else int(i / batch_size)
                    print('write user rec items ', part)
                    data_path = os.path.join(self.feature_root_data_path, 'itemcf_user_rec_items_%s.pkl' % str(part))
                    joblib.dump(ranks, open(data_path, mode='wb'))
                    del ranks
                    ranks = {}

        # joblib.dump(ranks, open(os.path.join(self.feature_root_data_path, 'itemcf_user_rec_items.pkl'), mode='wb'))

        # return ranks

    def compute_sim_item(self, train_df):

        """
        itemCF 思路：

            1. 获取每一个用户 点击的一系列 items list
            2. 获取每一个用户 点击的一系列 时间 list
            3. 获取每一个item 所有点击的用户 set
            4. 获取每一个item 点击最早的时间
            5. 定义sim_item：{item:{ralate_item:[总体相似度 （有很多因素求得总体相似度）,共线次数,位置_时间_该用户点击量 混合权重,
                                                时间差,位置差,最大node2vector 相似度,总和node2vector相似度,
                                                最大deepwalk 相似度, 总和deepwalk 相似度]}}
                for user,items in user_items:  ## 针对每一个用户 点击的所有的items
                    for current_item in items:  ## 循环每一个点击的item
                        for relate_item in items: ## 这里为了计算 当前item 与用户点击的一系列item 的相关度

                            ## 这里针对 都是 current_item 与 relate_item 在同一个用户下，计算这两个item的相似度，
                            ## 循环所有的用户点击的所有的item，累积求和  current_item 与 relate_item 的相似度，
                            ## 即可得到 每一个item与item 之间总体相似度情况。
                            ## 推荐过程：获取该用户所有的items：然后计算每一个item 相关联推荐，最终

                            ## 1. node_sim: 训练好的 node2vector 计算 item 与 relate_item 相似度
                            ## 2. deep_sim: 训练好的 deepwalk  计算 item 与 relate_item 相似度
                            ## 3. txt_sim: 计算文本相似度

                            1. 总体相似度 公式 备注（针对 current_item_relate_item 每计算一次 都是在原有的基础上求和）：
                                (node_sim ** 2)  ## node_sim
                                * deep_sim      ## deep_sim
                                * txt_sim       ## 文本相似度
                                * max(0.5, (0.9 ** (loc1 - loc2 - 1)))   ## 位置： 当前item 与ralate_item 位置间隔多少，如果靠的越近 越相似
                                * (max(0.5, 1 / (1 + delta_t)))  ## 时间间隔：当前item 与 relate_item 时间间隔，时间间隔越近 越相似

                                * (1.0 or 0.8) ## 顺序特征，如果current_item 在ralate_item 之前 则选择1.0 反之，0.8 表示逆序 权重值降低

                                /       ## 底下分母，值越大，总体相似度 越低
                                   (
                                    math.log(len(users) + 1)    ## len(users): 表示当前item 被多少用户点击,点击越多 表明item越热，
                                                                ## 这里为了降低推荐系统喜欢推荐热门商品，这里针对热度商品降权

                                    * math.log(1 + user_item_len) ## user_item_len：表示当前用户点击了多少item，如果用户特别喜欢点击商品，
                                                                ## 那么这种用户所点击的item 权重值要降低，不能推荐系统被某些活跃用户带偏了
                                   )


                            2. 共线次数：current_item 与 relate_item
                            3. 位置_时间_该用户点击量权重 公式：
                                    (0.8 ** (loc2 - loc1 - 1))   ## 当前item 与ralate_item 位置间隔多少，如果靠的越近 越相似
                                  * (1 - (t2 - t1) * 2000)   ## 时间间隔：当前item 与 relate_item 时间间隔，时间间隔越近 越相似
                                  / math.log(1 + len(items))  ## 那么这种用户所点击的item 权重值要降低，不能推荐系统被某些活跃用户带偏了

                            4. current_item 与 relate_item 最小时间间隔，在所有共线中 取最小
                            5. current_item 与 relate_item 最小位置间隔，在所有共线中 取最小
                            6. max_node_sim: 共线中，取最大值
                            7. sum_node_sim: 共线中，取累计值
                            8. max_deep_sim: 共线中，取最大值
                            9. sum_deep_sim: 共线中，取累计值

                ## 最终针对最终的相似度 再次针对热度 降权操作
                ## 总体相似度 / ((count(current_item) * count(relate_item)) ** 0.2)

        """

        train_df = train_df.sort_values(self.time_field_name)

        """item：users, key:item,value:表示哪些用户点击过该item"""
        item_users_df = train_df.groupby(self.item_field_name)[self.user_field_name].agg(set).reset_index()
        item_users_dict = dict(zip(item_users_df[self.item_field_name], item_users_df[self.user_field_name]))

        """每一个item 被点击的次数"""
        item_value_counts_dict = train_df[self.item_field_name].value_counts().to_dict()

        """每一个item，被点击 初始时间"""
        item_time_df = train_df.drop_duplicates(self.item_field_name, keep='first')
        item_time_dict = dict(zip(item_time_df[self.item_field_name], item_time_df[self.time_field_name]))

        """用户点击的 items 列表"""
        user_items_df = train_df.groupby(self.user_field_name)[self.item_field_name].agg(list).reset_index()
        user_items_dict = dict(zip(user_items_df[self.user_field_name], user_items_df[self.item_field_name]))

        """用户点击的 items 对应的时间列表"""
        user_times_df = train_df.groupby(self.user_field_name)[self.time_field_name].agg(list).reset_index()
        user_times_dict = dict(zip(user_times_df[self.user_field_name], user_times_df[self.time_field_name]))

        del item_users_df
        del item_time_df
        del user_items_df
        del user_times_df
        del train_df
        gc.collect()

        sim_item = {}

        print("begin compute sim")

        for user_id, items in tqdm(user_items_dict.items()):
            """
                根据用户点击序列，计算总体的item 与item 相关度
            """
            """获取用户点击序列"""
            times = user_times_dict[user_id]
            user_click_items_num = len(items)

            for loc1, item in enumerate(items):
                """当前点击的item，与用户所有的item，计算sim"""
                sim_item.setdefault(item, {})

                users_len = len(item_users_dict[item])

                for loc2, relate_item in enumerate(items):
                    if item == relate_item: continue

                    """
                      'sim': 0,------------------------0
                      'item_cf': 0,-------------------1
                      'item_cf_weighted': 0,----------2
                      'time_diff': np.inf,------------3
                      'loc_diff': np.inf,-------------4
                      'node_sim': -1e8,-----------5
                      'node_sim_sum':0,---------------6
                      'deep_sim': -1e8,-----------7
                      'deep_sim_sum':0----------------8
                    """
                    sim_item[item].setdefault(relate_item, np.array([0, 0, 0, np.inf, np.inf, -1e8, 0, -1e8, 0]))

                    """1. 时间差：
                    当前item 与 relate_itme 相隔时间差，时间差越大，表示相关性越小
                    这里 60 * 60 进行时间差 压缩，根据不同情况进行不同的缩放，
                    时间设置： 统计所有的时间差，然后取一个中值 即可
                    """
                    delta_time = self.myround(abs(times[loc1] - times[loc2]) / (60 * 60), 4)

                    """2. 位置差
                    两个item 点击的位置差，位置差越大，表示两item 相关性越小"""
                    delta_loc = abs(loc2 - loc1)

                    """3. node2vector 相似度
                    根据训练的node2vector 模型 判断 两item之间的相似情况
                    """
                    node_sim = self.myround(self.compute_node2vector_sim(item, relate_item), 4)

                    """4. deepwalker 相似度
                    根据训练的deepwalker 模型 判断 两item之间的相似情况
                    """
                    deep_sim = self.myround(self.compute_deepwalker_sim(item, relate_item), 4)

                    """5. 文本 相似度
                    使用提供的文本向量 判断 两item之间的相似情况
                    """
                    txt_sim = self.myround(self.compute_txt_sim(item, relate_item), 4)

                    """
                        计算两item 总体相似度情况， 
                        ## node_sim deep_sim 文本相似度 位置 时间差, item被去重用户点击的次数， 用户一共点击的item数目
                        
                    """

                    sim = (node_sim ** 2) \
                          * deep_sim \
                          * txt_sim \
                          * max(0.5, (0.9 ** (delta_loc - 1))) \
                          * (max(0.5, 1 / (1 + delta_time))) \
                          / (
                                  math.log(1 + users_len)
                                  * math.log(1 + user_click_items_num))

                    if loc1 > loc2:
                        sim = sim * 0.8  ## 表示逆序 需要降权

                    """1. 计算 两item 累积权重"""
                    sim_item[item][relate_item][0] = self.myround(sim_item[item][relate_item][0] + sim, 4)

                    """2. 计算 两item 共线次数"""
                    sim_item[item][relate_item][1] += 1

                    """3. 仅仅只考虑 位置 时间 用户序列长度 来评定 两个item的相关度 去除相关模型相似度"""
                    sim_item[item][relate_item][2] = self.myround(
                        sim_item[item][relate_item][2] + max(0.5, (0.9 ** (delta_loc - 1))) \
                        * (max(0.5, 1 / (1 + delta_time))) \
                        / math.log(1 + len(items)), 4)

                    """4. 取两item 时间差最短"""
                    if delta_time < sim_item[item][relate_item][3]:
                        sim_item[item][relate_item][3] = delta_time

                    """5. 取两item 位置差最短"""
                    if delta_loc < sim_item[item][relate_item][4]:
                        sim_item[item][relate_item][4] = delta_loc

                    """存储 node 及 deep 相似度 值"""
                    sim_item[item][relate_item][5] = node_sim
                    sim_item[item][relate_item][6] = self.myround(sim_item[item][relate_item][6] + node_sim, 4)

                    sim_item[item][relate_item][7] = deep_sim
                    sim_item[item][relate_item][8] = self.myround(sim_item[item][relate_item][8] + deep_sim, 4)

        """
            最终针对最终的相似度 再次针对热度 降权操作 
                总体相似度 / ((count(current_item) * count(relate_item)) ** 0.2) 
         """

        del self.node2vector_model
        del self.deepwalker_model
        del self.txt_model
        # gc.collect()

        file_nums = 10
        total_num = len(sim_item)
        batch_size = int(total_num / file_nums)

        tmp_sim_item = {}
        """
        针对每一个item，协同过滤出 100 相近的items,
        并写入到文件中
        """
        for i, item in enumerate(list(sim_item.keys())):

            related_items = sim_item.pop(item)

            related_items = dict(
                sorted(related_items.items(), key=lambda x: x[1][0], reverse=True)[:self.RELATE_ITEM_NUM])

            for relate_item, cij in related_items.items():
                """针对热门item 总体上 再次 降权"""
                cosine_sim = cij[0] / ((item_value_counts_dict[item] * item_value_counts_dict[relate_item]) ** 0.2)
                related_items[relate_item][0] = self.myround(cosine_sim, 4)

            tmp_sim_item[item] = related_items

            if i % batch_size == 0 or i == total_num - 1:
                if i != 0:
                    part = int(i / batch_size) + 1 if i == total_num - 1 else int(i / batch_size)
                    data_path = os.path.join(self.feature_root_data_path, 'itemcf_sim_item_%s.pkl' % str(part))
                    joblib.dump(tmp_sim_item,
                                open(data_path, mode='wb'))

                    del tmp_sim_item
                    tmp_sim_item = {}

        joblib.dump((user_items_dict, user_times_dict, item_value_counts_dict, item_time_dict),
                    open(os.path.join(self.feature_root_data_path, 'itemcf_other_sim_item.pkl'), mode='wb'))

    def myround(self, x, thres):
        temp = 10 ** thres
        return int(x * temp) / temp

    def compute_node2vector_sim(self, item, relate_item):
        """计算 两item ，node2vector 相似度，不相似 值为 0.5 """
        key = '_'.join(str(i) for i in sorted([item, relate_item]))
        if key in self.node2vector_sim_mapping:
            return self.node2vector_sim_mapping[key]
        try:
            node_sim = 0.5 * (self.node2vector_model.similarity(str(item), str(relate_item))) + 0.5
        except:
            node_sim = 0.5
        self.node2vector_sim_mapping[key] = node_sim
        return node_sim

    def compute_deepwalker_sim(self, item, relate_item):
        """计算 两item ，deepwalker 相似度，不相似 值为 0.5 """
        key = '_'.join(str(i) for i in sorted([item, relate_item]))
        if key in self.deepwalker_sim_mapping:
            return self.deepwalker_sim_mapping[key]
        try:
            deep_sim = 0.5 * (self.deepwalker_model.similarity(str(item), str(relate_item))) + 0.5
        except:
            deep_sim = 0.5
        self.deepwalker_sim_mapping[key] = deep_sim
        return deep_sim

    def compute_txt_sim(self, item, relate_item):
        """计算 两item ，deepwalker 相似度，不相似 值为 0.5 """
        key = '_'.join(str(i) for i in sorted([item, relate_item]))
        if key in self.txt_sim_mapping:
            return self.txt_sim_mapping[key]
        try:
            txt_sim = 0.5 * (self.txt_model.similarity(str(item), str(relate_item))) + 0.5
        except:
            txt_sim = 0.5
        self.txt_sim_mapping[key] = txt_sim
        return txt_sim

    def read_data(self):
        """读取数据集"""
        train_df = pd.DataFrame()

        if self.is_evaluate:
            validate_df = pd.read_csv(os.path.join(self.preprocess_root_data_path, 'validate.csv'))
        else:
            train_df = pd.read_csv(os.path.join(self.preprocess_root_data_path, 'train.csv'))
            validate_df = pd.read_csv(os.path.join(self.preprocess_root_data_path, 'validate.csv'))

        return train_df, validate_df

    def rec_fill_hot_items(self, user_rec_items_df, hot_items, topk=50):
        """针对召回数据进行评估
            预测结果：itemcf_推荐的前 top50
            真实结果：真实用户下一次点击的item
        """

        print('fill hot items ')
        user_rec_items_df = user_rec_items_df[['user_id', 'item_id', 'feature_0']]

        user_ids = list(user_rec_items_df['user_id'].unique())

        scores = [-i for i in range(1, len(hot_items) + 1)]

        hot_df = pd.DataFrame(user_ids * len(hot_items), columns=['user_id'])
        hot_df['item_id'] = hot_items * len(user_ids)
        hot_df['feature_0'] = scores * len(user_ids)

        user_rec_items_df = user_rec_items_df.append(hot_df)
        user_rec_items_df = user_rec_items_df.sort_values('feature_0', ascending=True)
        user_rec_items_df = user_rec_items_df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')

        user_rec_items_df['rank'] = user_rec_items_df.groupby('user_id')['feature_0'].rank(method='first',
                                                                                           ascending=False)

        user_rec_items_df = user_rec_items_df[user_rec_items_df['rank'] <= topk]

        # print(user_rec_items_df.head(100))

        user_rec_items_df.to_csv(os.path.join(self.feature_root_data_path, 'match_evaluate_itemcf_user_rec_items.csv'))

        # user_rec_items_df.grouby('user_id')['item_id']

    def func(self, file_name):
        print(file_name)
        sim_item_corr = joblib.load(open(file_name, mode='rb'))
        return sim_item_corr

    def evaluate(self, predictions, answers, rank_num):
        """
        ##https://blog.csdn.net/weixin_41332009/article/details/113343838
        ##推荐算法常用评价指标：ROC、AUC、F1、HR、MAP、NDCG
        召回评估
        predictions： {user_id:[items]}
        answers: {user_id:(item_id,item_count)}
        """

        list_item_degress = []
        for user_id in answers:
            item_id, item_degree = answers[user_id]
            list_item_degress.append(item_degree)
        list_item_degress.sort()
        median_item_degree = list_item_degress[len(list_item_degress) // 2]

        num_cases_full = 0.0
        ndcg_50_full = 0.0
        ndcg_50_half = 0.0
        num_cases_half = 0.0
        hitrate_50_full = 0.0
        hitrate_50_half = 0.0
        for user_id in answers:
            if user_id in predictions:
                item_id, item_degree = answers[user_id]
                rank = 0
                while rank < rank_num and predictions[user_id][rank] != item_id:
                    rank += 1
                num_cases_full += 1.0
                if rank < rank_num:
                    ndcg_50_full += 1.0 / np.log2(rank + 2.0)
                    hitrate_50_full += 1.0
                if item_degree <= median_item_degree:
                    num_cases_half += 1.0
                    if rank < rank_num:
                        ndcg_50_half += 1.0 / np.log2(rank + 2.0)
                        hitrate_50_half += 1.0
        ndcg_50_full /= num_cases_full
        hitrate_50_full /= num_cases_full
        ndcg_50_half /= num_cases_half
        hitrate_50_half /= num_cases_half

        print([ndcg_50_full, ndcg_50_half,
               hitrate_50_full, hitrate_50_half])

        return np.array([ndcg_50_full, ndcg_50_half,
                         hitrate_50_full, hitrate_50_half], dtype=np.float32)

    def run(self):

        if self.is_evaluate:

            user_items_dict, \
            user_times_dict, \
            item_value_counts_dict, \
            item_time_dict = joblib.load(os.path.join(self.feature_root_data_path, 'itemcf_other_sim_item.pkl'))

            _, validate_df = self.read_data()

            validate_df['item_count'] = validate_df['item_id'].apply(lambda x: item_value_counts_dict.get(x))

            path = os.path.join(self.feature_root_data_path, 'match_evaluate_itemcf_user_rec_items.csv')
            if os.path.exists(path):
                user_item_rec_df = pd.read_csv(path)

                user_item_rec_df = user_item_rec_df.sort_values('rank').groupby('user_id')['item_id'].agg(
                    list).reset_index()

                user_item_rec_df = user_item_rec_df.set_index('user_id')
                user_item_rec_dict = user_item_rec_df.to_dict(orient='index')
                user_item_rec_dict = {k: v.get('item_id') for k, v in user_item_rec_dict.items()}

                validate_df = validate_df.set_index('user_id')
                validate_dict = validate_df.to_dict(orient='index')
                validate_dict = {k: (v.get('item_id'), v.get('item_count')) for k, v in validate_dict.items()}

                self.evaluate(user_item_rec_dict, validate_dict, 50)


            else:

                user_rec_items_df = pd.read_csv(os.path.join(self.feature_root_data_path, 'itemcf_user_rec_items.csv'))

                user_items_dict, \
                user_times_dict, \
                item_value_counts_dict, \
                item_time_dict = joblib.load(os.path.join(self.feature_root_data_path, 'itemcf_other_sim_item.pkl'))

                hot_items = sorted(item_value_counts_dict.items(), key=lambda x: x[1], reverse=True)[:50]
                hot_items = [item[0] for item in hot_items]

                """针对 某些用户 推荐 可能 不足top 50,需要通过 热门 进行补充"""
                self.rec_fill_hot_items(user_rec_items_df, hot_items)







        else:
            train_df, validate_df = self.read_data()

            data_path = os.path.join(self.feature_root_data_path, 'itemcf_sim_item_1.pkl')

            if not os.path.exists(data_path):
                self.compute_sim_item(train_df)

            file_names = [os.path.join(self.feature_root_data_path, name)
                          for name in filter(lambda x: x.startswith('itemcf_sim_item_'),
                                             os.listdir(self.feature_root_data_path))]
            datas = process_parallel(file_names, self.func, n_jobs=len(file_names))
            sim_item_corr = datas.pop(0)
            for data in datas:
                sim_item_corr.update(data)

            user_items_dict, \
            user_times_dict, \
            item_value_counts_dict, \
            item_time_dict = joblib.load(os.path.join(self.feature_root_data_path, 'itemcf_other_sim_item.pkl'))

            user_ids = validate_df['user_id'].unique()

            self.recommend(user_ids, sim_item_corr,
                           user_items_dict, user_times_dict,
                           item_value_counts_dict, item_time_dict)

            """将用户推荐 数据 写成 csv 格式"""
            file_names = [os.path.join(self.feature_root_data_path, name)
                          for name in filter(lambda x: x.startswith('itemcf_user_rec_items_'),
                                             os.listdir(self.feature_root_data_path))]

            # file_names = [os.path.join(self.feature_root_data_path,'itemcf_user_rec_items_1.pkl')]

            datas = process_parallel(file_names, self.func, n_jobs=len(file_names))
            user_rec_items = datas.pop(0)
            for data in datas:
                user_rec_items.update(data)

            datas = []
            for user_id, rec_items_data in user_rec_items.items():
                try:
                    if len(rec_items_data) > 0:
                        users = np.array([[user_id]] * len(rec_items_data))
                        rec_items_data = np.array(rec_items_data)
                        items = rec_items_data[:, 0].reshape(-1, 1)
                        features = np.concatenate(rec_items_data[:, 1], axis=0).reshape(-1, 12)
                        data = np.concatenate([users, items, features], axis=1)
                        datas.extend(data)
                except Exception as e:
                    print(e)
                    print(rec_items_data)

            data_df = pd.DataFrame(datas,
                                   columns=list(['user_id', 'item_id']) + list(
                                       ['feature_' + str(i) for i in range(12)]))

            user_rec_items_path = os.path.join(self.feature_root_data_path, 'itemcf_user_rec_items.csv')
            data_df.to_csv(user_rec_items_path)

    def explore_time_importance(self):
        """
            探索 用户相邻点击的item,时间相关性，一般来说 上一次点击 如果与 当前点击的 商品 点击距离越近 则与之 关系越近

            相邻item： 时间关系权重，时间越大，相邻item 权重越小：
                            delta  = abs( time(item1) - time(item2) ) / （60 * 60 * 12）  ## 60 * 60 * 12 时间 约束小一点
                            1 / ( 1 + delta )
            min           0.000000
            1%            4.955338
            5%           10.901743
            10%          19.821351
            25%          62.437254
            50%         291.373855
            75%        6101.011731
            90%       46305.648330
            95%       77996.023756
            99%      152713.001474
            max      327660.801257
        :param train_df:
        :return:
        """
        train_df, validate_df = self.read_data()

        train_df = train_df.sort_values(self.time_field_name)

        def compute_interval(name, group_df):
            group_df['time_shift'] = group_df[self.time_field_name].shift(-1)

            group_df['time_interval'] = group_df['time_shift'] - group_df[self.time_field_name]

            return group_df

        train_df = df_group_parallel(train_df.groupby(self.user_field_name), compute_interval, n_jobs=1)

        train_df = train_df[train_df['time_interval'].notnull()]

        train_df['time_interval'] = train_df['time_interval'].apply(lambda x: x / (60 * 60))

        print(train_df['time_interval'].describe([.01, .05, .10, .25, .5, .75, .90, .95, .99]))


if __name__ == '__main__':
    ItemCFTask().run()
