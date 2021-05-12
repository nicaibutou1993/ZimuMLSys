# -*- coding: utf-8 -*-
import json
import requests
from tasks.tianchi_news_rec.constant import TF_SERVING_USER_TOWER_URL, FAISS_ITEM_TOWER_URL,TASK_NAME
from zimu_ml_sys.utils.data_util import read_encoding_mapping_data
from tasks.tianchi_news_rec.offine.match.tower_task import TowerTask
import numpy as np


class TowerService(object):

    def __init__(self):
        self.user_tower_urls = TF_SERVING_USER_TOWER_URL.split(',')

        self.item_tower_urls = FAISS_ITEM_TOWER_URL.split(',')

        encoding_mapping = read_encoding_mapping_data(task_name=TASK_NAME,feature_fields=['user_id','article_id'])

        self.id2user, _ = encoding_mapping.get('user_id')

        self.id2item, _ = encoding_mapping.get('user_id')

    def get_user_output_vectors(self, dataset_x, user_field_name='user_id'):
        """
        这里传入的是 tf dataset 数据格式 {key:tensor} ,相当于批量请求
        :param dataset_x:
        :param user_field_name: user field name
        :return:
        """

        batch_size = dataset_x.get(user_field_name).shape[0]

        user_features = {name: feature.numpy().reshape(batch_size, -1).tolist() for name, feature in dataset_x.items()}

        data = {"inputs": user_features}
        input_data_json = json.dumps(data)
        response = requests.post(np.random.choice(self.user_tower_urls), data=input_data_json)
        user_output_vectors = list(json.loads(response.text)['outputs'])

        user_ids = user_features.get(user_field_name)

        return user_output_vectors, user_ids

    def get_tower_recs(self):
        """
        针对所有的test dataset 进行测试，具体还需要改写
        :return:
        """
        tower_task = TowerTask()
        columns = tower_task.get_feature_columns()
        test_dataset = tower_task.get_dataset(columns,is_train=False)

        datas = []
        for x, y in test_dataset:
            user_output_vectors, user_ids = self.get_user_output_vectors(x)

            data = {"user_output_vectors": user_output_vectors, 'rec_nums': 10}

            input_data_json = json.dumps(data)
            response = requests.post(np.random.choice(self.item_tower_urls), json=input_data_json)

            rec_datas = eval(response.text)
            item_ids = rec_datas.get('item_ids')
            datas.extend(zip(user_ids, item_ids))

        res = []
        for user_id, item_id in datas:
            user_id = user_id[0]

            """历史点击内容 过滤这里 需要其他接口 读取，这里没有实现"""
            user_history_click_ids = self.user_history_click_items_mapping.get(user_id)
            decoder_item_ids = [self.id2item.get(id) for id in item_id if id not in user_history_click_ids][
                               :self.REC_ITEM_NUM]

            decoder_user_id = self.id2user.get(user_id)

            data = [decoder_user_id]
            data.extend(decoder_item_ids)

            res.append(data)
