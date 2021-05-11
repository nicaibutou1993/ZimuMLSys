# -*- coding: utf-8 -*-

"""
使用faiss IndexFlatIP: 使用基于内积 方式进行搜索
"""
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from tasks.tianchi_news_rec.constant import TOWER_OUTPUT_DIM, TASK_NAME
from zimu_ml_sys.constant import FEATURE_DATA_PATH
import pickle
from flask import Flask, request
import numpy as np
import faiss
import json

app = Flask(__name__)

"""输出向量维度"""
output_dim_size = TOWER_OUTPUT_DIM

item_output_vectors_path = os.path.join(FEATURE_DATA_PATH, TASK_NAME + '/item_output_vectors')

"""faiss 模型初始化加载 item 向量"""
item_output_vectors, item_ids = pickle.load(open(item_output_vectors_path, mode='rb'))
item_output_vectors = np.array(item_output_vectors).astype(np.float32)

"""IndexFlatIP：精准的内积搜索"""
faiss_model = faiss.IndexFlatIP(output_dim_size)
faiss_model.add(item_output_vectors)


@app.route('/tower/rec_items', methods=["POST"])
def get_tower_rec_items():
    """
    双塔召回模型：召回大量的items, 传入到 排序模型中
    根据传入的用户向量，为用户 推荐 top N item,
    :return:
    """
    data = {}
    try:
        data = eval(request.get_json())
        user_output_vectors = data.get("user_output_vectors")

        rec_nums = 100
        if "rec_nums" in data:
            rec_nums = data.get("rec_nums")

        user_output_vectors = np.array(user_output_vectors).astype(np.float32)

        '''内积 cosine'''
        item_scores, item_ids = faiss_model.search(np.ascontiguousarray(user_output_vectors), rec_nums)

        data = {"item_scores": item_scores.tolist(),
                "item_ids": item_ids.tolist()}
    except Exception as e:
        print(e)

    data_json = json.dumps(data)
    return data_json


@app.route('/')
def test():
    return "ok!"


if __name__ == '__main__':
    # app.run(port=10001, debug=True)
    app.run()
else:
    application = app
