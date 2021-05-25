# -*- coding: utf-8 -*-
import os.path

import pandas as pd

from tasks.kddcup_2020.constant import TASK_DATA_PATH
from zimu_ml_sys.constant import PREPROCESS_DATA_PATH, FEATURE_DATA_PATH
from tasks.kddcup_2020.constant import TASK_NAME

pd.set_option('display.max_row', None)

"""
1. 生成 训练集 测试集 验证集
2. 生成 item 文本 图片向量

"""


class PreprocessData(object):
    """
            1. 将用户其他历史点击数据，为每一个用户召回 1000 条候选 item，可以用 验证集 查看召回的效果
            2. 根据召回的1000 item，提取特征，形成 训练集，训练集大小为  len(user_id) * 1000, 根据验证集 打标签，
                看下一个点击的是召回的哪一个item，命中则为 1，否则为0
            3. 采样： 正样本为 点击标签为1，负样本有999个，需要从中进行采样，采样5倍，然后进行重复 负采样6次，
                训练集共6份，每一个份 训练集大小：len(user_id) * ( 1 + 5)
            4. 训练6套模型后，进行模型融合
            5. 预测阶段：一个用户，根据历史点击，召回1000item，然后走模型 进行融合 排序，取top 50 进行推荐
    """

    stage = 9

    random_number_1 = 41152582
    random_number_2 = 1570909091

    preprocess_root_data_path = os.path.join(PREPROCESS_DATA_PATH, TASK_NAME)

    feature_root_data_path = os.path.join(FEATURE_DATA_PATH, TASK_NAME)

    if not os.path.exists(preprocess_root_data_path):
        os.makedirs(preprocess_root_data_path)

    def generate_train_and_test_validate_data(self):
        """生成 训练集 测试集 验证集
            1. 读取 kddcup_2020 训练集 及测试集
            2. 将用户最后一条点击数据 作为测试集
            3. 将用户倒数第二条点击数据，作为验证集
        """

        data_names = []
        for tag in ['underexpose_train', 'underexpose_test']:
            root_path = os.path.join(TASK_DATA_PATH, tag)
            list_names = [os.path.join(root_path, name)
                          for name in filter(lambda x: x.startswith(tag + '_click'),
                                             os.listdir(root_path))]

            data_names.append(list_names)

        columns = ['user_id', 'item_id', 'time']

        all_df = pd.DataFrame()
        for i, pathes in enumerate(zip(*data_names)):

            if i > self.stage:
                break

            for path in pathes:
                part_df = pd.read_csv(path, header=None, names=columns)
                all_df = pd.concat([all_df, part_df])

        all_df = all_df.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')

        all_df['time'] = all_df['time'].apply(lambda x: x * self.random_number_2 + self.random_number_1)

        all_df = all_df.reset_index(drop=True)

        all_df.groupby('user_id')

        all_df['rank'] = all_df['time'].groupby(all_df['user_id']).rank(ascending=True, method='first')

        validate_df = all_df[all_df['rank'] == 2].reset_index(drop=True).drop(labels='rank', axis=1)
        test_df = all_df[all_df['rank'] == 1].reset_index(drop=True).drop(labels='rank', axis=1)
        train_df = all_df[all_df['rank'] > 2].reset_index(drop=True).drop(labels='rank', axis=1)

        validate_df.to_csv(os.path.join(self.preprocess_root_data_path, 'validate.csv'), index=False)
        test_df.to_csv(os.path.join(self.preprocess_root_data_path, 'test.csv'), index=False)
        train_df.to_csv(os.path.join(self.preprocess_root_data_path, 'train.csv'), index=False)

    def generate_txt_and_image_feature(self):
        """写入 文本 及 image 向量"""

        root_path = os.path.join(TASK_DATA_PATH, 'underexpose_train')

        item_df = pd.read_csv(os.path.join(root_path, 'underexpose_item_feat.csv'), header=None)

        item_df[1] = item_df[1].apply(lambda x: float(str(x).replace('[', '')))
        item_df[256] = item_df[256].apply(lambda x: float(str(x).replace(']', '')))
        item_df[128] = item_df[128].apply(lambda x: float(str(x).replace(']', '')))
        item_df[129] = item_df[129].apply(lambda x: float(str(x).replace('[', '')))

        item_df.columns = ['item_id'] + ['txt_vec_{}'.format(f) for f in range(0, 128)] + \
                          ['img_vec_{}'.format(f) for f in range(0, 128)]

        # item_df.to_csv(os.path.join(self.feature_root_data_path, 'item_feat.csv'))

        item_nun = item_df['item_id'].nunique()

        item_df[['item_id'] + ['img_vec_{}'.format(f) for f in range(0, 128)]].to_csv(
            os.path.join(self.feature_root_data_path, "w2v_img_vec.txt"),
            sep=" ",
            header=[str(item_nun), '128'] + [
                ""] * 127,
            index=False,
            encoding='UTF-8')

        item_df[['item_id'] + ['txt_vec_{}'.format(f) for f in range(0, 128)]].to_csv(
            os.path.join(self.feature_root_data_path, "w2v_txt_vec.txt"),
            sep=" ",
            header=[str(item_nun), '128'] + [
                ""] * 127,
            index=False,
            encoding='UTF-8')


if __name__ == '__main__':
    PreprocessData().generate_train_and_test_validate_data()
