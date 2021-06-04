# -*- coding: utf-8 -*-
import tensorflow as tf
import random
import numpy as np
import os

"""固定随机种子"""
seed_value = 39
random.seed(seed_value)
np.random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

tf.random.set_seed(seed_value)
