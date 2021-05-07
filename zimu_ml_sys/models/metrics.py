import tensorflow.keras as keras
import numpy as np
import os
import tensorflow as tf


def get_callback(test_dataset, model_path, version="0001"):
    class Evaluate(keras.callbacks.Callback):

        def __init__(self, test_dataset, model_path, version):
            self.test_dataset = test_dataset
            self.model_path = model_path
            self.version = version
            self.f1 = 0.0

            self.validation_data = test_dataset

            # self.validation_data

            _path = os.path.join(self.model_path, version)

            if not os.path.exists(_path):
                os.makedirs(_path)

        def on_epoch_end(self, epoch, logs=None):

            true_y = []
            pred_y = []
            for test_x, test_y in self.test_dataset:
                array = np.array(self.model.predict(test_x))

                _array = np.where(array > 0.5, 1, 0)

                pred_y.extend(_array)
                true_y.extend(test_y.numpy()[:, np.newaxis])

            acc, precision, recall, f1 = get_metrics(true_y, pred_y)

            if self.f1 < f1:
                version = self.version

                tf.saved_model.save(self.model, os.path.join(self.model_path, version))

                self.f1 = f1

    evaluate = Evaluate(test_dataset, model_path, version)

    return evaluate


def get_metrics(true_y, pred_y, is_print=True):
    pred_y = np.array(pred_y)
    true_y = np.array(true_y)
    acc = sum(pred_y == true_y) / float(len(pred_y))

    precision = sum(true_y[true_y == pred_y] == 1) / float(sum(pred_y == 1))

    recall = sum(true_y[true_y == pred_y] == 1) / float(sum(true_y == 1))

    f1 = 2 * precision * recall / (precision + recall)
    if is_print:
        print("acc: ", acc, "precision: ", precision, "recall: ", recall, "f1: ", f1)

    return acc, precision, recall, f1
