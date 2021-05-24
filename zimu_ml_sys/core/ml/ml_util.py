# -*- coding: utf-8 -*-
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(estimator, X, y, title='',
                        scoring=None,
                        cv=None,  # 交叉验证
                        n_jobs=None,  # 设定索要使用的线程
                        train_sizes=np.linspace(0.1, 1.0, 10)
                        ):
    """
    学习曲线 : 看模型受 训练集大小的影响
        针对 是不同数量的数据集 针对 训练集 和 测试集 上面的打分
        主要看是不是 我们的数据集 是否满足的模型需要
        这个与交叉验证 不是一个东西
        学习曲线不是调节参数，只是查看数据集大小 对模型的影响

        eg:
            parm = {"objective":"reg:squarederror"}
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            plot_learning_curve(XGBR(parm=parm,n_estimators=100,random_state=420),Xtrain,Ytrain,'xgb',cv=cv)
            plt.show()
    :param estimator: 模型
    :param title:
    :param X:
    :param y:
    :param scoring:
    :param ax:
    :param ylim:
    :param cv:
    :param n_jobs:
    :return:
    """

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
                                                            , shuffle=True
                                                            , cv=cv
                                                            # ,random_state=420
                                                            , scoring=scoring
                                                            , n_jobs=n_jobs
                                                            , train_sizes=train_sizes)
    ax = plt.figure()
    ax.set_title(title)

    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()  # 绘制网格，不是必须
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-'
            , color="r", label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-'
            , color="g", label="Test score")
    ax.legend(loc="best")
    plt.show()


def plot_validation_curve(estimator, X, y,
                          param_name,
                          param_range,
                          scoring=None,
                          title='',
                          cv=None,  # 交叉验证
                          n_jobs=None,  # 设定索要使用的线程
                          ):
    """
    交叉验证: 一般会选择模型,固定其他所有参数，只有一个参数，给定该参数范围，
        然后通过使用5折交叉验证，来评定哪个参数对模型效果最好

        eg:
        validation_curve(Ridge(), X, y, "alpha",
                                        np.logspace(-7, 3, 3),
                                        cv=5)

    :param estimator:
    :param X:
    :param y:
    :param param_name:
    :param param_range:
    :param scoring:
    :param title:
    :param cv:
    :param n_jobs:
    :return:
    """

    train_scores, test_scores = validation_curve(
        estimator, X, y, cv=cv, param_name="gamma", param_range=param_range,
        scoring=scoring, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
