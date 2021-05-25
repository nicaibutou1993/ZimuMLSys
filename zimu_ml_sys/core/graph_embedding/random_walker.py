# -*- coding: utf-8 -*-
import itertools
import random
import numpy as np
from joblib import Parallel, delayed
from multiprocessing.pool import ThreadPool


def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [None] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:

        small_idx, large_idx = small.pop(), large.pop()

        accept[small_idx] = area_ratio_[small_idx]

        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])

        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1
    return accept, alias


def alias_sample(accept, alias):
    """
    根据权重 进行采样
    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]


class RandomWalker:

    def __init__(self, G, p=1, q=1):
        """
        1. 默认 p,q = 1 表示 deepwalk 节点与节点 进行随机采样，节点与节点的权重都是1
        2. 引入 当前节点 v 表示， 上一轮节点 t 表示： 当前节点 v 的邻居节点 用 x 表示

            1. 针对 v返回t:  w * 1/p    p：默认是大于1 ，给定的是2： 表示尽量不要在返回自己，处于小圈子 活动
            2. 针对 v 进入 x:
                    1. x 没有与 t 连接，即 x 不是t 的邻居： w * 1/q : 默认q :0.5 表示 深度优先，尽量游走远方
                    2. x 是t 的邻居： 则 w * 1
            详细看：https://time.geekbang.org/column/article/296672
            p 和 q 共同控制着随机游走的倾向性。参数 p 被称为返回参数（Return Parameter），p 越小，随机游走回节点 t 的可能性越大，
            Node2vec 就更注重表达网络的结构性。参数 q 被称为进出参数（In-out Parameter），
            q 越小，随机游走到远方节点的可能性越大，Node2vec 更注重表达网络的同质性。反之，当前节点更可能在附近节点游走。

        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        """
        self.G = G
        self.p = p
        self.q = q

    def deepwalk_walk(self, walk_length, start_node):
        """
        deepwalk: 随机游走：生成 sequence
        :param walk_length: 指定 sequence 长度
        :param start_node: 当前节点
        :return:
        """
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            """获取邻节点"""
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                """随机选择邻节点"""
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def node2vec_walk(self, walk_length, start_node):
        """
        node2vec: 根据权重进行采样：生成 sequence
        :param walk_length: 指定 sequence 长度
        :param start_node: 当前节点
        :return:
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                # 由于node2vec采样需要cur节点v，prev节点t，所以当没有前序节点时，直接使用当前顶点和邻居顶点之间的边权作为采样依据

                """ 1. 当前为首节点： 获取该首节点 所有邻居，然后根据 首节点 与邻居节点的权重，进行采样 一个邻居节点
                    2. 当前不为首节点： 则根据 上一轮节点 与 当前节点 的边， 进行采样下一个节点，具体细节
                    
                        1. 默认 p,q = 1 表示 deepwalk 节点与节点 进行随机采样，节点与节点的权重都是1
                        2. 引入 当前节点 v 表示， 上一轮节点 t 表示： 当前节点 v 的邻居节点 用 x 表示
                            
                            1. 针对 v返回t:  w * 1/p    p：默认是大于1 ，给定的是2： 表示尽量不要在返回自己，处于小圈子 活动
                            2. 针对 v 进入 x:
                                    1. x 没有与 t 连接，即 x 不是t 的邻居： w * 1/q : 默认q :0.5 表示 深度优先，尽量游走远方
                                    2. x 是t 的邻居： 则 w * 1  
                            详细看：https://time.geekbang.org/column/article/296672
                            p 和 q 共同控制着随机游走的倾向性。参数 p 被称为返回参数（Return Parameter），p 越小，随机游走回节点 t 的可能性越大，
                            Node2vec 就更注重表达网络的结构性。参数 q 被称为进出参数（In-out Parameter），
                            q 越小，随机游走到远方节点的可能性越大，Node2vec 更注重表达网络的同质性。反之，当前节点更可能在附近节点游走。
                    
                """
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0], alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        """
        """
        G = self.G
        nodes = list(G.nodes())
        # results = Parallel(n_jobs=workers, verbose=verbose, )(
        #     delayed(self._simulate_walks)(nodes, num, walk_length) for num in
        #     partition_num(num_walks, workers))

        # walks = list(itertools.chain(*results))

        walks = self._simulate_walks(nodes, num_walks, walk_length)

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, ):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                #print(_, v)
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(
                        walk_length=walk_length, start_node=v))
                else:
                    walks.append(self.node2vec_walk(
                        walk_length=walk_length, start_node=v))
        return walks

    def get_alias_edge(self, t, v):
        """
        1. 默认 p,q = 1 表示 deepwalk 节点与节点 进行随机采样，节点与节点的权重都是1
        2. 引入 当前节点 v 表示， 上一轮节点 t 表示： 当前节点 v 的邻居节点 用 x 表示

            1. 针对 v返回t:  w * 1/p    p：默认是大于1 ，给定的是2： 表示尽量不要在返回自己，处于小圈子 活动
            2. 针对 v 进入 x:
                    1. x 没有与 t 连接，即 x 不是t 的邻居： w * 1/q : 默认q :0.5 表示 深度优先，尽量游走远方
                    2. x 是t 的邻居： 则 w * 1
            详细看：https://time.geekbang.org/column/article/296672
            p 和 q 共同控制着随机游走的倾向性。参数 p 被称为返回参数（Return Parameter），p 越小，随机游走回节点 t 的可能性越大，
            Node2vec 就更注重表达网络的结构性。参数 q 被称为进出参数（In-out Parameter），
            q 越小，随机游走到远方节点的可能性越大，Node2vec 更注重表达网络的同质性。反之，当前节点更可能在附近节点游走。


        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.G
        p = self.p
        q = self.q
        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight / p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight / q)

        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        归一化权重
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        alias_nodes = {}

        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0) for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)
        alias_edges = {}

        # for edge in G.edges():
        #     alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        print(len(G.edges()))

        def paralize_func(edge):
            # print(edge)
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        pool_size = 20
        pool = ThreadPool(pool_size)  # 创建一个线程池
        pool.map(paralize_func, G.edges())  # 往线程池中填线程
        pool.close()  # 关闭线程池，不再接受线程
        pool.join()

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        return
