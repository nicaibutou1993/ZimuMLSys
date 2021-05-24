# -*- coding: utf-8 -*-
from zimu_ml_sys.core.graph_embedding.random_walker import RandomWalker
from gensim.models.word2vec import Word2Vec


class Node2Vec:

    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1):

        self.graph = graph
        self._embeddings = {}

        self.walker = RandomWalker(graph, p=p, q=q, )
        self.walker.preprocess_transition_probs()
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, epochs=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = epochs

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")
        self.w2v_model = model

        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings

    def get_topK(self, item, k=50):
        if not isinstance(item, str):
            item = str(item)
        recom_list = list(map(lambda x: [x[0], x[1]], self.w2v_model.wv.most_similar(positive=[item], topn=k)))
        return recom_list
