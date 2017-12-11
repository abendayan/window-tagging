import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TopK:
    def __init__(self, vector_file, vocab_file):
        self.vecs = np.loadtxt(vector_file)
        self.vocab = np.array(open(vocab_file, 'r').read().split('\n'))

    def index_in_vocab(self, word):
        index_arr,  = np.where(self.vocab==word)
        index = index_arr[0]
        return index

    def most_similar(self, word, k):
        index = self.index_in_vocab(word)
        len_vec = len(self.vecs[0])
        vector_word = self.vecs[index]
        vec_similars = [(-1, -float("inf"))] * k
        for i, vec in enumerate(self.vecs):
            if i != index:
                cos_sim = cosine_similarity(vector_word.reshape(1, -1), vec.reshape(1, -1))[0][0]
                for j, (index_vec, value) in enumerate(vec_similars):
                    if index_vec == -1:
                        vec_similars[j] = (i, cos_sim)
                        break
                    else:
                        if value < cos_sim:
                            vec_similars[j] = (i, cos_sim)
                            break
        vec_similars_words = []
        for vec in vec_similars:
            vec_similars_words.append(self.vocab[vec[0]])
        return vec_similars_words


if __name__ == '__main__':
    vector_file = sys.argv[1]
    vocab_file = sys.argv[2]
    top_k = TopK(vector_file, vocab_file)
    print "dog: "
    print top_k.most_similar("dog", 5)
    print "england: "
    print top_k.most_similar("england", 5)
    print "john: "
    print top_k.most_similar("john", 5)
    print "explode: "
    print top_k.most_similar("explode", 5)
    print "office: "
    print top_k.most_similar("office", 5)
