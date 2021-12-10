import math
import sys
import time

import metapy
import pytoml

class BM25L(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA.
    """
    def __init__(self, k1, b, k3, free_parameter):
        #self.param = some_param

        self.k1 = k1    # k1 must be >= 0
        self.b = b      # b must be within [0,1]
        self.k3 = k3    # k3 must be >= 0 

        self.fp = free_parameter # Usually set to 0.5 for best performance

        super(BM25L, self).__init__()

    def score_one(self, sd):
        # https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html

        IDF = math.log((sd.num_docs + 1.0)/(sd.doc_count + 0.5), 2)
        ctd = sd.doc_term_count / ((1.0 - self.b) + self.b * (sd.doc_size / sd.avg_dl))
        TF = ((self.k1 + 1.0) * (ctd + self.fp)) / (self.k1 + ctd + self.fp)

        # Smoothing
        QTF = ((self.k3 + 1.0) * sd.query_term_weight) / (self.k3 + sd.query_term_weight)

        score = TF * IDF * QTF
        return score

def load_ranker(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, e.g. return InL2Ranker(some_param=1.0) 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index. You can ignore this for MP2.
    """
    # Modified BM25L
    return BM25L(1.2, 0.75, 500, 0.5)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    print('> Creating Inverted Index')
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker(cfg)
    ev = metapy.index.IREval(cfg)

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    start_time = time.time()
    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)
    
    # Inverted Index Construction
    print("> Inverted Index Construction Successful.")

    query = metapy.index.Document()
    #print('Running Queries')
    with open(query_path) as query_file:
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)
            avg_p = ev.avg_p(results, query_start + query_num, top_k)
            #print("Query {} average precision: {}".format(query_num + 1, avg_p))
            print(avg_p)
    print("> Mean Average Precision: {}".format(ev.map()))
    #print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
