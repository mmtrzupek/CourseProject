import math
import sys
import time

import metapy
import pytoml

class BM25plus(metapy.index.RankingFunction):
    # Initialize BM25plus Ranker Function
    def __init__(self, k1, b, k3, free_parameter):

        self.k1 = k1    # k1 must be >= 0
        self.b = b      # b must be within [0,1]
        self.k3 = k3    # k3 must be >= 0 
        self.fp = free_parameter # Usually set to 1.0 by default
        
        super(BM25plus, self).__init__()

    def score_one(self, sd):

        IDF = math.log((1.0 + (sd.num_docs - sd.doc_count + 0.5) / (sd.doc_count + 0.5)), 2)
        TF = (((self.k1 + 1.0) * sd.doc_term_count) / ((self.k1 * ((1.0 - self.b) + self.b * sd.doc_size / sd.avg_dl)) + sd.doc_term_count)) + self.fp

        # Smoothing
        QTF = ((self.k3 + 1.0) * sd.query_term_weight) / (self.k3 + sd.query_term_weight)
        score = TF * IDF * QTF
        return score

def load_ranker(cfg_file):
    # Modified BM25+
    return BM25plus(1.8, 0.75, 500, 1.0)

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
    with open(query_path) as query_file:
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)
            avg_p = ev.avg_p(results, query_start + query_num, top_k)
            # print("Query {} average precision: {}".format(query_num + 1, avg_p))
            # print(avg_p)
    print("> Mean Average Precision: {}".format(ev.map()))
