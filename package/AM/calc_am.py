import argparse
import logging
import string
import numpy as np
from numpy import linalg as LA
import jsonlines
import codecs
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--hyp_file", type=str, help="path to hypothesis file")
parser.add_argument("--ref_file", type=str, help="path to reference file")
parser.add_argument("--strategy", type=str, 
     help="am score computation strategy", default='top-layer-embedding-average')
parser.add_argument("--num_test", type=int, help="total number of test cases")
parser.add_argument("--save_path", type=str, help="path to save system level score")
args = parser.parse_args()

def calc_am_single(hyp_list, ref_list):
    score_mat = cosine_similarity(np.array(hyp_list), np.array(ref_list))
    return score_mat.diagonal()

def calc_am_batch(hyp_list, ref_list):
    score_mat = cosine_similarity(np.array(hyp_list), np.array(ref_list))
    score_mat = np.amax(score_mat, axis=1).T
    return score_mat

def absmaxND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)
    

if __name__=='__main__':

    logging.info("Loading hypothesis features -------------------------------------------------------")
    hyp_sent_embedding = []
    tq = tqdm(total=args.num_test)
    with jsonlines.open(args.hyp_file) as reader:
        for obj in reader:
            if args.strategy == 'top-layer-embedding-average':
                # embedding average
                obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.sum(obj_emb, axis=0)
                obj_emb =  obj_emb / LA.norm(obj_emb)
            elif args.strategy == 'top-layer-max-pool':
                # top layer max pooling
                obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.max(obj_emb, axis=0)
            elif args.strategy == 'top-layer-mean-pool':
                # top layer mean pooling
                obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.mean(obj_emb, axis=0)
            elif args.strategy == 'top-layer-concat-vector-extrema':
                 # top-layer-vector-extrema
                 obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                 obj_emb = absmaxND(obj_emb, axis=0)
            hyp_sent_embedding.append(obj_emb)
            tq.update(1)
    tq.close()	
    logging.info("Done loading hypothesis features ---------------------------------------------------")

    assert len(hyp_sent_embedding) == args.num_test, "wrong number of hypotheses, please check the number of hypotheses"
    logging.info("Loading references features --------------------------------------------------------")
    ref_sent_embedding = []
    tq = tqdm(total=args.num_test)
    with jsonlines.open(args.ref_file) as reader:
        for obj in reader:
            if args.strategy == 'top-layer-embedding-average':
                # embedding average
                obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.sum(obj_emb, axis=0)
                obj_emb =  obj_emb / LA.norm(obj_emb)
            elif args.strategy == 'top-layer-max-pool':
                # top layer max pooling
                obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.max(obj_emb, axis=0)
            elif args.strategy == 'top-layer-mean-pool':
                # top layer mean pooling
                obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.mean(obj_emb, axis=0)
            elif args.strategy == 'top-layer-concat-vector-extrema':
                 # top-layer-vector-extrema
                 obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                 obj_emb = absmaxND(obj_emb, axis=0)
            ref_sent_embedding.append(obj_emb)
            tq.update(1)
    tq.close()
    logging.info("Done loading references features ---------------------------------------------------")

    assert len(ref_sent_embedding) == args.num_test, "wrong number of inferences, please check the number of references"
    
    # calculate AM score
    logging.info("Computing am score ---------------------------------------------------")
    full_scores = []
    hyp_arr = np.array(hyp_sent_embedding)
    ref_arr = np.array(ref_sent_embedding)
    scores = calc_am_single(hyp_arr, ref_arr).tolist()

    with codecs.open(args.save_path, mode='w', encoding='utf-8') as wf:
        for s in scores:
            wf.write(str(s) + '\n')
    logging.info("Done saving am score to {} ---------------------------------------------------".format(args.save_path))
    
    
