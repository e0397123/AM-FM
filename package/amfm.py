import argparse
import logging
import numpy as np
import codecs
from scipy.stats import pearsonr
from scipy.stats import spearmanr


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--am_score", type=str, help="path to am score file")
parser.add_argument("--fm_score", type=str, help="path to fm score file")
parser.add_argument("--lambda_value", type=float, help="weight of am component")
parser.add_argument("--save_path", type=str, help="path to save the combined score")
args = parser.parse_args()


if __name__=='__main__':

    logging.info("Combining AM and FM scores -------------------------------------------------------")
    
    logging.info("Loading model scores -------------------------------------------------------------")
    with codecs.open(args.am_score, mode='r', encoding='utf-8') as rf:
        am_scores = rf.readlines()
    am_scores = [float(s.strip()) for s in am_scores]	
    
    with codecs.open(args.fm_score, mode='r', encoding='utf-8') as rf:
        fm_scores = rf.readlines()
    fm_scores = [float(s.strip()) for s in fm_scores]	

    score = [args.lambda_value * x + (1-args.lambda_value) * y for x, y in zip(am_scores, fm_scores)]
   
    with codecs.open(args.save_path, mode='w', encoding='utf-8') as wf:
        for s in score:
            wf.write(str(s) + '\n')
    logging.info("Done writing scores to {} -------------------------------------------------------------".format(args.save_path))    
