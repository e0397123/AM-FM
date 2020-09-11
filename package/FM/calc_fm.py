import argparse
import logging
import string
import codecs
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--hyp_file", type=str, help="path to hypothesis file")
parser.add_argument("--ref_file", type=str, help="path to reference file")
parser.add_argument("--num_test", type=int, help="number of test cases")
parser.add_argument("--save_path",  type=str, help='path to save system level fm score')
args = parser.parse_args()

def calc_fm(hyp, ref):
    return min(1/hyp, 1/ref)/max(1/hyp, 1/ref)

if __name__=='__main__':

    logging.info("Reading hypothesis perplexity -------------------------------------------------------")
    hyp_sent_ppl = []
    with codecs.open(args.hyp_file, mode='r', encoding='utf-8') as rf:
        for line in rf.readlines():
            hyp_sent_ppl.append(float(line.strip()))

    assert len(hyp_sent_ppl) == args.num_test, 'number of hypotheses not equal to {}'.format(args.num_test)
    
    logging.info("Reading references perplexity -------------------------------------------------------")
    ref_sent_ppl = []
    with codecs.open(args.ref_file, mode='r', encoding='utf-8') as rf:
        for line in rf.readlines():
            ref_sent_ppl.append(float(line.strip()))

    assert len(ref_sent_ppl) == args.num_test, 'number of references not equal to {}'.format(args.num_test)

    logging.info("computing FM scores -------------------------------------------------------")
    # calculate FM score
    full_scores = []
    for hyp, ref in zip(hyp_sent_ppl, ref_sent_ppl):
        score = calc_fm(hyp, ref)
        full_scores.append(score)
    
    with codecs.open(args.save_path, mode='w', encoding='utf-8') as wf:
        for score in full_scores:
            wf.write(str(score) + '\n') 

    logging.info("Done writing FM scores to {}-------------------------------------------------------".format(args.save_path)) 
