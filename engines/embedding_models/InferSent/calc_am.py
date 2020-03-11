import argparse
import logging
import string
import numpy as np
from numpy import linalg as LA
import jsonlines
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--hyp_file", type=str, help="path to hypothesis file")
parser.add_argument("--ref_file", type=str, help="path to reference file")
args = parser.parse_args()


def calc_am_batch(hyp_list, ref_list):
    score_mat = cosine_similarity(np.array(hyp_list), np.array(ref_list))
    score_mat = np.amax(score_mat, axis=1).T
    return score_mat


def absmaxND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)


if __name__ == '__main__':

    logging.info("Loading hypothesis features -------------------------------------------------------")
    hyp_sent_embedding = np.load(args.hyp_file)
    logging.info("Loading references features --------------------------------------------------------")
    ref_sent_embedding = np.load(args.ref_file)
    logging.info("Done loading references features ---------------------------------------------------")

    logging.info("rearranging test cases -------------------------------------------------------------")
    hyp_per_all_sys = []
    hyp_per_sys = []
    for i, line in enumerate(tqdm(hyp_sent_embedding)):
        hyp_per_sys.append(line)
        if (i + 1) % 2000 == 0:
            hyp_per_all_sys.append(hyp_per_sys)
            hyp_per_sys = []

    hyp_per_dialogues = []
    hyp_per_single_dialogue = []
    for i, item in enumerate(hyp_per_all_sys[0]):
        hyp_per_single_dialogue.append(item)
        hyp_per_single_dialogue.append(hyp_per_all_sys[1][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[2][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[3][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[4][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[5][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[6][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[7][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[8][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[9][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[10][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[11][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[12][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[13][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[14][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[15][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[16][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[17][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[18][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[19][i])
        hyp_per_dialogues.append(hyp_per_single_dialogue)
        hyp_per_single_dialogue = []

    assert len(hyp_per_dialogues) == 2000, 'number of hypothesis test cases not equal to 2000'

    ref_per_all_sys = []
    ref_per_sys = []
    for i, line in enumerate(tqdm(ref_sent_embedding)):
        ref_per_sys.append(line)
        if (i + 1) % 2000 == 0:
            ref_per_all_sys.append(ref_per_sys)
            ref_per_sys = []

    ref_per_dialogues = []
    ref_per_single_dialogue = []
    for i, item in enumerate(ref_per_all_sys[0]):
        ref_per_single_dialogue.append(item)
        ref_per_single_dialogue.append(ref_per_all_sys[1][i])
        ref_per_single_dialogue.append(ref_per_all_sys[2][i])
        ref_per_single_dialogue.append(ref_per_all_sys[3][i])
        ref_per_single_dialogue.append(ref_per_all_sys[4][i])
        ref_per_single_dialogue.append(ref_per_all_sys[5][i])
        ref_per_single_dialogue.append(ref_per_all_sys[6][i])
        ref_per_single_dialogue.append(ref_per_all_sys[7][i])
        ref_per_single_dialogue.append(ref_per_all_sys[8][i])
        ref_per_single_dialogue.append(ref_per_all_sys[9][i])
        ref_per_single_dialogue.append(ref_per_all_sys[10][i])
        ref_per_dialogues.append(ref_per_single_dialogue)
        ref_per_single_dialogue = []

    assert len(ref_per_dialogues) == 2000, 'number of references test cases not equal to 2000'
    logging.info("Done rearranging test cases --------------------------------------------------------")

    # calculate AM score
    full_scores = []
    for hyp, ref in zip(hyp_per_dialogues, ref_per_dialogues):
        scores = calc_am_batch(hyp, ref)
        full_scores.append(scores)
    full_scores = np.array(full_scores)
    system_level_scores = np.mean(full_scores, axis=0)
    logging.info('The final system level scores:')
    for score in system_level_scores.tolist():
        print(score)
