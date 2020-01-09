import codecs
import argparse
import logging
import operator
from collections import defaultdict
from tqdm import tqdm


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, help="path to training file")
parser.add_argument("--vocab_file", type=str, help="path to vocab file")
args = parser.parse_args()

train_lines = []
with codecs.open(args.train_file, mode='r', encoding='utf-8') as rf:
    for line in rf.readlines():
        if line.strip():
            train_lines.append(line)

vocab = defaultdict(int)

for line in tqdm(train_lines):
    splitted_line = line.split()
    for token in splitted_line:
        vocab[token] += 1

sorted_d = dict(sorted(vocab.items(), key=operator.itemgetter(0),reverse=True))

logging.info("sorted vocab in descending order: ")
for i, (k, v) in enumerate(sorted_d.items()):
    logging.info("{} : {}".format(k, v))
    if i == 5:
        break

with codecs.open(args.vocab_file, mode='w', encoding='utf-8') as wf:
    wf.truncate()

for k in sorted_d.keys():
    with codecs.open(args.vocab_file, mode='a', encoding='utf-8') as wf:
        wf.write(k + '\n')
        
