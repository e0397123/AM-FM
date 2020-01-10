import glob
import codecs
import subprocess
from tqdm import tqdm

f_list = glob.glob('../../data/twitter/dstc6_t2_evaluation/hypotheses/hyp/x*')
f_list.sort()

with codecs.open('hyp_ppl.txt', encoding='utf-8', mode='w') as wf:
    wf.truncate()

for f in tqdm(f_list):
    output = subprocess.check_output("ngram -ppl " + f + " -order 4 -lm twitter_100k.lm", shell=True)
    with codecs.open('hyp_ppl.txt', encoding='utf-8', mode='a') as wf:
        wf.write(output.decode("utf-8").split('ppl= ')[1].split(' ppl1')[0] + '\n')

