import codecs
import argparse
import logging
import twokenize
import random
import emoji
from tqdm import tqdm

random.seed(1234)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, help="path to training file")
parser.add_argument("--valid_file", type=str, help="path to validation file")
parser.add_argument("--train_output", type=str, help="path to clean training output file")
parser.add_argument("--valid_output", type=str, help="path to clean valid output file")
parser.add_argument("--data_size", type=int, help="training data size")
args = parser.parse_args()

chat_words_str = """
AFAIK=As Far As I Know
AFK=Away From Keyboard
ASAP=As Soon As Possible
ATK=At The Keyboard
ATM=At The Moment
A3=Anytime, Anywhere, Anyplace
BAK=Back At Keyboard
BBL=Be Back Later
BBS=Be Back Soon
BFN=Bye For Now
B4N=Bye For Now
BRB=Be Right Back
BRT=Be Right There
BTW=By The Way
B4=Before
B4N=Bye For Now
CU=See You
CUL8R=See You Later
CYA=See You
FAQ=Frequently Asked Questions
FC=Fingers Crossed
FWIW=For What It's Worth
FYI=For Your Information
GAL=Get A Life
GG=Good Game
GN=Good Night
GMTA=Great Minds Think Alike
GR8=Great!
G9=Genius
IC=I See
ICQ=I Seek you (also a chat program)
ILU=ILU: I Love You
IMHO=In My Honest/Humble Opinion
IMO=In My Opinion
IOW=In Other Words
IRL=In Real Life
KISS=Keep It Simple, Stupid
LDR=Long Distance Relationship
LMAO=Laugh My A.. Off
LOL=Laughing Out Loud
LTNS=Long Time No See
L8R=Later
MTE=My Thoughts Exactly
M8=Mate
NRN=No Reply Necessary
OIC=Oh I See
PITA=Pain In The A..
PRT=Party
PRW=Parents Are Watching
ROFL=Rolling On The Floor Laughing
ROFLOL=Rolling On The Floor Laughing Out Loud
ROTFLMAO=Rolling On The Floor Laughing My A.. Off
SK8=Skate
STATS=Your sex and age
ASL=Age, Sex, Location
THX=Thank You
TTFN=Ta-Ta For Now!
TTYL=Talk To You Later
U=You
U2=You Too
U4E=Yours For Ever
WB=Welcome Back
WTF=What The F...
WTG=Way To Go!
WUF=Where Are You From?
W8=Wait...
7K=Sick:-D Laugher
"""

chat_words_map_dict = {}
chat_words_list = []
for line in chat_words_str.split("\n"):
    if line != "":
        cw = line.split("=")[0]
        cw_expanded = line.split("=")[1]
        chat_words_list.append(cw)
        chat_words_map_dict[cw] = cw_expanded
chat_words_list = set(chat_words_list)

def chat_words_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

def convert_emojis(text):
    return emoji.demojize(text)
	
if __name__=='__main__':
    logging.info("----------------procesing training set ------------------------------")
    with codecs.open(args.train_file, encoding='utf-8', mode='r') as rf:
        train_lines = rf.readlines()
    logging.info("----------------select training set ---------------------------------")
    train_dialogues = []
    single_dialogue = []
    for line in tqdm(train_lines):
        if line.strip():
            single_dialogue.append(line.strip())
        else:
            train_dialogues.append(single_dialogue)
            single_dialogue = []
    selected_dialogues = random.choices(train_dialogues, k=args.data_size)
    selected_lines = []
    for dialogue in tqdm(selected_dialogues):
        selected_lines.extend(dialogue)
        selected_lines.append('')    
    logging.info("----------------step 1. chatword conversion----- --------------------")
    selected_lines = [chat_words_conversion(text[3:]) if text else '' for text in tqdm(selected_lines)]
    logging.info("----------------step 2. emoji replacement ---------------------------")
    selected_lines = [convert_emojis(text) if text else '' for text in tqdm(selected_lines)]
    logging.info("----------------step 3. cleaning up the text ---------------------------")
    selected_lines = [' '.join(twokenize.tokenize(text)) if text else '' for text in tqdm(selected_lines)]  
    logging.info("----------------step 4. write data to file  ---------------------------")
    with codecs.open(args.train_output, encoding='utf-8', mode='w') as wf:
        wf.truncate()
    for item in tqdm(selected_lines):
        with codecs.open(args.train_output, encoding='utf-8', mode='a') as wf:
            wf.write(item + '\n')
    logging.info("----------------Done processing training set ------------------------")


    logging.info("----------------procesing valid set ------------------------------")
    with codecs.open(args.valid_file, encoding='utf-8', mode='r') as rf:
        valid_lines = rf.readlines() 
    logging.info("----------------step 1. chatword conversion----- --------------------")
    valid_lines = [chat_words_conversion(text[3:]) if text else '' for text in tqdm(valid_lines)]
    logging.info("----------------step 2. emoji replacement ---------------------------")
    valid_lines = [convert_emojis(text) if text else '' for text in tqdm(valid_lines)]
    logging.info("----------------step 3. cleaning up the text ---------------------------")
    valid_lines = [' '.join(twokenize.tokenize(text)) if text else '' for text in tqdm(valid_lines)]  
    logging.info("----------------step 4. write data to file  ---------------------------")
    with codecs.open(args.valid_output, encoding='utf-8', mode='w') as wf:
        wf.truncate()
    for item in tqdm(valid_lines):
        with codecs.open(args.valid_output, encoding='utf-8', mode='a') as wf:
             wf.write(item + '\n')           
    logging.info("----------------Done processing training set ------------------------")



