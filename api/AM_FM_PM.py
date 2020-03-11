import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .dynamic_import import dynamic_import
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags = tf.flags
flags.DEFINE_string('lm_model', "rnnlm", 'language model name')
flags.DEFINE_string('lm_dataset', "dstc6", 'dataset name which the lm trained on it')
flags.DEFINE_string('do_dstc_eval', "bert", 'sentence encoder model name')
flags.DEFINE_string('se_model', "dstc6", 'dataset name which the se trained on it')

args = flags.FLAGS
args.mark_as_parsed()

model_flags = {
    "bert": {"init_checkpoint": None,
             "bert_config_file": "/media/datadrive/PycharmProject/Singapore/A_F_PM/"
                                 "model_pretrain/bert/uncased_L-12_H-768_A-12/bert_config.json",
             "vocab_file": "/media/datadrive/PycharmProject/Singapore/A_F_PM/"
                           "model_pretrain/bert/uncased_L-12_H-768_A-12/vocab.txt",
             },
    "rnnlm": {
        "model_path": "/media/datadrive/PycharmProject/Singapore/A_F_PM/AM_FM_PM/examples/{}/pretrain/{}/",
        "tokenizer_path": "/media/datadrive/PycharmProject/Singapore/A_F_PM/AM_FM_PM/"
                          "examples/dstc6/pretrain/sentencepiece/twitter_10000.model",
        "bpe_vocab": "/media/datadrive/PycharmProject/Singapore/A_F_PM/AM_FM_PM/"
                     "examples/dstc6/pretrain/sentencepiece/twitter_10000.vocab"
    }
}


# ###Delete all flags before declare#####

def del_all_flags():
    flags_ = tf.flags.FLAGS
    flags_dict = flags_._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        flags_.__delattr__(keys)


def load_language_model(lm="language_model_name", dataset="dataset_name"):
    # return lm model

    model_class_path = "AM_FM_PM.engines.language_models.{}.api:EncodeModel".format(lm)
    model_class = dynamic_import(model_class_path)

    kwargs = model_flags[lm]
    model_dir = kwargs["model_path"].format(dataset, lm)
    kwargs["model_path"] = model_dir

    return model_class(kwargs=kwargs)


def load_se_model(se="sentence_encoder_name", dataset="dataset_name"):
    # return se model
    model_dir = "/media/datadrive/PycharmProject/Singapore/A_F_PM/AM_FM_PM/examples/{}/pretrain/{}".format(dataset, se)
    model_class_path = "AM_FM_PM.engines.embedding_models.{}.api:EncodeModel".format(se)
    model_class = dynamic_import(model_class_path)

    kwargs = model_flags[se]
    kwargs["init_checkpoint"] = model_dir

    return model_class(kwargs=kwargs)


class AmFmEvaluationModel:
    def __init__(self, lm_model, se_model, lm_dataset, se_dataset, combine_f='average'):
        del_all_flags()

        lm_model_graph = tf.Graph()
        with lm_model_graph.as_default():
            self.lm_model = load_language_model(lm_model, lm_dataset)

        del_all_flags()

        se_model_graph = tf.Graph()
        with se_model_graph.as_default():
            self.se_model = load_se_model(se_model, se_dataset)

        del_all_flags()

        if combine_f == 'average':
            self.combine = self.average

        print("AMFM model loading completed !!!!!!!!")

    def test(self):
        a = "hello"
        b = "how are you"
        c = "good morning"

        output = self.cal_am_fm([a, b, c], [c, a, b, c, a])
        print(output)
        print("AM_FM_PM.py test Done")

    def lm_encode(self, sentences):
        return self.lm_model.encode(sentences)

    def se_encode(self, sentences):
        return self.se_model.encode(sentences)

    def cal_am(self, sentences1, sentences2):
        embedding1 = self.se_encode(sentences1)
        embedding2 = self.se_encode(sentences2)

        return self.calc_am_batch(embedding1, embedding2)

    def cal_fm(self, sentences1, sentences2):
        ppl1 = self.lm_encode(sentences1)
        ppl2 = self.lm_encode(sentences2)

        return self.calc_fm_batch(ppl1, ppl2)

    def cal_am_fm(self, sentences1, sentences2):
        am = self.cal_am(sentences1, sentences2)
        fm = self.cal_fm(sentences1, sentences2)
        return self.combine(am, fm)

    @staticmethod
    def average(a, b):
        return (a + b) / 2

    @classmethod
    def calc_am_batch(cls, hyp_list, ref_list):

        score_mat = cosine_similarity(np.array(hyp_list), np.array(ref_list))
        score_mat = np.amax(score_mat, axis=1).T
        return score_mat

    @classmethod
    def calc_fm_batch(cls, hyp_list, ref_list):
        per_sys_score = []
        for hyp in hyp_list:
            temp = []
            for ref in ref_list:
                temp.append(cls.calc_fm(hyp, ref))
            per_sys_score.append(np.amax(temp) - np.amin(temp))
            # per_sys_score.append(np.amax(temp))
            # per_sys_score.append(np.amin(temp))
            # per_sys_score.append(np.mean(temp))
        return np.asarray(per_sys_score)

    @classmethod
    def calc_fm(cls, hyp, ref):
        return min(1 / hyp, 1 / ref) / max(1 / hyp, 1 / ref)


if __name__ == '__main__':
    eval_model = AmFmEvaluationModel(args.lm_model, args.se_model, args.lm_dataset, args.se_dataset)
    eval_model.test()
