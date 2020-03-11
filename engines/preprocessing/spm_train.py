
import sys
import sentencepiece as spm


if __name__ == '__main__':
    argument = ' '.join(sys.argv[1:])
    spm.SentencePieceTrainer.Train(argument)
