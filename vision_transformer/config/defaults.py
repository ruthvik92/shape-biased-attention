import os
from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.PRE_TRAINED_HUBERT_BASE = "facebook/hubert-base-ls960"
_C.MODEL.NUM_LABELS = 6
_C.MODEL.LOAD_EXISTING_CKPT = False
_C.MODEL.EXISTING_CKPT_PATH = ""
_C.MODEL.NUM_CHARS_TO_SKIP = 7
_C.MODEL.UNFREEZE_LAST_N_LAYERS = 10
_C.MODEL.CHECKPOINT_DIR = "../runs"
# Each of the 11 `HubertEncoderLayer` 16 layers inside, -10 indicates
# that we will be training all the layers except the first.
# See the following notebook for the extra input layers that this model has
# /notebooks/speech_sentiment_classification/
# btown-speech-emotion-recognition/notebooks/trainer_testing.ipynb

_C.DATASET = CN()
_C.DATASET.ROOT_DIR = ""
# _C.DATASET.LABEL_DICT = [{"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}]
_C.DATASET.LABEL_DICT = CN({"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5})
_C.DATASET.MERGE_TRAIN_TEST = True  # if true merges train, test... NOT val
_C.DATASET.MAX_AUDIO_LENGTH = None

_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 30
_C.SOLVER.BASE_LR = 2e-5
_C.SOLVER.WARMUP_STEPS = 100
_C.SOLVER.BATCH_SIZE = 18

_C.INFERENCE = CN()
_C.INFERENCE.CHECKPOINT = ""  # a run that needs to be tested
_C.INFERENCE.INPUT_DATA = ""  # location of the input *.wav files (no labels needed)

_C.TESTING = CN()  # will work on it soon
