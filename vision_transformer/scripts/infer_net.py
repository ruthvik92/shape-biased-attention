from pathlib import Path
import os
import sys
import logging
import argparse
import glob


import torch
import torch.nn as nn
import librosa
from tqdm import tqdm
from yacs.config import CfgNode

from btown_ser import make_finetuning_hubert
from btown_ser import InferenceModel
from btown_ser.config import cfg

logging.getLogger().setLevel(logging.INFO)
# Check for CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Current device:{}".format(device))


def do_infer(cfg: CfgNode):
    logging.info("Picked up the checkpoint from : {}".format(cfg.INFERENCE.CHECKPOINT))
    logging.info("Picked up Input data from : {}".format(cfg.INFERENCE.INPUT_DATA))
    cfg_labels = cfg.DATASET.LABEL_DICT
    cfg_labels = {val: key for key, val in cfg_labels.items()}
    logging.info("Labels against which this model was trained : {}".format(cfg_labels))

    audio_paths = glob.glob(cfg.INFERENCE.INPUT_DATA + "/*.wav")
    fine_tuned_model = make_finetuning_hubert(cfg=cfg)
    inference_model = InferenceModel(model=fine_tuned_model)
    inference_model.model_init()
    # pbar = tqdm(
    #    audio_paths,
    #    ncols=100,
    #    unit="files",
    # )
    for audio_path in audio_paths:
        logging.info(
            "Base filename / label is : {}".format(os.path.basename(audio_path))
        )
        audio, _ = librosa.load(
            audio_path, sr=16000, mono=False
        )  # Assuming sample rate of 16kHz
        predicted_emotion = inference_model.get_sentiment(input_wav=audio).item()
        logging.info("Predicted emotion is : {}".format(cfg_labels[predicted_emotion]))
        logging.info("\n")

    return


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "config_file",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    print(args.opts)
    # sys.exit()
    if args.opts != None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    do_infer(cfg=cfg)


if __name__ == "__main__":
    main()
