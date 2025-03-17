import logging

import torch
from torch.utils.data import DataLoader
from btown_ser import AudioDataset
from btown_ser import HubertBase
from btown_ser import TestingModel
from btown_ser import make_hubert_base
from types import SimpleNamespace

# Should come from the config
major_label_dict = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
batch_size = 18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger().setLevel(logging.INFO)
# Check for CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Current device:{}".format(device))


# Load datasets and make dataloaders
data_root_dir = "/home/ruthvik/speech_sentiment_classification/speech-emotion/\
cache_crema/downloads/extracted/bce0b52694cd7bd33785d8c3a052e940072ab6f3f29a8ce9151a2f326090bfa4/AudioWAV"
val_audio_data = AudioDataset(
    data_root_dir,
    max_audio_length=None,
    split="val",
    major_label_dict=major_label_dict,
    device=device,
)
logging.info("Length of val:{}".format(len(val_audio_data)))

val_loader = DataLoader(
    val_audio_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=AudioDataset.collate_fn_rnn,
)

# Should come from a config
existing_ckpt_path = "/home/ruthvik/speech_sentiment_classification/speech-emotion/notebooks/model_weights/FineTuningModel_epoch22.pt"
# Should come from a config
kwargs = {
    "PRE_TRAINED_HUBERT_BASE": "facebook/hubert-base-ls960",
    "NUM_LABELS": 6,
    "LOAD_EXISTING_CKPT": True,
    "EXISTING_CKPT_PATH": existing_ckpt_path,
    "NUM_CHARS_TO_SKIP": 7,
}
hubert_finetuned_model = make_hubert_base(**kwargs)
inference_model = TestingModel(hubert_finetuned_model, device="cuda")
validation_accuracy = inference_model.evaluate(val_loader)
