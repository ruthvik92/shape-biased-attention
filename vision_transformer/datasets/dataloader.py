# Define your custom audio dataset class
import glob
import os
import sys
from typing import Dict
import logging


import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import librosa
from transformers import (
    Wav2Vec2FeatureExtractor,
)
from sklearn.model_selection import train_test_split
from yacs.config import CfgNode
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

logging.getLogger().setLevel(logging.INFO)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root_dir,
        max_audio_length=None,
        major_label_dict: Dict = {},
        split="train",
        device="cuda",
    ):
        self.root_dir = data_root_dir
        self.major_label_dict = major_label_dict
        self.device = device
        # self.audio_paths = os.listdir(data_root_dir)
        self.audio_paths = glob.glob(self.root_dir + "/*.wav")
        self.max_audio_length = max_audio_length
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        # self.feature_extractor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        # don't need a processor because it also has a tokenizer.. if you are training speech to text then
        # you'll need a tokenizer
        # Perform train-test-validation split
        train_files, test_files = train_test_split(
            self.audio_paths, test_size=0.2, random_state=42
        )
        train_files, val_files = train_test_split(
            train_files, test_size=0.1, random_state=42
        )

        if split == "train":
            self.audio_paths = train_files
        elif split == "test":
            self.audio_paths = test_files
        elif split == "val":
            self.audio_paths = val_files
        else:
            raise ValueError("Invalid split. Must be 'train', 'test', or 'val'.")

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        file_name = os.path.basename(audio_path)
        label = self.major_label_dict[self.get_label(file_name)]

        # audio_features = self.process_audio(audio_path, self.max_audio_length)
        audio_features = self.process_audio(audio_path)

        return {
            "input_values": audio_features,  # .to(self.device),
            "attention_mask": torch.ones(audio_features.shape[0]),  # .to(self.device),
            "label": torch.tensor(label, dtype=torch.long),  # .to(self.device),
        }
        # some times you may not be able to load data all at once, so you might have
        # to remove the .to(self.device) and do it for each batch in the train/eval loops

    # def process_audio(self, audio_path, max_audio_length):
    #     #print(audio_path)
    #     audio, _ = librosa.load(audio_path, sr=16000, mono=False)  # Assuming sample rate of 16kHz
    #     if len(audio) > max_audio_length:
    #         audio = audio[:max_audio_length]
    #     else:
    #         audio = librosa.util.pad_center(audio, size=max_audio_length)
    #     return torch.FloatTensor(audio)
    def process_audio(self, audio_path):
        audio, _ = librosa.load(
            audio_path, sr=16000, mono=False
        )  # Assuming sample rate of 16kHz
        features = self.feature_extractor(
            audio, sampling_rate=16000, padding=True, return_tensors="pt"
        )
        input_values = features.input_values.squeeze(0)
        return torch.FloatTensor(input_values)

    def get_label(self, file_name):
        # Extract the label from the file name
        label = file_name.split("_")[2]
        return label

    @staticmethod
    def collate_fn(batch):
        # Find the maximum audio length in the batch
        max_audio_length = max(len(item["input_values"]) for item in batch)
        # Pad the audio sequences to the maximum length
        for item in batch:
            audio_length = len(item["input_values"])
            padding_length = max_audio_length - audio_length
            item["input_values"] = F.pad(item["input_values"], (0, padding_length))
            item["attention_mask"] = F.pad(item["attention_mask"], (0, padding_length))

        # Stack the input_values, attention_mask, and label tensors
        input_values = torch.stack([item["input_values"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "label": labels,
        }

    @staticmethod
    def collate_fn_rnn(batch):
        # Sort the batch based on the length of the input sequences
        sorted_batch = sorted(batch, key=lambda x: len(x["input_values"]), reverse=True)

        # Pad the audio sequences to the maximum length
        input_values = pad_sequence(
            [item["input_values"] for item in sorted_batch], batch_first=True
        )

        # Pad the attention masks to match the input sequence lengths
        attention_mask = pad_sequence(
            [item["attention_mask"] for item in sorted_batch], batch_first=True
        )

        # Stack the labels
        labels = torch.stack([item["label"] for item in sorted_batch])

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "label": labels,
        }


# def make_audio_dataset(
#    root_dir,
#    max_audio_length,
#    split,
#    major_label_dict,
#    device,
# ):
#    audio_data = AudioDataset(
#        root_dir,
#        max_audio_length,
#        split,
#        major_label_dict,
#        device,
#    )
#    return audio_data


def make_data_loaders(cfg: CfgNode, device="cuda", val_shuffle=True):
    train_audio_data = AudioDataset(
        cfg.DATASET.ROOT_DIR,
        max_audio_length=cfg.DATASET.MAX_AUDIO_LENGTH,
        split="train",
        major_label_dict=dict(cfg.DATASET.LABEL_DICT),
        device=device,
    )
    test_audio_data = AudioDataset(
        cfg.DATASET.ROOT_DIR,
        max_audio_length=cfg.DATASET.MAX_AUDIO_LENGTH,
        split="test",
        major_label_dict=dict(cfg.DATASET.LABEL_DICT),
        device=device,
    )
    val_audio_data = AudioDataset(
        cfg.DATASET.ROOT_DIR,
        max_audio_length=cfg.DATASET.MAX_AUDIO_LENGTH,
        split="val",
        major_label_dict=dict(cfg.DATASET.LABEL_DICT),
        device=device,
    )
    logging.info(
        "Length of train:{}, test:{}, val:{}".format(
            len(train_audio_data), len(test_audio_data), len(val_audio_data)
        )
    )
    if cfg.DATASET.MERGE_TRAIN_TEST:
        train_audio_data = ConcatDataset([train_audio_data, test_audio_data])
        logging.warn(
            "Train and Test datasets were merged, use Val set for the hold out"
        )
    train_loader = DataLoader(
        train_audio_data,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=True,
        collate_fn=AudioDataset.collate_fn_rnn,
    )
    test_loader = DataLoader(
        train_audio_data,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=True,
        collate_fn=AudioDataset.collate_fn_rnn,
    )
    val_loader = DataLoader(
        val_audio_data,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=val_shuffle,
        collate_fn=AudioDataset.collate_fn_rnn,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_root_dir = "/home/ruthvik/speech_sentiment_classification/speech-emotion/\
    cache_crema/downloads/extracted/bce0b52694cd7bd33785d8c3a052e940072ab6f3f29a8ce9151a2f326090bfa4/AudioWAV"
    batch_size = 18
    train_audio_data = AudioDataset(data_root_dir, max_audio_length=None, split="train")
    test_audio_data = AudioDataset(data_root_dir, max_audio_length=None, split="test")
    val_audio_data = AudioDataset(data_root_dir, max_audio_length=None, split="val")
    print(
        "Length of train:{}, test:{}, val:{}".format(
            len(train_audio_data), len(test_audio_data), len(val_audio_data)
        )
    )
