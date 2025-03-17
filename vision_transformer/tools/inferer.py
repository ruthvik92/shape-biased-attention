import sys

import torch
import torch.nn as nn
import numpy as np

from transformers import (
    Wav2Vec2FeatureExtractor,
)

# """
# Initialize the InferenceModel object.

# Parameters:
#    model (torch.nn.Module):
#    device (str):
# """


class InferenceModel:
    """
    A class for performing inference on a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to use for inference.
    device : str, optional
        cpu or gpu ? by default "cuda" iftorch.cuda.is_available() else "cpu"

    """

    def __init__(
        self,
        model: torch.nn.Module,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

    def model_init(self):
        """Separate init to move the model to GPU and set eval mode"""
        self.model = self.model.to(
            self.device
        )  # Move the model to the specified device
        self.model.eval()
        return

    def preprocess_wav(self, input_wav: np.ndarray) -> torch.FloatTensor:
        """Given a NumPy array of .wav file loaded using librosa, this
        method performs a feature extraction step.

        Parameters
        ----------
        input_wav : np.ndarray
            Input .wav

        Returns
        -------
        torch.FloatTensor
            Extracted features in Torch tensor format.
        """
        features = self.feature_extractor(
            input_wav, sampling_rate=16000, padding=True, return_tensors="pt"
        )
        audio_features = features.input_values  # .squeeze(0)
        return torch.FloatTensor(audio_features)

    def get_sentiment(self, input_wav: np.ndarray) -> torch.Tensor:
        """Given a .wav in NumPy format loaded using librosa, perform inference.

        Parameters
        ----------
        input_wav : np.ndarray
            Input .wav

        Returns
        -------
        torch.Tensor
            Torch tensor containing prediction classes.
        """
        audio_features = self.preprocess_wav(input_wav)
        audio_features = audio_features.to(self.device)

        # with torch.no_grad():
        outputs = self.model(audio_features, attention_mask=None)
        _, predicted = torch.max(outputs, dim=1)

        return predicted


# def make_pytorch_inferer(model, dataloader):
#    return
# will need a factory in future once the instantiation logic gets complex
