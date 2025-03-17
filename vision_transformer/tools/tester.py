import torch
import torch.nn as nn
import numpy as np


class TestingModel:
    """
    A class for performing inference on a PyTorch model using a given DataLoader.

    Parameters:
        model (torch.nn.Module): The PyTorch model to use for inference.
        device (str): The device to use for inference ('cuda' or 'cpu').

    Attributes:
        model (torch.nn.Module): The PyTorch model used for inference.
        device (str): The device used for inference ('cuda' or 'cpu').

    Example Usage:
        # Assuming you have already defined your model and created the DataLoader called 'data_loader'
        # model = YourModelClass()  # Replace YourModelClass with your actual model class
        # inference_model = InferenceModel(model, device='cuda')
        # validation_accuracy = inference_model.evaluate(data_loader)
    """

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the InferenceModel object.

        Parameters:
            model (torch.nn.Module): The PyTorch model to use for inference.
            device (str): The device to use for inference ('cuda' or 'cpu').
        """
        self.model = model
        self.device = device

    def evaluate(self, data_loader):
        """
        Perform inference on the given DataLoader and calculate validation accuracy.

        Parameters:
            data_loader (torch.utils.data.DataLoader): The DataLoader containing the validation data.

        Returns:
            float: The validation accuracy.
        """
        self.model = self.model.to(
            self.device
        )  # Move the model to the specified device
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                input_values = batch["input_values"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                # some times you may not be able to load data all at once.

                outputs = self.model(input_values, attention_mask=attention_mask).logits
                _, predicted = torch.max(outputs, dim=1)

                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / total_samples
        print("Validation Accuracy: {}".format(accuracy))
        print("Total Number of Data Points: {}".format(total_samples))
        print("Total Number of Correctly Identified Classes: {}".format(total_correct))

        return accuracy


# Example Usage:
# Assuming you have already defined your model and created the DataLoader called 'data_loader'
# model = YourModelClass()  # Replace YourModelClass with your actual model class
# inference_model = InferenceModel(model, device='cuda')
# validation_accuracy = inference_model.evaluate(data_loader)
