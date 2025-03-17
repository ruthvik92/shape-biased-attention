import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup

from btown_ser.utils import CheckpointSaver


class PyTorchModelTrainer:
    """
    A class for training a PyTorch model using tqdm to show progress and losses.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        criterion (torch.nn.Module): The loss function to be used for training.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        device (str, optional): The device to use for training (default is 'cuda').

    Attributes:
        model (torch.nn.Module): The PyTorch model being trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        criterion (torch.nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        device (str): The device used for training.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        wandb_obj,
        model_dir,
        device="cuda",
        log_interval=25,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.wandb_obj = wandb_obj
        self.scheduler = scheduler
        self.best_accuracy = 0.0
        self.model_dir = model_dir
        self.checkpoint_saver = CheckpointSaver(
            dirpath=self.model_dir, decreasing=False, top_n=5
        )
        self.log_interval = log_interval

        self.model.to(self.device)
        self.criterion.to(self.device)

    def train(self, num_epochs):
        """
        Train the PyTorch model for a specified number of epochs.

        Parameters:
            num_epochs (int): The number of epochs to train the model.
        """
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            epoch_loss = 0
            log_interval_loss = 0.0
            pbar = tqdm(
                self.train_loader,
                desc="Epoch: {}/{}, Epoch Loss:{}".format(
                    epoch, num_epochs, epoch_loss
                ),
                ncols=100,
                unit="batches",
            )
            batch_num = 0

            for batch in pbar:
                inputs, labels, attn_mask = (
                    batch["input_values"].to(self.device),
                    batch["label"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )

                self.optimizer.zero_grad()
                output_logits = self.model(inputs, attn_mask)
                loss = self.criterion(output_logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                log_interval_loss += loss.item()
                pbar.set_postfix({"Batch Loss": loss.item()})
                if (batch_num + 1) % self.log_interval == 0:
                    self.wandb_obj.log(
                        {"Training loss": log_interval_loss / self.log_interval}
                    )
                    log_interval_loss = 0

                batch_num += 1
                loss_update_string = "Epoch {}/{}, Epoch Loss {:.4f}".format(
                    epoch, num_epochs, epoch_loss / batch_num
                )
                pbar.set_description(loss_update_string)

            val_accuracy = self.evaluate(epoch, num_epochs)
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.save_checkpoint(epoch, self.model.state_dict(), val_accuracy)

    def evaluate(self, epoch, num_epochs):
        """
        Evaluate the PyTorch model on the validation set.

        Parameters:
            epoch (int): The current epoch number.
            num_epochs (int): The total number of epochs for training.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc="Validating Epoch:{}/{}, Loss:{}".format(
                    epoch, num_epochs, total_loss
                ),
                ncols=100,
                unit="batches",
            )
            for batch in pbar:
                inputs, labels, attn_mask = (
                    batch["input_values"].to(self.device),
                    batch["label"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )

                outputs = self.model(inputs, attn_mask)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                # print(total_loss, correct, total)
                val_loss_update_str = "Val Loss: {:.4f}, Val Acc: {:.4f}%".format(
                    total_loss / total,
                    correct / total,
                )
                pbar.set_postfix({"Batch Loss": loss.item()})
                pbar.set_description(val_loss_update_str)

                # pbar.set_postfix(
                #    {"Vaildation Loss": total_loss / total, "Acc": correct / total}
                # )

            val_accuracy = 100.0 * correct / total
            avg_val_loss = total_loss / len(self.val_loader)

            self.wandb_obj.log(
                {"Validation Loss": avg_val_loss, "Validation Accuracy": val_accuracy}
            )

        # print("\nValidation Results - Epoch {}/{}:".format(epoch, num_epochs))
        # print(
        #    "Average Loss: {:.4f}, Accuracy: {:.2f}%".format(
        #        total_loss / len(self.val_loader), val_accuracy
        #    )
        # )
        return val_accuracy

    def save_checkpoint(self, epoch, state_dict, accuracy):
        """
        Save the current model checkpoint if the validation accuracy is
        higher than the best seen so far.

        Parameters:
            epoch (int): The current epoch number.
            state_dict (dict): The model's state dictionary to be saved.
            accuracy (float): The validation accuracy achieved with this
            checkpoint.
        """
        print("Saved checkpoint at epoch:{}".format(epoch))
        checkpoint = {"epoch": epoch, "state_dict": state_dict, "accuracy": accuracy}
        torch.save(checkpoint, "best_checkpoint.pt")
        self.checkpoint_saver(self.model, epoch, accuracy)


def make_pytorch_trainer(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    model_dir,
    wandb_obj,
):
    os.mkdir(model_dir)

    trainer = PyTorchModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_dir=model_dir,
        wandb_obj=wandb_obj,
    )
    return trainer
