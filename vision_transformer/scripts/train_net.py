from pathlib import Path
import os
import sys
import logging
import argparse


import torch
import torch.nn as nn
from torch.optim import Adam
from transformers.optimization import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
from yacs.config import CfgNode

from btown_ser import make_data_loaders
from btown_ser import make_finetuning_hubert
from btown_ser import make_pytorch_trainer
from btown_ser.config import cfg

logging.getLogger().setLevel(logging.INFO)
# Check for CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Current device:{}".format(device))


def initialize_wandb(
    cfg: CfgNode, project: str = "fine-tuning-hubert-api", entity: str = "tatertech"
):
    # Initialize wandb
    wandb.init(project=project, entity=entity)
    # Create a wandb artifact for the model checkpoint

    # Log hyperparameters
    wb_config = wandb.config
    wb_config.num_epochs = cfg.SOLVER.EPOCHS
    wb_config.learning_rate = cfg.SOLVER.BASE_LR
    wb_config.batch_size = cfg.SOLVER.BATCH_SIZE
    wb_config.unfreeze_last_n_layers = cfg.MODEL.UNFREEZE_LAST_N_LAYERS
    return wb_config


def do_train(cfg: CfgNode):
    train_loader, val_loader, _ = make_data_loaders(cfg)
    fine_tuning_model = make_finetuning_hubert(cfg)
    wb_config = initialize_wandb(cfg=cfg)
    wandb.watch(fine_tuning_model)
    model_dir = os.path.join(
        cfg.MODEL.CHECKPOINT_DIR,
        wandb.run.name,
    )

    # Define the optimizer and learning rate scheduler (should come from config file
    # and made by a factory func)
    optimizer = Adam(fine_tuning_model.parameters(), lr=cfg.SOLVER.BASE_LR)
    total_steps = len(train_loader) * cfg.SOLVER.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.SOLVER.WARMUP_STEPS,
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss()

    trainer = make_pytorch_trainer(
        model=fine_tuning_model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_dir=model_dir,
        wandb_obj=wandb,
    )

    trainer.train(num_epochs=cfg.SOLVER.EPOCHS)
    return


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "config_file",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts != None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    do_train(cfg=cfg)


if __name__ == "__main__":
    main()
