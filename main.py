from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import utils
from utils import logger

from config import config
from dataset import CustomTensorDataset
from models import Resnet, fcn_autoencoder, conv_autoencoder, VAE
from loss import VaeLoss
import numpy as np
import pandas as pd
from torch_optimizer import AdaBound

model_classes = {
    "resnet": Resnet(),
    "fcn": fcn_autoencoder(),
    "cnn": conv_autoencoder(),
    "vae": VAE(),
}


def train(device):
    logger.info("start training...")

    if config.use_wandb:
        wandb.login(key="8e7f40150f49731a42b58b0284c816eed0e2e9c5")
        wandb.init(project="MLHW8", entity="shawnhung", config=config)

    dataset = CustomTensorDataset(Path(config.root_dir) / "trainingset.npy")
    train_set, val_set = utils.train_val_dataset(dataset)

    train_dataloader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
    model = model_classes[config.model_type].to(device)

    optimizer = AdaBound(model.parameters(), **config.optim_hparas)
    criterion = nn.MSELoss()
    if config.model_type == 'vae':
        criterion = VaeLoss(criterion)

    if config.resume_model:
        utils.load_checkpoint(model, optimizer)

    if config.use_wandb:
        wandb.watch(model)

    min_loss = 9999

    for epoch in range(config.n_epochs):
        model.train()
        total_loss = []

        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            imgs = batch.to(device)
            if config.model_type == 'fcn':
                imgs = torch.flatten(imgs, 1)

            # ===================forward=====================
            preds = model(imgs)
            loss = criterion(preds, imgs)
            total_loss.append(loss.item())
            # ===================backward====================
            loss.backward()
            optimizer.step()

        mean_loss = np.mean(total_loss)
        if config.use_wandb:
            wandb.log(
                {"train/loss": mean_loss}
            )
        logger.info(f"Training | Epoch {epoch + 1} | loss: {mean_loss:.4f}")

        val_loss = evaluate(model, val_dataloader, criterion, imgs)
        model.eval()
        if config.use_wandb:
            wandb.log(
                {"Val/loss": val_loss}
            )
        logger.info(f"Validation | Epoch {epoch + 1} | val_loss = {val_loss:.3f}")

        # Save model to specified path
        if val_loss < min_loss:
            min_loss = val_loss
            logger.info(f"Saving model (epoch = {epoch+1}, min_loss = {min_loss:.4f}")
            utils.save_chekcpoint(model)

        logger.info("finish validation")

    logger.info("finish training")


def evaluate(model, dataloader, criterion, imgs):

    total_loss = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs = batch.to(device)
            if config.model_type == 'fcn':
                imgs = torch.flatten(imgs, 1)

            output = model(imgs)
            loss = criterion(output, imgs)
            total_loss.append(loss.item())

    return np.mean(total_loss)


def test(device):
    eval_loss = nn.MSELoss(reduction='none')
    model = model_classes[config.model_type].to(device)
    utils.load_checkpoint(model)
    # model = accelerator.prepare(model)

    dataset = CustomTensorDataset(Path(config.root_dir) / "testingset.npy")
    test_loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    logger.info("Evaluating Test Set ...")
    result = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            imgs = batch.to(device)
            if config.model_type == 'fcn':
                imgs = torch.flatten(imgs, 1)

            output = model(imgs)
            if config.model_type in ['cnn', 'resnet', 'fcn']:
                output = output
            elif config.model_type == 'vae':
                output = output[0]

            if config.model_type in ['fcn']:
                loss = eval_loss(output, imgs).sum(-1)
            else:
                loss = eval_loss(output, imgs).sum([1, 2, 3])
            result.append(loss)

    anomality = torch.cat(result, axis=0).cpu()
    anomality = torch.sqrt(anomality).reshape(len(dataset), 1).numpy()

    df = pd.DataFrame(anomality, columns=['Predicted'])
    df.to_csv(config.out_file, index_label='Id')

    logger.info("Testing Completed!")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utils.fix_seed(19530615)
    train(device)
    test(device)
