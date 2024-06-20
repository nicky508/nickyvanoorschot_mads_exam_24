from mads_datasets.base import BaseDatastreamer
from mltrainer.preprocessors import BasePreprocessor
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch import nn
import torch
import gin
from streamer import VAEstreamer
import vae
from loguru import logger

import sys
import datasets, metrics
import mltrainer
from mltrainer import ReportTypes, Trainer, TrainerSettings
mltrainer.__version__

logger.add("logs/vae.log")

def main():
    logger.info("starting exam.py")
    gin.parse_config_file(Path(__file__).parent / 'config.gin')
    
    # use the binary data for training the Variational Autoencoder
    trainfileVAE = Path('./data/heart_train.parq').resolve()
    testfileVAE = Path('./data/heart_test.parq').resolve()

    # use the big dataset for training the classification model
    trainfileClass = Path('./data/heart_big_train.parq').resolve()
    testfileClass = Path('./data/heart_big_test.parq').resolve()

    # use the valid dataset (20% from the testfile) for validation of the ensembled models
    # validFileTotal <--- from testfileClass = Path('../data/heart_big_test.parq').resolve()

    # Remove outliers for training the VAE
    traindatasetVAE = datasets.HeartDataset1D(trainfileVAE, target="target", outliersRemoval=True)
    testdatasetVAE = datasets.HeartDataset1D(testfileVAE, target="target", outliersRemoval=True)
    
    #  Keep outliers for validation and finding a appropriate reconstructionloss
    validsdatasetVAE = datasets.HeartDataset1D(testfileVAE, target="target", outliersRemoval=False)
    
    trainstreamerVAE = VAEstreamer(traindatasetVAE, batchsize=32).stream()
    teststreamerVAE = VAEstreamer(testdatasetVAE, batchsize=32).stream()

    X1, X2 = next(trainstreamerVAE)

    encoder = vae.Encoder()
    decoder = vae.Decoder()

    latent = encoder(X1)
    logger.info(f"the latent shape : {latent.shape}")

    x = decoder(latent)
    logger.info(f"the shape after: {x.shape}")

    lossfn = vae.ReconstructionLoss()
    loss = lossfn(x, X2)
    logger.info(f"Untrained loss: {loss}")

    logger.info(f"starting training for {100} epochs")
    autoencoder = vae.AutoEncoder()
    
    settings = TrainerSettings(
        epochs=100,
        metrics=[lossfn],
        logdir="logs",
        train_steps=200,
        valid_steps=200,
        reporttypes=[ReportTypes.TENSORBOARD],
        scheduler_kwargs={"factor": 0.5, "patience": 10},
    )

    trainer = Trainer(
        model=autoencoder,
        settings=settings,
        loss_fn=lossfn,
        optimizer=torch.optim.Adam,
        traindataloader=trainstreamerVAE,
        validdataloader=teststreamerVAE,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
    trainer.loop()
    modeldir = Path("models")

    if not modeldir.exists():
        modeldir.mkdir(parents=True)

    modelpath = modeldir / Path("vaemodel.pt")

    torch.save(autoencoder, modelpath)

    logger.success("finished autoencode.py")

if __name__ == "__main__":
    main()

