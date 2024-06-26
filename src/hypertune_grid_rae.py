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
import rae_model
from loguru import logger
import numpy as np
import sys
import datasets, metrics
import mltrainer
from mltrainer import ReportTypes, Trainer, TrainerSettings
mltrainer.__version__
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Hypertune using grid for latent space and hidden state, metric is hausdorff distance, to measure how well the model seperates normal and abnormal heart samples
logger.add("logs/rae_hypertune_latent_hidden.log")

def predict(autoencoder, dataset):
    predictions, losses = [], []
    criterion = rae_model.ReconstructionLoss()
    with torch.no_grad():
        autoencoder.eval()

        for input_sequence, target in dataset:
            input_sequence = input_sequence.unsqueeze(0).to(device) 

            seq_pred = autoencoder(input_sequence)
            loss = criterion(seq_pred, input_sequence.squeeze(0))
            
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses

def calculateOverlapNormalAbnormalReconstructionLoss(dataset, autoencoder):
    losses_per_target = {}

    # Predict and accumulate losses per target
    _, all_losses = predict(autoencoder, dataset)
    for loss, (_, target) in zip(all_losses, dataset):
        target_class = "abnormal" if target.item() == 1 else "normal"
        if target_class not in losses_per_target:
            losses_per_target[target_class] = []
        losses_per_target[target_class].append(loss)

    # Calculate overlap using Hausdorff distance
    normal_losses = losses_per_target['normal']
    abnormal_losses = losses_per_target['abnormal']
    
    all_losses_flat = normal_losses + abnormal_losses
    targets_flat = [0] * len(normal_losses) + [1] * len(abnormal_losses)
    
    return roc_auc_score(targets_flat, all_losses_flat)
    
def main():
    gin.parse_config_file(Path(__file__).parent / 'rae_config.gin')

    logger.info("hypertune rae latent and hidden size")
    # gin.parse_config_file(Path(__file__).parent / 'config.gin')
    
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

    for ls in [8,16,24,32,40]:
        for hs in [20,30,40,50,60]:
            
            logger.info('start tuning latent space size: '+str(ls)+' hidden state size: '+str(hs))
            
            hypertune_config = {
                    "num_layers" : 1,
                    "seq_len" : 192,
                    "latent" : ls,
                    "hidden" : hs,
                    "dropout" : 0.2,
                    "features" : 1,
                }

            lossfn = rae_model.ReconstructionLoss()
            autoencoder = rae_model.RecurrentAutoencoder(hypertune_config,hypertune_config)
            
            settings = TrainerSettings(
                epochs=200,
                metrics=[lossfn],
                logdir="logs",
                train_steps=200,
                valid_steps=200,
                optimizer_kwargs = {"lr": 1e-4},
                reporttypes=[ReportTypes.TENSORBOARD],
                earlystop_kwargs = {
                    "save": False,
                    "verbose": True,
                    "patience": 10,
                },
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
            logger.success('latent space size: '+str(ls))
            logger.success('hidden state size: '+str(hs))
            logger.success(f'AUC: {calculateOverlapNormalAbnormalReconstructionLoss(validsdatasetVAE, autoencoder):.4f}')
            logger.success('-------------------------------')

    logger.success("finished hypertuning")

if __name__ == "__main__":
    main()
