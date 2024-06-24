from pathlib import Path
from typing import Dict
import ray
import torch
from filelock import FileLock
from loguru import logger
import mlflow
from mltrainer import Trainer, TrainerSettings, ReportTypes
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch import nn
from mads_datasets.base import BaseDatastreamer
from mltrainer.preprocessors import BasePreprocessor
import metrics, datasets

import sys
# sys.path.append('../')
# from src import datasets, metrics

SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float

class Accuracy:
    def __repr__(self) -> str:
        return "Accuracy"

    def __call__(self, y, yhat):
        return (np.argmax(yhat, axis=1) == y).sum() / len(yhat)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
        self.normalization = None
        
    def forward(self, x):
        identity = x.clone()
        x = self.conv(x)
        self.normalization = nn.LayerNorm([x.size(2), x.size(3)])
        x = self.normalization(x + identity)
        return x
    
class CNN(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden = config['hidden']
        dropout = config['dropout']
        self.convolutions = nn.ModuleList([
            ConvBlock(1, hidden, dropout),
        ])

        for i in range(config['num_layers']):
            self.convolutions.extend([ConvBlock(hidden, hidden, dropout), nn.ReLU()])
        self.convolutions.append(nn.MaxPool2d(2, 2))

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear((8*6) * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config['num_classes']),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convolutions:
            x = conv(x)
        x = self.dense(x)
        return x

def train(config: Dict):
    trainfile = data_dir / 'heart_train.parq'
    testfile = data_dir / 'heart_test.parq'
    print(trainfile)
    
    f1micro = metrics.F1Score(average='micro')
    f1macro = metrics.F1Score(average='macro')
    precision = metrics.Precision('micro')
    recall = metrics.Recall('macro')
    accuracy = metrics.Accuracy()
        
    shape = (16, 12)
    traindataset = datasets.HeartDataset2D(trainfile, target="target", shape=shape)
    testdataset = datasets.HeartDataset2D(testfile, target="target", shape=shape)
    
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = "cpu"

    traindataset.to(device)
    testdataset.to(device)
    
    
    
    with FileLock(data_dir / ".lock"):
        # we lock the datadir to avoid parallel instances trying to
        # access the datadir
        trainstreamer = BaseDatastreamer(traindataset, preprocessor = BasePreprocessor(), batchsize=config["batch_size"])
        teststreamer = BaseDatastreamer(testdataset, preprocessor = BasePreprocessor(), batchsize=config["batch_size"])

    model = CNN(config)
    model.to(device)

    mlflow.set_tracking_uri("sqlite:///mads_exam.db")
    mlflow.set_experiment("2D conv model")
    
    loss_fn = torch.nn.CrossEntropyLoss()

    with mlflow.start_run():
        optimizer = torch.optim.Adam

        settings = TrainerSettings(
            epochs=50,
            metrics=[recall, accuracy, f1micro, f1macro, precision],
            logdir="heart2D",
            train_steps=len(trainstreamer),
            valid_steps=len(teststreamer),
            reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.RAY, ReportTypes.MLFLOW],
            scheduler_kwargs=None,
            earlystop_kwargs=None
        )

        # modify the tags when you change them!
        mlflow.set_tag("model", "Conv2D")
        mlflow.set_tag("dataset", "heart_small_binary")
        mlflow.log_param("scheduler", "None")
        mlflow.log_param("earlystop", "None")

        mlflow.log_params(config)
        mlflow.log_param("epochs", settings.epochs)
        mlflow.log_param("shape0", shape[0])
        mlflow.log_param("optimizer", str(optimizer))
        mlflow.log_params(settings.optimizer_kwargs)

        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer,
            traindataloader=trainstreamer.stream(),
            validdataloader=teststreamer.stream(),
            scheduler=None,
            )
        trainer.loop()

if __name__ == "__main__":
    ray.init()

    data_dir = Path("./data").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
    tune_dir = Path("./models/ray").resolve()

    config = {
        "num_classes": 2,
        "tune_dir": tune_dir,
        "data_dir": data_dir,
        "dropout": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        "hidden": tune.sample_from(lambda _: 2**np.random.randint(3, 7)),
        'num_layers': tune.choice([1, 2, 3, 4, 5, 6, 7]),
        "batch_size": tune.sample_from(lambda _: 2**np.random.randint(2, 8)),
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=3,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB()

    analysis = tune.run(
        train,
        config=config,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        local_dir=str(config["tune_dir"]),
        num_samples=50,
        search_alg=bohb_search,
        scheduler=bohb_hyperband,
        verbose=1,
    )

    ray.shutdown()
