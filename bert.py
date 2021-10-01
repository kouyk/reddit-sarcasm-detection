import os

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from transformers import logging

from detector import SarcasmDataModule, SarcasmDetector

logging.set_verbosity_error()
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if __name__ == "__main__":
    cfg = {
        'pretrained_name': 'bert-base-cased',
        'lr': 2e-5,
        'freeze_backbone': 12,
        'max_epochs': 4,
        'batch_size': 128,
        'dropout': 0.1,
        'scheduler': 'onecycle'
    }

    dm = SarcasmDataModule(train_path='train-balanced.csv',
                           pretrained_name=cfg['pretrained_name'],
                           batch_size=cfg['batch_size'])

    callbacks = [
        ModelCheckpoint(
            filename='{epoch}-{val_loss:.2f}-{val_accuracy:.2f}',
            monitor='val_accuracy',
            mode='max',
            save_last=False,
            save_top_k=1,
            every_n_epochs=1
        ),
        LearningRateMonitor(
            logging_interval='step',
            log_momentum=True
        ),
    ]
    logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name=f'bert',
        default_hp_metric=False
    )

    trainer = Trainer(
        gpus=1,
        callbacks=callbacks,
        weights_save_path="checkpoints",
        logger=logger,
        log_every_n_steps=50,
        max_epochs=cfg['max_epochs'],
        weights_summary=None,
        precision=16 if torch.cuda.is_available() else 32,
        #accelerator="ddp"
    )

    model = SarcasmDetector(config=cfg)
    trainer.fit(model, datamodule=dm)
