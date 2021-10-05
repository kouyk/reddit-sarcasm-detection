from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.training_type import DDPPlugin
from transformers import logging

from detector import SarcasmDataModule, SarcasmDetector

logging.set_verbosity_error()

if __name__ == "__main__":
    cfg = {
        'pretrained_name': 'bert-base-cased',
        'lr': 2e-5,
        'freeze_backbone': 0,
        'max_epochs': 4,
        'batch_size': 32,
        'dropout': 0.1,
        'scheduler': 'onecycle',
        'deterministic': False,
        'num_gpus': 1
    }

    if cfg['deterministic']:
        seed_everything(3244, workers=True)

    dm = SarcasmDataModule(train_path='dataset/train.csv',
                           val_path='dataset/val.csv',
                           test_path='dataset/test-balanced.csv',
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
        name=cfg['pretrained_name'],
        default_hp_metric=False
    )

    multi_gpu = cfg['num_gpus'] > 1

    trainer = Trainer(
        gpus=1,
        callbacks=callbacks,
        weights_save_path="checkpoints",
        logger=logger,
        log_every_n_steps=50,
        max_epochs=cfg['max_epochs'],
        weights_summary=None,
        precision=16 if cfg['num_gpus'] > 0 else 32,
        deterministic=cfg['deterministic'],
        accelerator="ddp" if multi_gpu else None,
        plugins=DDPPlugin(find_unused_parameters=False) if multi_gpu else None
    )

    model = SarcasmDetector(config=cfg)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, verbose=False)
