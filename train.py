import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.training_type import DDPPlugin
from transformers import logging

from detector import SarcasmDataModule, SarcasmDetector

logging.set_verbosity_error()


def main(args: argparse.Namespace):

    if args.deterministic:
        seed_everything(args.seed, workers=True)

    dm = SarcasmDataModule.from_argparse_args(args)

    callbacks = [
        ModelCheckpoint(
            filename='{epoch}-{val_loss:.2f}-{val_accuracy:.2f}',
            monitor='val_loss',
            mode='min',
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
        save_dir=args.logdir,
        name=args.pretrained_name,
        default_hp_metric=False
    )

    multi_gpu = args.gpus > 1

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        weights_summary=None,
        accelerator='ddp' if multi_gpu else None,
        plugins=DDPPlugin(find_unused_parameters=False) if multi_gpu else None
    )

    model = SarcasmDetector(**vars(args))
    trainer.fit(model, datamodule=dm)
    if args.test:
        trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    # General args
    parser.add_argument('--logdir', help="Logging directory for tensorboard", default='lightning_logs')
    parser.add_argument('--seed', help="Seed to use if deterministic flag is set", default=3244, type=int)
    parser.add_argument('--test', help="Perform testing at the end of training", action='store_true')

    parser = SarcasmDataModule.add_argparse_args(parser)
    parser = SarcasmDetector.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1,
                        precision=16,
                        max_epochs=4,
                        weights_save_path="checkpoints")

    main(parser.parse_args())
