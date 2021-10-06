import re
from collections import OrderedDict

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import F1, Precision, Recall, CohenKappa, MetricCollection
from transformers import AutoModel, get_constant_schedule_with_warmup

from .util import StageType


class SarcasmDetector(LightningModule):
    num_classes = 2

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("SarcasmDectector")
        parser.add_argument('--pretrained_name', help="Name of pretrained model", default='bert-base-cased')
        parser.add_argument('--lr', help="Learning rate", default=1e-5, type=float)
        parser.add_argument('--freeze_extractor', help="Number of layers to freeze in the pretrained model", default=0,
                            type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--scheduler', help="LR scheduler", default='onecycle', choices=['onecycle', 'warmup_only'])

        return parent_parser

    def __init__(self, return_logits=False, **kwargs):
        super().__init__()
        self.save_hyperparameters("pretrained_name", "lr", "freeze_extractor", "dropout", "scheduler", "batch_size",
                                  "max_length")
        self.return_logits = return_logits

        self.extractor = self.get_extractor()
        self.freeze_extractor_layers()

        # Adapted from RoBERTa's classification head
        self.classifier = torch.nn.Sequential(OrderedDict([
            ('dense', nn.Linear(self.extractor.config.hidden_size, self.extractor.config.hidden_size)),
            ('dropout', nn.Dropout(self.hparams.dropout)),
            ('out_proj', nn.Linear(self.extractor.config.hidden_size, self.num_classes))
        ]))

        # Initialise the metrics
        metrics = MetricCollection({'f1': F1(average='macro', num_classes=self.num_classes),
                                    'precision': Precision(average='macro', num_classes=self.num_classes),
                                    'recall': Recall(average='macro', num_classes=self.num_classes),
                                    'kappa': CohenKappa(num_classes=self.num_classes)})
        self.metrics = nn.ModuleDict({step_type.value: metrics.clone(prefix=f'{step_type.value.lower()}_')
                                      for step_type in StageType if step_type != StageType.PREDICT})
        self.metrics.requires_grad_(False)

    def get_extractor(self):
        try:
            extractor = AutoModel.from_pretrained(
                self.hparams.pretrained_name,
                hidden_dropout_prob=self.hparams.dropout
            )
        except TypeError:
            extractor = AutoModel.from_pretrained(
                self.hparams.pretrained_name,
                dropout=self.hparams.dropout
            )

        return extractor

    def freeze_extractor_layers(self):
        if self.hparams.freeze_extractor is not None:
            assert (isinstance(self.hparams.freeze_extractor, int)
                    and 0 <= self.hparams.freeze_extractor <= self.extractor_layer_count), \
                f'freeze_extractor must be an integer between 0 and {self.extractor_layer_count}'

            for name, param in self.extractor.base_model.named_parameters():
                if f'layer.{self.hparams.freeze_extractor}.' in name:
                    break

                param.requires_grad = False

    @property
    def extractor_layer_count(self) -> int:
        pattern = re.compile(r"\.layer\.(\d+)\.")
        layer_index = set()
        for name, param in self.extractor.named_parameters():
            match = pattern.search(name)
            if match:
                layer_index.add(int(match.group(1)))

        return max(layer_index) + 1

    @property
    def num_training_steps(self) -> int:
        """
        Total training steps inferred from datamodule and devices.

        Adapted from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449#issuecomment-774265729
        """

        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        return (batches // self.trainer.accumulate_grad_batches) * self.trainer.max_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr,
                                      weight_decay=0.01)

        if self.hparams.scheduler == 'onecycle':
            lr_scheduler = OneCycleLR(optimizer,
                                      total_steps=self.num_training_steps,
                                      max_lr=self.hparams.lr)
        else:
            num_warmup_steps = self.num_training_steps * 3 // 10
            lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}
        }

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.extractor(input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)

        if labels is None:
            return logits

        loss = cross_entropy(logits, labels)
        return loss, logits

    def common_step(self, batch, step_type: StageType):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['targets']

        loss, logits = self(input_ids, attention_mask=attention_mask, labels=targets)
        predictions = torch.argmax(logits, dim=1)

        self.log(
            f'{step_type.value.lower()}_loss', loss, prog_bar=True, logger=step_type != StageType.TEST,
            sync_dist=step_type in (StageType.VAL, StageType.TEST)
        )
        metric_output = self.metrics[step_type.value](predictions, targets)
        self.log_dict(metric_output, prog_bar=False, logger=step_type != StageType.TEST)

        return {'loss': loss, 'predictions': predictions, 'targets': targets}

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/val_loss": 0})

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, step_type=StageType.TRAIN)

    def validation_step(self, batch, batch_idx):
        return_dict = self.common_step(batch, step_type=StageType.VAL)
        self.log('hp/val_loss', return_dict['loss'])
        return return_dict

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, step_type=StageType.TEST)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        logits = self(input_ids, attention_mask=attention_mask)

        if self.return_logits:
            return logits

        predictions = torch.argmax(logits, 1)

        return predictions

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('loss')
        return items
