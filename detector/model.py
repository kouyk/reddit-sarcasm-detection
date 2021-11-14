import re
from math import ceil

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import F1, Precision, Recall, CohenKappa, MetricCollection, Accuracy
from transformers import AutoModelForSequenceClassification, get_constant_schedule_with_warmup

from .util import StageType, get_device_count


class SarcasmDetector(LightningModule):
    NUM_CLASSES = 2

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("SarcasmDetector")
        parser.add_argument('--pretrained_name', help="Name of pretrained model", default='bert-base-cased')
        parser.add_argument('--lr', help="Learning rate", default=1e-5, type=float)
        parser.add_argument('--freeze_extractor', help="Number of layers to freeze in the pretrained model", default=0,
                            type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--scheduler', help="LR scheduler", default='warmup', choices=['onecycle', 'warmup'])
        parser.add_argument('--enable_parent', help="Use the parent comment for classification", action='store_true')

        return parent_parser

    def __init__(self, return_logits=False, **kwargs):
        super().__init__()
        self.save_hyperparameters("pretrained_name", "lr", "freeze_extractor", "dropout", "scheduler", "batch_size",
                                  "max_length", "enable_parent")
        self.return_logits = return_logits

        self.classifier = self.get_classifier()
        self.freeze_classifier_layers()

        # Initialise the metrics
        metrics = MetricCollection({
            'accuracy': Accuracy(num_classes=self.NUM_CLASSES, average='micro'),
            'f1': F1(num_classes=self.NUM_CLASSES, average='micro', ignore_index=0),
            'kappa': CohenKappa(num_classes=self.NUM_CLASSES),
            'precision': Precision(num_classes=self.NUM_CLASSES, average='micro', ignore_index=0),
            'recall': Recall(num_classes=self.NUM_CLASSES, average='micro', ignore_index=0),
        })
        self.metrics = nn.ModuleDict({step_type.value: metrics.clone(prefix=f'{step_type.value.lower()}_')
                                      for step_type in StageType if step_type != StageType.PREDICT})

    def get_classifier(self):
        try:
            extractor = AutoModelForSequenceClassification.from_pretrained(
                self.hparams.pretrained_name,
                hidden_dropout_prob=self.hparams.dropout,
                problem_type='single_label_classification',
                num_labels=self.NUM_CLASSES
            )
        except TypeError:
            extractor = AutoModelForSequenceClassification.from_pretrained(
                self.hparams.pretrained_name,
                dropout=self.hparams.dropout,
                problem_type='single_label_classification',
                num_labels=self.NUM_CLASSES
            )

        return extractor

    def freeze_classifier_layers(self):
        if self.hparams.freeze_extractor is not None:
            assert (isinstance(self.hparams.freeze_extractor, int)
                    and 0 <= self.hparams.freeze_extractor <= self.extractor_layer_count), \
                f'freeze_extractor must be an integer between 0 and {self.extractor_layer_count}'

            for name, param in self.classifier.base_model.named_parameters():
                if f'layer.{self.hparams.freeze_extractor}.' in name:
                    break

                param.requires_grad = False

    @property
    def extractor_layer_count(self) -> int:
        pattern = re.compile(r"layer\.(\d+)\.")
        layer_index = set()
        for name, param in self.classifier.named_parameters():
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

        if self.trainer.max_steps > 0:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        effective_accum = self.trainer.accumulate_grad_batches
        if not isinstance(limit_batches, int):
            batches = int(len(self.trainer.datamodule.train_dataloader()) * limit_batches)
            num_devices = get_device_count(self.trainer.devices)
            num_devices *= self.trainer.num_nodes
            effective_accum *= num_devices
        else:
            batches = limit_batches

        return ceil(batches / effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr,
                                      weight_decay=0.01)

        if self.hparams.scheduler == 'onecycle':
            lr_scheduler = OneCycleLR(optimizer,
                                      total_steps=self.num_training_steps,
                                      max_lr=self.hparams.lr)
        else:  # 'warmup'
            num_warmup_steps = 1000
            lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}
        }

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        output = self.classifier(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)

        if labels is None:
            return output.logits

        return output.loss, output.logits

    def common_step(self, batch, step_type: StageType):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        targets = batch['targets']

        loss, logits = self(input_ids, token_type_ids, attention_mask, targets)
        predictions = torch.argmax(logits, dim=1)

        self.log(
            f'{step_type.value.lower()}_loss', loss, prog_bar=True, logger=step_type != StageType.TEST,
            sync_dist=step_type in (StageType.VAL, StageType.TEST)
        )
        metric_output = self.metrics[step_type.value](predictions, targets)
        self.log_dict(metric_output, prog_bar=False, logger=step_type != StageType.TEST)

        return {'loss': loss, 'predictions': predictions, 'targets': targets}

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, step_type=StageType.TRAIN)

    def validation_step(self, batch, batch_idx):
        return_dict = self.common_step(batch, step_type=StageType.VAL)
        self.log('hp_metric', return_dict['loss'])
        return return_dict

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, step_type=StageType.TEST)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']

        logits = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        if self.return_logits:
            return logits

        predictions = torch.argmax(logits, 1)

        return predictions
