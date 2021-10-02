import re
from math import ceil, floor

import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import Accuracy, F1, Precision, Recall, MatthewsCorrcoef, CohenKappa, MetricCollection
from transformers import AutoModelForSequenceClassification, get_constant_schedule_with_warmup

from .util import StageType


class SarcasmDetector(LightningModule):

    def __init__(self, config, return_logits=False):
        super().__init__()
        self.save_hyperparameters(config)
        self.hparams.num_classes = 2
        self.return_logits = return_logits

        self.classifier = self.get_classifier()
        self.freeze_layers()

        # Initialise the metrics
        metrics = MetricCollection({'accuracy': Accuracy(num_classes=self.hparams.num_classes),
                                    'f1': F1(average='macro', num_classes=self.hparams.num_classes),
                                    'precision': Precision(average='macro', num_classes=self.hparams.num_classes),
                                    'recall': Recall(average='macro', num_classes=self.hparams.num_classes),
                                    'mcc': MatthewsCorrcoef(num_classes=self.hparams.num_classes),
                                    'kappa': CohenKappa(num_classes=self.hparams.num_classes)})
        self.metrics = {step_type: metrics.clone(prefix=step_type.value)
                        for step_type in StageType if step_type != StageType.PREDICT}

    def get_classifier(self):
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.hparams.pretrained_name,
                hidden_dropout_prob=self.hparams.dropout,
                num_labels=self.hparams.num_classes,
                problem_type='single_label_classification'
            )
        except TypeError:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.hparams.pretrained_name,
                dropout=self.hparams.dropout,
                num_labels=self.hparams.num_classes,
                problem_type='single_label_classification'
            )

        return model

    def freeze_layers(self):
        if self.hparams.freeze_backbone is not None:
            layer_max = self.get_layer_max()
            assert (isinstance(self.hparams.freeze_backbone, int)
                    and 0 <= self.hparams.freeze_backbone <= layer_max), \
                f'freeze_backbone must be an integer between 0 and {layer_max}'

            for name, param in self.classifier.base_model.named_parameters():
                if f'layer.{self.hparams.freeze_backbone}.' in name:
                    break

                param.requires_grad = False

    def get_layer_max(self):
        pattern = re.compile(r"\.layer\.(\d+)\.")
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

        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

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
        return self.classifier(input_ids, attention_mask=attention_mask, labels=labels)

    def common_step(self, batch, step_type: StageType):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['targets']

        output = self(input_ids, attention_mask=attention_mask, labels=targets)
        predictions = torch.argmax(output.logits, dim=1).cpu()
        targets = targets.cpu()

        self.log(
            f'{step_type.value}_loss', output.loss, prog_bar=True, logger=step_type != StageType.TEST,
            sync_dist=step_type in (StageType.VAL, StageType.TEST)
        )
        metric_output = self.metrics[step_type](predictions, targets)
        self.log_dict(metric_output, prog_bar=False, logger=step_type != StageType.TEST)

        return {'loss': output.loss, 'predictions': predictions, 'targets': targets}

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/val_loss": 0, "hp/val_acc": 0})

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, step_type=StageType.TRAIN)

    def validation_step(self, batch, batch_idx):
        return_dict = self.common_step(batch, step_type=StageType.VAL)
        self.log_dict({'hp/val_loss': return_dict['loss'], 'hp/val_acc': self.metrics['val_acc'].compute()})
        return return_dict

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, step_type=StageType.TEST)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        output = self(input_ids, attention_mask=attention_mask)

        if self.return_logits:
            return output.logits

        predictions = torch.argmax(output.logits, 1)

        return predictions

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('loss')
        return items
