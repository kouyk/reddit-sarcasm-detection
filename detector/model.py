import re

import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import Accuracy, F1, Precision, Recall, MatthewsCorrcoef, CohenKappa, AUROC
from transformers import AutoModelForSequenceClassification, get_constant_schedule_with_warmup

from .util import StageType


class SarcasmDetector(LightningModule):

    def __init__(self, config, return_logits=False):
        super().__init__()
        self.save_hyperparameters(config)
        self.hparams.num_classes = 2
        self.return_logits = return_logits

        # Initialise the metrics
        self.classifier = self.get_classifier()
        self.accuracy = {step_type: Accuracy(num_classes=self.hparams.num_classes) for step_type in StageType}
        self.f1_micro = {step_type: F1(average='micro', num_classes=self.hparams.num_classes)
                         for step_type in StageType}
        self.f1_macro = {step_type: F1(average='macro', num_classes=self.hparams.num_classes)
                         for step_type in StageType}
        self.pre = {step_type: Precision(average='macro', num_classes=self.hparams.num_classes)
                          for step_type in StageType}
        self.recall = {step_type: Recall(average='macro', num_classes=self.hparams.num_classes)
                       for step_type in StageType}
        self.mcc = {step_type: MatthewsCorrcoef(num_classes=self.hparams.num_classes) for step_type in StageType}
        self.kappa = {step_type: CohenKappa(num_classes=self.hparams.num_classes) for step_type in StageType}

        self.freeze_layers()

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr,
                                      weight_decay=0.01)

        if self.trainer.datamodule is not None:
            num_steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        else:
            num_steps_per_epoch = len(self.train_dataloader())

        total_steps = num_steps_per_epoch * self.trainer.max_epochs

        if self.hparams.scheduler == 'onecycle':
            lr_scheduler = OneCycleLR(optimizer,
                                      total_steps=total_steps,
                                      max_lr=self.hparams.lr)
        else:
            num_warmup_steps = total_steps * 3 // 10
            lr_scheduler = get_constant_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=num_warmup_steps)

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

        prefix = step_type.value
        self.log(f'{prefix}_loss', output.loss, prog_bar=True, logger=step_type != StageType.TEST)
        self.log_dict(
            {
                f'{prefix}_accuracy': self.accuracy[step_type](predictions, targets),
                f'{prefix}_f1_micro': self.f1_micro[step_type](predictions, targets),
                f'{prefix}_f1_macro': self.f1_macro[step_type](predictions, targets),
                f'{prefix}_precision': self.pre[step_type](predictions, targets),
                f'{prefix}_recall': self.recall[step_type](predictions, targets),
                f'{prefix}_mcc': self.mcc[step_type](predictions, targets),
                f'{prefix}_kappa': self.kappa[step_type](predictions, targets),
            },
            prog_bar=False, logger=step_type != StageType.TEST
        )

        return {'loss': output.loss, 'predictions': predictions, 'targets': targets}

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, step_type=StageType.TRAIN)

    def validation_step(self, batch, batch_idx):
        return_dict = self.common_step(batch, step_type=StageType.VAL)
        self.log('hp/val_loss', return_dict['loss'])
        return return_dict

    def validation_epoch_end(self, validation_step_outputs):
        self.log('hp/val_acc', self.accuracy[StageType.VAL].compute())

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
