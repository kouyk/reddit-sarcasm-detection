import os
import platform

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .sarcasm_dataset import SarcasmDataset
from .util import StageType


class SarcasmDataModule(LightningDataModule):

    def __init__(self,
                 train_path: str = 'dataset/train.csv',
                 val_path: str = 'dataset/val.csv',
                 test_path: str = 'dataset/test-balanced.csv',
                 pretrained_name: str = 'bert-base-cased',
                 batch_size: int = 32,
                 max_length: int = 128):
        super().__init__()

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.max_length = max_length

        self.num_workers = os.cpu_count() if platform.system() == 'Linux' else 1  # workaround Windows worker issue
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            if not self.train_path:
                return

            if self.val_path:
                train_df = pd.read_csv(self.train_path)
                val_df = pd.read_csv(self.val_path)
            else:
                # Perform train-val split
                df = pd.read_csv(self.train_path)
                train_df, val_df = train_test_split(df, test_size=0.2)
                train_df.reset_index(drop=True)
                val_df.reset_index(drop=True)

            self.train_dataset = SarcasmDataset(train_df, self.tokenizer, self.max_length)
            self.val_dataset = SarcasmDataset(val_df, self.tokenizer, self.max_length)

        if stage in (None, 'test'):
            if not self.test_path:
                return

            test_df = pd.read_csv(self.test_path)
            self.test_dataset = SarcasmDataset(test_df, self.tokenizer, self.max_length)

    def gen_dataloader(self, dataset, dataset_type: StageType):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=dataset_type == StageType.TRAIN,
                          num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available())

    def train_dataloader(self):
        return self.gen_dataloader(self.train_dataset, StageType.TRAIN)

    def val_dataloader(self):
        return self.gen_dataloader(self.val_dataset, StageType.VAL)

    def test_dataloader(self):
        try:
            return self.gen_dataloader(self.test_dataset, StageType.TEST)
        except NameError:
            raise Exception("Test dataframe needs to be supplied!")
