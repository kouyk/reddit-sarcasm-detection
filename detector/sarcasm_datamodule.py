import os
import platform

import joblib
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .sarcasm_dataset import SarcasmDataset
from .util import StageType


class SarcasmDataModule(LightningDataModule):

    def __init__(self,
                 train_path: str = 'dataset/train.csv',
                 val_path: str = 'dataset/val.csv',
                 test_path: str = 'dataset/test.csv',
                 type_dict_path: str = 'dataset/input_types.joblib',
                 pretrained_name: str = 'bert-base-cased',
                 batch_size: int = 32,
                 max_length: int = 512,
                 enable_parent: bool = True):
        super().__init__()

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.type_dict_path = type_dict_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.enable_parent = enable_parent

        self.num_workers = os.cpu_count() // 2 if platform.system() == 'Linux' else 1  # workaround Windows worker issue
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    def setup(self, stage=None):
        key_types = joblib.load(self.type_dict_path)

        if stage in (None, 'fit'):
            train_df = pd.read_csv(self.train_path, dtype=key_types)
            val_df = pd.read_csv(self.val_path, dtype=key_types)

            self.train_dataset = SarcasmDataset(train_df, self.tokenizer, self.max_length, self.enable_parent)
            self.val_dataset = SarcasmDataset(val_df, self.tokenizer, self.max_length, self.enable_parent)

        if stage in (None, 'test'):
            test_df = pd.read_csv(self.test_path, dtype=key_types)
            self.test_dataset = SarcasmDataset(test_df, self.tokenizer, self.max_length, self.enable_parent)

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
