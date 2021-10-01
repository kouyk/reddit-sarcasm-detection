import os
import pickle
import platform

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .util import StageType, tokenize_as_pickle
from .sarcasm_dataset import SarcasmDataset


class SarcasmDataModule(LightningDataModule):

    def __init__(self,
                 train_path: str,
                 test_path: str = None,
                 pretrained_name: str = 'bert-base-cased',
                 batch_size=32):
        super().__init__()

        self.train_path = train_path
        self.train_pkl = train_path.replace('.csv', '.pkl') if train_path else None
        self.test_path = test_path
        self.test_pkl = test_path.replace('.csv', '.pkl') if test_path else None
        self.batch_size = batch_size

        self.num_workers = os.cpu_count() if platform.system() == 'Linux' else 1  # workaround Windows worker issue
        self.pretrained_name = pretrained_name

    def prepare_data(self):
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_name)

        if self.train_path:
            tokenize_as_pickle(self.train_path, self.train_pkl, tokenizer)
        if self.test_path:
            tokenize_as_pickle(self.test_path, self.test_pkl, tokenizer)

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            with open(self.train_pkl, 'rb') as f:
                encoded = pickle.load(f)

            # Perform train-val split
            keys = ['input_ids', 'attention_mask', 'targets']
            split = train_test_split(*[encoded[k] for k in keys], test_size=0.2)
            train = {k: v for k, v in zip(keys, split[::2])}
            val = {k: v for k, v in zip(keys, split[1::2])}

            self.train_dataset = SarcasmDataset(train)
            self.val_dataset = SarcasmDataset(val)

        if stage in (None, 'test'):
            if not self.test_pkl:
                return

            with open(self.test_pkl, 'rb') as f:
                encoded = pickle.load(f)

            self.test_dataset = SarcasmDataset(encoded)

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
