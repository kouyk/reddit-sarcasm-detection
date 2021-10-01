import os
import pickle
from enum import Enum

import pandas as pd
import torch

from .sarcasm_dataset import SarcasmDataset

class StageType(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    PREDICT = 'predict'


def tokenize_as_pickle(csv_path, pkl_path, tokenizer):
    if not os.path.isfile(pkl_path):
        df = pd.read_csv(csv_path)
        encoded = tokenizer(df[SarcasmDataset.TEXT_COLUMN].to_list(),
                            padding='max_length',
                            return_token_type_ids=False,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt')
        if SarcasmDataset.LABEL_COLUMN in df.columns:
            encoded['targets'] = torch.tensor(df[SarcasmDataset.LABEL_COLUMN].values)

        with open(pkl_path, 'wb') as f:
            pickle.dump(encoded, f)
