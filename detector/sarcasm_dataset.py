import torch
from pandas import DataFrame
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .util import Column, COL_ONEHOT_CLS


class SarcasmDataset(Dataset):
    """
    Dataset for all Reddit Sarcasm text, text length support is dependent on the underlying model.
    """

    def __init__(self,
                 df: DataFrame,
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: int = 512,
                 disabled_features=None):
        super().__init__()

        if disabled_features is None:
            disabled_features = []

        self.df = df.copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.disabled_features = disabled_features

        self.use_parent = Column.PARENT.value in self.disabled_features
        self.use_score = Column.SCORE.value in self.disabled_features
        self.extra_features = [k for k in COL_ONEHOT_CLS.keys() if k not in self.disabled_features]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        selected = self.df.iloc[index]
        encoded = self.tokenizer(
            text=selected[Column.PARENT.value] if self.use_parent else selected[Column.COMMENT.value],
            text_pair=selected[Column.COMMENT.value] if self.use_parent else None,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoded = {k: v.flatten() for k, v in encoded.items()}

        features = [one_hot(torch.as_tensor(selected[f]), num_classes=COL_ONEHOT_CLS[f]) for f in self.extra_features]
        if Column.SCORE.value not in self.disabled_features:
            features.append(torch.as_tensor(selected[Column.SCORE.value], dtype=torch.float).unsqueeze(dim=0))
        encoded['features'] = torch.cat(features) if features else torch.empty(0)

        if Column.LABEL.value in self.df.columns:
            encoded['targets'] = torch.tensor(selected[Column.LABEL.value])

        return encoded
