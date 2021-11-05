import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .util import Column


class SarcasmDataset(Dataset):
    """
    Dataset for all Reddit Sarcasm text, text length support is dependent on the underlying model.
    """

    def __init__(self,
                 df: DataFrame,
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: int = 128,
                 enable_parent: bool = True):
        super().__init__()

        self.df = df.copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_parent = enable_parent

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        selected = self.df.iloc[index]
        encoded = self.tokenizer(
            text=selected[Column.PARENT.value] if self.enable_parent else selected[Column.COMMENT.value],
            text_pair=selected[Column.COMMENT.value] if self.enable_parent else None,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoded = {k: v.flatten() for k, v in encoded.items()}

        if Column.LABEL.value in self.df.columns:
            encoded['targets'] = torch.tensor(selected[Column.LABEL.value])

        return encoded
