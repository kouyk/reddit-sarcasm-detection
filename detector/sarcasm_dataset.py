import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class SarcasmDataset(Dataset):
    """
    Dataset for all Reddit Sarcasm text, text length support is dependent on the underlying model.
    """
    TEXT_COLUMN = 'comment'
    LABEL_COLUMN = 'label'
    PARENT_COLUMN = 'parent_comment'

    def __init__(self, df: DataFrame,
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: int = 512,
                 use_parent: bool = True):
        super().__init__()

        self.df = df.copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_parent = use_parent

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        selected = self.df.loc[index]
        encoded = self.tokenizer(selected[SarcasmDataset.TEXT_COLUMN],
                                 text_pair=selected[SarcasmDataset.PARENT_COLUMN] if self.use_parent else None,
                                 padding='max_length',
                                 return_token_type_ids=True,
                                 truncation=True,
                                 max_length=self.max_length,
                                 return_tensors='pt')
        encoded = {k: v.flatten() for k, v in encoded.items()}

        if SarcasmDataset.LABEL_COLUMN in self.df.columns:
            encoded['targets'] = torch.tensor(selected[SarcasmDataset.LABEL_COLUMN])

        return encoded
