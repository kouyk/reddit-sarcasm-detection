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

    def __init__(self, df: DataFrame, tokenizer: PreTrainedTokenizerBase):
        super().__init__()

        self.df = df.copy()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        selected_comment = self.df.loc[index][self.TEXT_COLUMN]
        encoded = self.tokenizer(selected_comment,
                                 padding='max_length',
                                 return_token_type_ids=False,
                                 truncation=True,
                                 max_length=128,
                                 return_tensors='pt')
        encoded = {k: v.flatten() for k, v in encoded.items()}

        if SarcasmDataset.LABEL_COLUMN in self.df.columns:
            encoded['targets'] = torch.tensor(self.df.loc[index][SarcasmDataset.LABEL_COLUMN])

        return encoded
