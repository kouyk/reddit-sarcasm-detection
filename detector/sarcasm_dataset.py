import torch
from pandas import DataFrame
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class SarcasmDataset(Dataset):
    """
    Dataset for all Reddit Sarcasm text, text length support is dependent on the underlying model.
    """
    TEXT_COLUMN = 'comment'
    LABEL_COLUMN = 'label'
    PARENT_COLUMN = 'parent_comment'
    AUTHOR_COLUMN = 'author_cluster'
    SUBREDDIT_COLUMN = 'subreddit_cluster'
    SCORE_COLUMN = 'score'
    HOUR_COLUMN = 'hour'
    MONTH_COLUMN = 'month'

    def __init__(self,
                 df: DataFrame,
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: int = 512,
                 use_parent: bool = True,
                 no_extra: bool = False):
        super().__init__()

        self.df = df.copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_parent = use_parent
        self.no_extra = no_extra

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        selected = self.df.iloc[index]
        encoded = self.tokenizer(
            text=selected[SarcasmDataset.PARENT_COLUMN] if self.use_parent else selected[SarcasmDataset.TEXT_COLUMN],
            text_pair=selected[SarcasmDataset.TEXT_COLUMN] if self.use_parent else None,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoded = {k: v.flatten() for k, v in encoded.items()}
        if not self.no_extra:
            encoded['features'] = torch.cat([
                torch.as_tensor(selected[SarcasmDataset.SCORE_COLUMN], dtype=torch.float).unsqueeze(dim=0),
                one_hot(torch.as_tensor(selected[SarcasmDataset.MONTH_COLUMN]), num_classes=96),
                one_hot(torch.as_tensor(selected[SarcasmDataset.HOUR_COLUMN]), num_classes=24),
                one_hot(torch.as_tensor(selected[SarcasmDataset.AUTHOR_COLUMN]), num_classes=6),
                one_hot(torch.as_tensor(selected[SarcasmDataset.SUBREDDIT_COLUMN]), num_classes=5)
            ])

        if SarcasmDataset.LABEL_COLUMN in self.df.columns:
            encoded['targets'] = torch.tensor(selected[SarcasmDataset.LABEL_COLUMN])

        return encoded
