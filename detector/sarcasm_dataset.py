from torch.utils.data import Dataset


class SarcasmDataset(Dataset):
    """
    Dataset for all Reddit Sarcasm text, text length support is dependent on the underlying model.
    """
    TEXT_COLUMN = 'comment'
    LABEL_COLUMN = 'label'

    def __init__(self, encoded: dict):
        super().__init__()

        self.encoded = encoded

    def __len__(self):
        return next(iter(self.encoded.values())).shape[0]

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.encoded.items()}
