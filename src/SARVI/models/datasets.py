import torch
from torch.utils.data import Dataset

class SpanDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Token-level embeddings
        token_embeddings = row["Embeddings"]
        if token_embeddings.dim() == 1:
            token_embeddings = token_embeddings.unsqueeze(0)
        span_repr, _ = torch.max(token_embeddings, dim=0)
        # CLS
        cls_repr = row["CLS Embedding"]
        # Width
        span_width = torch.tensor(token_embeddings.size(0), dtype=torch.long)

        return span_repr, cls_repr, span_width

class ICD10Dataset(Dataset):
    def __init__(self, inputs, queries, all_desc):
        self.inputs = inputs
        self.queries = queries
        self.all_desc = all_desc

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.queries[idx], self.all_desc[idx]
