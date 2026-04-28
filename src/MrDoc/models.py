import torch
import pandas as pd
from torch import Tensor
import torch.nn as nn
from asyncio import Semaphore
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any
from torch.utils.data import Dataset

from .config import AppPaths

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
        # Label
        label = torch.tensor(row["label_id"], dtype=torch.long)

        return span_repr, cls_repr, span_width, label

class SpanClassifier(nn.Module):
    def __init__(self, span_dim, cls_dim, num_classes, max_span_width, width_emb_dim=25, dropout=0.1):
        super().__init__()

        # Width embeddings (TRAINABLE)
        self.width_embeddings = nn.Embedding(max_span_width + 1, width_emb_dim)
        nn.init.xavier_uniform_(self.width_embeddings.weight)

        # Clasificador (TRAINABLE)
        input_dim = span_dim + cls_dim + width_emb_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, num_classes)
        )

    def forward(self, span_repr, cls_repr, span_width):
        """
        span_repr:  [N, span_dim]
        cls_repr:   [N, cls_dim]
        span_width: [N]
        """

        # To NOT train prior encoder data
        span_repr = span_repr.detach()
        cls_repr = cls_repr.detach()

        width_emb = self.width_embeddings(span_width)  # [N, width_emb_dim]

        x = torch.cat([span_repr, cls_repr, width_emb], dim=-1)
        logits = self.classifier(x)

        return logits

class DisabledOptionError(Exception):
    pass

class LLMConfig(BaseModel):
    service: str
    model: str
    lora_model: str|None = None
    device: str
    num_threads: int = 16
    num_interop_threads: int = 2

class DOCXToJSONSConfig(BaseModel):
    report_list: list[Path]
    prompt: str|None
    llm: Any
    semaforo: Semaphore
    modelo: SpanClassifier|None

    model_config = {
        "arbitrary_types_allowed": True
    }

class JSONToXLSXConfig(BaseModel):
    df_reference: pd.DataFrame
    CIE10_full_list: list[str]
    df_reference_embeddings: Tensor
    llm: Any
    semaforo: Semaphore
    docx_lista: dict[str, str]
    prompts: dict[str, str]

    model_config = {
        "arbitrary_types_allowed": True
    }

class PipelineContext(BaseModel):
    paths: AppPaths
    ussage: str

    # MUTABLES
    folder_and_archive_name: str
    json_parse: bool = True
    llm_config: LLMConfig
    cie_10_version: str
    
    # MUTABLES (async ONLY)
    MAX_CONCURRENCY: int = 5

    # Variables auxiliares
    vars: dict[str, Any] = Field(default_factory=dict)