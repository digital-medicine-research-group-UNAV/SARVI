from __future__ import annotations

import pandas as pd
from torch import Tensor
from asyncio import Semaphore
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .neural_networks import ICD10Predictor_HS_Head, ICD10Predictor_NO_HS, ICD10Predictor_HS_CrossEntropyLoss, BIO_NERClassifier, Span_NERClassifier, REClassifier, ATTClassifier
    from ..config import AppPaths
    from hierarchicalsoftmax import SoftmaxNode
    NERModel: TypeAlias = BIO_NERClassifier | Span_NERClassifier
    REModel: TypeAlias = REClassifier
    ATTModel: TypeAlias = ATTClassifier
    ICD10HeadModel: TypeAlias = ICD10Predictor_HS_Head | ICD10Predictor_NO_HS
    ICD10PredictionModel: TypeAlias = ICD10Predictor_HS_CrossEntropyLoss | None
else:
    AppPaths = Any
    NERModel = Any
    REModel = Any
    ATTModel = Any
    ICD10HeadModel = Any
    ICD10PredictionModel = Any
    SoftmaxNode = Any

class DisabledOptionError(Exception):
    pass
  
class LLMConfig(BaseModel):
    service: str
    model: str
    lora_model: str|None = None
    device: str
    num_threads: int = 16
    num_interop_threads: int = 2
    gpu_memory_utilization: float = 0.88
    max_num_seqs: int = 32

class DOCXToJSONSConfig(BaseModel):
    report_list: list[Path]
    prompt: str|None
    llm: Any
    semaforo: Semaphore
    modelo_ner: list[NERModel]|None
    modelo_re: list[REModel]|None
    modelo_att: list[ATTModel]|None
    tokenizer_re: list[Any]|None
    tokenizer_att: list[Any]|None
    node_list: dict|None
    modelo_icd10_head: list[ICD10HeadModel]|None
    modelo_icd10_prediction: list[ICD10PredictionModel]|None
    label2id_NER: list[dict]|None
    id2label_NER: list[dict]|None
    label2id_RE: list[dict]|None
    id2label_RE: list[dict]|None
    label2id_ATT: list[dict]|None
    id2label_ATT: list[dict]|None
    label2id_ICD10: list[dict]|None
    id2label_ICD10: list[dict]|None
    id_no_hs_to_id_hs: dict|None
    re_token_distance_bins: dict|None
    re_negative_difficulty_by_bin: dict|None
    icd10_thresholds: dict|None
    root: SoftmaxNode|None

    model_config = {
        "arbitrary_types_allowed": True
    }

class JSONToXLSXConfig(BaseModel):
    df_reference: pd.DataFrame
    CIE10_full_list: list[str]
    df_reference_embeddings: Tensor
    llm: Any
    semaforo: Semaphore
    doc_lista: dict[str, str]
    prompts: dict[str, str]

    model_config = {
        "arbitrary_types_allowed": True
    }

class PipelineContext(BaseModel):
    paths: AppPaths
    ussage: str
    device: str
    base_encoder_name: str

    # MUTABLES
    folder_and_archive_name: str
    json_parse: bool = True
    llm_config: LLMConfig
    cie_10_version: str
    
    # MUTABLES (async ONLY)
    MAX_CONCURRENCY: int = 5

    # Variables auxiliares
    vars: dict[str, Any] = Field(default_factory=dict)
