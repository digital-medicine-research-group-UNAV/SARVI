import pandas as pd
from torch import Tensor
from asyncio import Semaphore
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any

from .neural_networks import ICD10Predictor_HS_Head, ICD10Predictor_NO_HS, ICD10Predictor_HS_CrossEntropyLoss, SpanClassifier
from ..config import AppPaths

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
    modelo_ner: list[SpanClassifier]|None
    node_list: dict|None
    modelo_icd10_head: list[ICD10Predictor_HS_Head|ICD10Predictor_NO_HS]|None
    modelo_icd10_prediction: list[ICD10Predictor_HS_CrossEntropyLoss|None]|None
    label2id_ICD10: list[dict]|None
    id2label_ICD10: list[dict]|None
    id_no_hs_to_id_hs: dict|None
    icd10_thresholds: dict|None

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
    deterministic_use_llm_for_corrections: bool = False
    device: str

    # MUTABLES
    folder_and_archive_name: str
    json_parse: bool = True
    llm_config: LLMConfig
    cie_10_version: str
    
    # MUTABLES (async ONLY)
    MAX_CONCURRENCY: int = 5

    # Variables auxiliares
    vars: dict[str, Any] = Field(default_factory=dict)