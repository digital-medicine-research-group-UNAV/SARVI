import pandas as pd
from torch import Tensor
from asyncio import Semaphore
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any

from .config import AppPaths

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