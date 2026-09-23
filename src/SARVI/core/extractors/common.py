from __future__ import annotations

import contextlib
import gc as py_gc
import os
import random

import numpy as np
import pandas as pd
import torch
from transformers import set_seed as transformers_set_seed

from ...data_io.reader import read_docx_list, read_txt_list

SEED = 8


def seed_everything(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def cleanup_cuda(*objects) -> None:
    for obj in objects:
        if isinstance(obj, torch.nn.Module):
            with contextlib.suppress(Exception):
                obj.eval()
                obj.zero_grad(set_to_none=True)
                for p in obj.parameters(recurse=True):
                    p.grad = None
                obj.to("cpu")

        elif isinstance(obj, torch.Tensor):
            with contextlib.suppress(Exception):
                obj.detach_()

        elif isinstance(obj, torch.optim.Optimizer):
            with contextlib.suppress(Exception):
                obj.zero_grad(set_to_none=True)
                obj.state.clear()

    py_gc.collect()

    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def load_input_dataframe(ctx) -> pd.DataFrame:
    """
    Load the EHRs to analyze into a dataframe

    Parameters
    ----------
        `ctx`: PipelineContext
            - Context with the types of variables to be used

    Returns
    -------
        ``: pd.DataFrame
            - DataFrame with the EHR to analyze
    """
    input_folder = ctx.paths.data_input / ctx.folder_and_archive_name

    if any(input_folder.glob("*.docx")):
        dict_data = read_docx_list(input_folder)
    elif any(input_folder.glob("*.txt")):
        dict_data = read_txt_list(input_folder)
    else:
        raise FileNotFoundError("No .docx or .txt files found")

    return pd.DataFrame(list(dict_data.items()), columns=["archivo_origen", "Text"])
