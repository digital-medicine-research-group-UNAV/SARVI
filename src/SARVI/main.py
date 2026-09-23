import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import gc
import signal
import torch
import asyncio
import argparse
import contextlib
import logging
from transformers.utils import logging as hf_logging

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
hf_logging.disable_default_handler()
hf_logging.disable_propagation()

from .models.schemas import LLMConfig, PipelineContext, DisabledOptionError
from .config import paths
from .logging_redirect import enable_stdout_logging
from .core.s1 import (
    run_s1_deterministic_sync, run_s1_genrative_sync, 
    run_s1_deterministic_async, run_s1_genrative_async
)
from .core.s2 import (
    run_s2_sync,
    run_s2_async
)

def running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        return shell_name in {"ZMQInteractiveShell", "Shell"}
    except Exception:
        return False

def terminate_process(signum=None, frame=None, exit_code: int = 0, **kwargs) -> None:
    with contextlib.suppress(Exception):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()
            
    if running_in_notebook():
        return
    if signum is not None:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    else:
        raise SystemExit(exit_code)


def run(ctx: PipelineContext, tarea: str, ussage: str, modo: str, folder_and_archive_name: str) -> None:

    ############################################
    if ctx.llm_config.lora_model != None:
        raise DisabledOptionError("Actually disabled, please do not select any LoRA model")
    if modo == "async":
        raise DisabledOptionError("Actually disabled, please do not run in async mode")
    ############################################

    signal.signal(signal.SIGTERM, terminate_process)
    signal.signal(signal.SIGINT, terminate_process)

    log_file = ctx.paths.logs_dir / folder_and_archive_name / f"{tarea}_{modo.upper()}.log"
    enable_stdout_logging(log_file)

    print(f"\033[1mEjecutando {tarea} // modo {modo} // ussage {ussage}...\033[0m")

    if tarea == "s1":
        if modo == "sync":
            if ussage == "deterministic":
                run_s1_deterministic_sync(ctx)
            else:
                run_s1_genrative_sync(ctx)
        else:
            if ussage == "deterministic":
                asyncio.run(run_s1_deterministic_async(ctx))
            else:
                asyncio.run(run_s1_genrative_async(ctx))

    elif tarea == "s2":
        if modo == "sync":
            run_s2_sync(ctx)
        else:
            asyncio.run(run_s2_async(ctx))

    print("\n\033[1mEjecución finalizada\033[0m")
    terminate_process()    


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta una de los dos posibles opciones: S1 (DOCX/TXT -> JSONs) // S2 (JSONs -> XLSX). Además de decidir si se quiere en modo sync // async"
    )

    parser.add_argument(
        "--tarea",
        type=str,
        required=True,
        choices=["s1", "s2"],
        help="Tarea a ejecutar",
    )

    parser.add_argument(
        "--cie_10_version",
        type=str,
        required=True,
        choices=["2018", "2024", "2026"],
        help="Seleccionar el año de versión de los códigos CIE10"
    )

    parser.add_argument(
        "--ussage",
        type=str,
        required=True,
        choices=["deterministic", "generative"],
        help="Selecciona que tipo de modo quieres usar para ejecutar la tarea. SOLAMENTE TENDRÁ USO EN LA TAREA `s1`"
    )

    parser.add_argument(
        "--modo",
        type=str,
        required=True,
        choices=["sync", "async"],
        help="Modo de ejecución: sync o async"
    )

    parser.add_argument(
        "--llm_service",
        type=str,
        required=True,
        choices=["openai", "ollama", "transformers", "vllm"],
        help="Servicio desde el cual inicializar y ejecutar el LLM"
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        required=True,
        help="Modelo de LLM  utilizar"
    )

    parser.add_argument(
        "--lora_model",
        type=str,
        required=False,
        default=None,
        help="LoRA a utilizar sobre el LLM si se quiere"
    )

    parser.add_argument(
        "--folder_and_archive_name",
        type=str,
        required=True,
        help="Nombre de los archivos y carpetas creadas"
    )

    parser.add_argument(
        "--base_encoder_name",
        type=str,
        required=False,
        default="IIC/RigoBERTa-Clinical",
        help="Modelo de encoder a utilizar en los modelos.\n\n\t\tAUNQUE NODIFICABLE, UNICAMENTE FUNCIONA SI SE DEJA COMO VALOR PREDETERMINADO YA QUE LOS MODELOS ESTÁN ENTRENADOS BAJO ESE MODELO"
    )

    parser.add_argument(
        "--json_parse",
        type=bool,
        required=False,
        default=True,
        help="Si se necesita realizar un parseo específico o no. Default es True"
    )

    parser.add_argument(
        "--max_concurrency",
        type=int,
        required=False,
        default=5,
        help="Maxima concurrencia si se usa async. Default es 5"
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        required=False,
        default=16,
        help="Numero de hilos a ejecutar. Default es 16"
    )

    parser.add_argument(
        "--num_interop_threads",
        type=int,
        required=False,
        default=2,
        help="Numero de hilos interoperables a ejecutar. Default es 2"
    )

    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        required=False,
        default=0.88,
        help="Fracción de memoria GPU que vLLM puede reservar. Default es 0.88"
    )

    parser.add_argument(
        "--max_num_seqs",
        type=int,
        required=False,
        default=32,
        help="Número máximo de secuencias concurrentes para vLLM. Default es 32"
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_llm = LLMConfig(
        service=args.llm_service, 
        model=args.llm_model,
        lora_model=args.lora_model,
        device=device,
        num_threads=args.num_threads,
        num_interop_threads=args.num_interop_threads,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs
    )
    ctx = PipelineContext(
        ussage=args.ussage,
        folder_and_archive_name=args.folder_and_archive_name,
        json_parse=args.json_parse,
        llm_config=cfg_llm,
        base_encoder_name=args.base_encoder_name,
        MAX_CONCURRENCY=args.max_concurrency,
        paths=paths,
        cie_10_version=args.cie_10_version,
        device=device
    )

    run(ctx, args.tarea, args.ussage, args.modo, args.folder_and_archive_name)

if __name__ == "__main__":
    main()