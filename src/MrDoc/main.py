import os
import gc
import time
import signal
import torch
import asyncio
import argparse
import contextlib
from .models import LLMConfig, PipelineContext
from .config import paths
from .logging_redirect import enable_stdout_logging
from .core.create_jsons_per_llm_model import run_docx_to_jsons_sync, run_docx_to_jsons_async
from .core.complete_excel_per_llm_model import run_jsons_to_xlsx_sync, run_jsons_to_xlsx_async

def terminate_process(**kwargs) -> None:
    with contextlib.suppress(Exception):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    time.sleep(2)
    os.kill(os.getpid(), signal.SIGTERM)


def run(ctx: PipelineContext, tarea: str, modo: str, folder_and_archive_name: str) -> None:
    signal.signal(signal.SIGINT, terminate_process)

    log_file = ctx.paths.logs_dir / f"{tarea}_{modo.upper()}_{folder_and_archive_name}.log"
    enable_stdout_logging(log_file)

    print(f"Ejecutando {tarea} // modo {modo}...")

    if tarea == "docx_to_jsons":
        if modo == "sync":
            run_docx_to_jsons_sync(ctx)
        else:
            asyncio.run(run_docx_to_jsons_async(ctx))

    elif tarea == "jsons_to_xlsx":
        if modo == "sync":
            run_jsons_to_xlsx_sync(ctx)
        else:
            asyncio.run(run_jsons_to_xlsx_async(ctx))

    print("Ejecución finalizada.")
    terminate_process()    


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta una de los dos posibles opciones: DOCX -> JSONs // JSONs -> XLSX. Además de decidir si se quiere en modo sync // async"
    )
    parser.add_argument(
        "--tarea",
        type=str,
        required=True,
        choices=["docx_to_jsons", "jsons_to_xlsx"],
        help="Tarea a ejecutar",
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
        "--folder_and_archive_name",
        type=str,
        required=True,
        help="Nombre de los archivos y carpetas creadas"
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

    args = parser.parse_args()

    cfg_llm = LLMConfig(
        service=args.llm_service, 
        model=args.llm_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_threads=args.num_threads,
        num_interop_threads=args.num_interop_threads
    )
    ctx = PipelineContext(
        folder_and_archive_name=args.folder_and_archive_name,
        json_parse=args.json_parse,
        llm_config=cfg_llm,
        MAX_CONCURRENCY=args.max_concurrency,
        paths=paths
    )

    run(ctx, args.tarea, args.modo, args.folder_and_archive_name)

if __name__ == "__main__":
    main()