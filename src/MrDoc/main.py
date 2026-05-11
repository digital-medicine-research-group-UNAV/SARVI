import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import signal
import torch
import asyncio
import argparse
import contextlib
from .models.schemas import LLMConfig, PipelineContext, DisabledOptionError
from .config import paths
from .logging_redirect import enable_stdout_logging
from .core.create_jsons_per_llm_model import run_docx_to_jsons_deterministic_sync, run_docx_to_jsons_genrative_sync, run_docx_to_jsons_deterministic_async, run_docx_to_jsons_genrative_async
from .core.complete_excel_per_llm_model import run_jsons_to_xlsx_sync, run_jsons_to_xlsx_async


def terminate_process(**kwargs) -> None:
    with contextlib.suppress(Exception):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    time.sleep(2)
    os.kill(os.getpid(), signal.SIGTERM)


def run(ctx: PipelineContext, tarea: str, ussage: str, modo: str, folder_and_archive_name: str) -> None:
    ############################################
    if ctx.llm_config.lora_model != None:
        raise DisabledOptionError("Actually disabled, please do not select any LoRA model")
    if modo == "async":
        raise DisabledOptionError("Actually disabled, please do not run in async mode")
    ############################################

    signal.signal(signal.SIGINT, terminate_process)

    log_file = ctx.paths.logs_dir / f"{tarea}_{modo.upper()}_{folder_and_archive_name}.log"
    enable_stdout_logging(log_file)

    print(f"Ejecutando {tarea} // modo {modo} // ussage {ussage}...")

    if tarea == "docx_to_jsons":
        if modo == "sync":
            if ussage == "deterministic":
                return run_docx_to_jsons_deterministic_sync(ctx)
            else:
                run_docx_to_jsons_genrative_sync(ctx)
        else:
            if ussage == "deterministic":
                asyncio.run(run_docx_to_jsons_deterministic_async(ctx))
            else:
                asyncio.run(run_docx_to_jsons_genrative_async(ctx))

    elif tarea == "jsons_to_xlsx":
        if modo == "sync":
            run_jsons_to_xlsx_sync(ctx)
        else:
            asyncio.run(run_jsons_to_xlsx_async(ctx))

    print("Ejecución finalizada.")
    # terminate_process()    


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
        help="Selecciona que tipo de modo quieres usar para ejecutar la tarea. SOLAMENTE TENDRÁ USO EN LA TAREA `docx_to_jsons`"
    )

    parser.add_argument(
        "--deterministic_use_llm_for_corrections",
        type=bool,
        required=False,
        choices=[True, False],
        default=False,
        help="Si se usa el modo determinista, declarar si se quiere utilizar un LLM para posibles correcciones"
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
        lora_model=args.lora_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_threads=args.num_threads,
        num_interop_threads=args.num_interop_threads
    )
    ctx = PipelineContext(
        ussage=args.ussage,
        deterministic_use_llm_for_corrections=args.deterministic_use_llm_for_corrections,
        folder_and_archive_name=args.folder_and_archive_name,
        json_parse=args.json_parse,
        llm_config=cfg_llm,
        MAX_CONCURRENCY=args.max_concurrency,
        paths=paths,
        cie_10_version=args.cie_10_version
    )

    run(ctx, args.tarea, args.ussage, args.modo, args.folder_and_archive_name)

if __name__ == "__main__":
    main()