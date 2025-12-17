import traceback
from natsort import natsorted

from ..models import (
    PipelineContext,
    DOCXToJSONSConfig
)

from ..io.reader import (
    textwrap,
    cargar_docx_single
)
from ..io.writing import (
    create_intermediate_folder_name,
    write_json
)

from ..services.llm_loader import load_llm
from ..services.common import (
    tqdm,
    prompts as prompts_total
)
from ..services.sync_funcs import (
    procesar_docx as procesar_docx_SYNC
)
from ..services.async_funcs import (
    asyncio,
    procesar_docx as procesar_docx_ASYNC
)

def initialize_variables(ctx: PipelineContext):
    create_intermediate_folder_name(ctx)

    report_list = natsorted(p for p in (ctx.paths.data_input / ctx.folder_and_archive_name).iterdir() if p.is_file())
    prompt = textwrap.dedent(prompts_total["report_to_data"])
    llm = load_llm(ctx.llm_config)
    semaforo = asyncio.Semaphore(ctx.MAX_CONCURRENCY)

    config = DOCXToJSONSConfig(
        report_list=report_list,
        prompt=prompt,
        llm=llm,
        semaforo=semaforo
    )

    return config


def run_docx_to_jsons_sync(ctx: PipelineContext):
    config = initialize_variables(ctx)

    for report in tqdm(config.report_list, desc="Extrayendo CIE10 de archivos...", unit="informe"):
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                informe_texto = cargar_docx_single(report)
                respuesta_llm = procesar_docx_SYNC(informe_texto, report, config.prompt, config.llm, ctx.json_parse, ctx.paths.docs_dir)
                write_json(ctx, report, respuesta_llm)
                break
            except Exception as e:
                print(f"⚠️ Error al procesar informe {report}: {e!r}")
                traceback.print_exc()
                if attempt < max_retries:
                    print(f"↻ Reintentando ({attempt}/{max_retries})...")
                else:
                    print(f"❌ Falló definitivamente el informe: {report}\n")

async def run_docx_to_jsons_async(ctx: PipelineContext):
    config = initialize_variables(ctx)

    async def procesar_con_reintentos(report, max_retries=5):
        for attempt in range(1, max_retries + 1):
            try:
                informe_texto = cargar_docx_single(report)
                respuesta_llm = await procesar_docx_ASYNC(informe_texto, report, config.prompt, config.llm, ctx.json_parse, ctx.paths.docs_dir, config.semaforo)
                write_json(ctx, report, respuesta_llm)
                break
            except Exception as e:
                print(f"⚠️ Error al procesar informe {report}: {e!r}")
                traceback.print_exc()
                if attempt < max_retries:
                    print(f"↻ Reintentando ({attempt}/{max_retries})...")
                else:
                    print(f"❌ Falló definitivamente el informe: {report}\n")

    tareas = [procesar_con_reintentos(report) for report in config.report_list]

    for future in tqdm(asyncio.as_completed(tareas), total=len(tareas), desc="Extrayendo CIE10 de archivos...", unit="informe"):
        await future