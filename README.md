<h1 align="center"><strong>SARVI</strong></h1>
<h2 align="center">System for Automated Recognition and
Validation of ICD-10 Diagnoses</h2>

# Instalation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
- If in windows, subsistute the first and second command
    - ```bash
        py -3.12 -m venv .venv
        .\.venv\Scripts\Activate.ps1
        ```

> The command `pip install -e .` allows to just do `import SARVI` into any project

- Rename the file `private/data.env.template` to `private/data.env` and add the corresponding values 

# How to execute
- **Package style** -> see `notebooks/package_functionality.ipynb`
- **Command style** -> 
```bash
python -m src.SARVI.main --tarea docx_to_jsons --cie_10_version 2026 --modo sync --llm_service vllm --llm_model openai/gpt-oss-20b --folder_and_archive_name Full_CUN_V0.1
```

## Minimum variables to stablish

1. `tarea`

2. `modo`

3. `llm_service`

4. `llm_model`

5. `folder_and_archive_name`

6. `device`

7. `paths`

> 6. and 7. are only necessary when executing in package style

## Order to execute

0. All the `.docx` files to work with, must be in the directory `data/input/{folder_and_archive_name}`. Important to keep the same *folder_and_archive_name* through all the execution

1. `docx_to_jsons`

2. `jsons_to_xlsx`

# Available models

- **ASYNC** -> Cloud LLMs
    - OpenAI API *(LangChain)* - `gpt-5-chat-latest` or `gpt-5-nano-2025-08-07`

- **SYNC** -> Local LLMs
    - MedGemma *(Transformers)* - `google/medgemma-4b-it`
    - GPT-oss *(Ollama)* - `gpt-oss:20b`
    - GPT-oss *(vLLM)* - `openai/gpt-oss-20b`

> Other models or exchanges between ASYNC and SYNC are NOT TESTED

# **Warnings**

- Although implemented, **--lora_model** is actually disabled and will return error
- Although implemented, **--modo** will not work in **async** mode. It is actually disabled and will return error