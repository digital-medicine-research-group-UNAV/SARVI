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
python -m src.SARVI.main --tarea s1 --cie_10_version 2026 --ussage deterministic --modo sync --llm_service vllm --llm_model openai/gpt-oss-20b --folder_and_archive_name EHRs
```

## Variables to stablish

1. `tarea` -> **Minimum** / *Choices: "s1", "s2"*
2. `cie_10_version` -> **Minimum** / *Choices: "2018", "2024", "2026"*
3. `ussage` -> **Minimum** / *Choices: "deterministic", "generative"*
4. `modo` -> **Minimum** / *Choices: "sync", "async"*
5. `llm_service` -> **Minimum** / *Choices: "openai", "ollama", "transformers", "vllm"*
6. `llm_model` -> **Minimum**
7. `lora_model` -> *Default: None*
8. `folder_and_archive_name` -> **Minimum**
9. `base_encoder_name` -> *Default: "IIC/RigoBERTa-Clinical"*
10. `json_parse` -> *Default: "True"*
11. `max_concurrency` -> *Default: 5*
12. `num_threads` -> *Default: 16*
13. `num_interop_threads` -> *Default: 2*
14. `gpu_memory_utilization` -> *Default: 0.88*
15. `max_num_seqs` -> *Default: 32*
16. `device`
17. `paths`

> 16. and 17. are only necessary when executing in package style

## Order to execute

0. All the `.docx` or `.txt` files to work with, must be in the directory `data/input/{folder_and_archive_name}`. Important to keep the same *folder_and_archive_name* through all the execution
1. `s1`
2. `s2`

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
