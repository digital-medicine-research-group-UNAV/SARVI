import os
import tqdm
import torch
import ollama
import asyncio
from peft import PeftModel
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.lora.request import LoRARequest

from ..models import LLMConfig
from ..io.reader import read_json_single

#################################################################################################################
class LLMTransformersWrapper:
    def __init__(self, model, tok, device):
        self.model = model
        self.tok = tok
        self.device = device

    def invoke(self, messages, max_new_tokens=2816, do_sample=False, temperature=0.0, **kwargs):
        messages_dict = [
            {"role": "assistant", "content": messages[0]},
            {"role": "user",   "content": messages[1]},
        ]
        
        prompt_text = self.tok.apply_chat_template(
            messages_dict,
            tokenize=False,
            truncation=True,
            max_length=max_new_tokens,
            padding=False,
            add_generation_prompt=True,
        )

        inputs = self.tok(prompt_text, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            # temperature=temperature,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
        )

        with torch.no_grad():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model.generate(**inputs, **gen_kwargs)

        return self.tok.decode(outputs[0], skip_special_tokens=True)
    

    async def ainvoke(self, messages, max_new_tokens=8192, do_sample=False, temperature=0.0, **kwargs):
        return await asyncio.to_thread(self.invoke, messages, max_new_tokens, do_sample, temperature)


class LLMOllamaWrapper:
    def __init__(self, model, lora_model):
        if lora_model is not None:
            raise SystemError("LoRA is not supported with Ollama. Please do not declare any LoRA implementation")
        self.model = model
        ollama.pull(model)

    def invoke(self, messages: list, stream: bool = False, **kwargs):
        """
        Ejecuta una generación con ollama.generate, combinando los prompts.
        """
        response = ollama.generate(
            model=self.model,
            system=messages[0],
            prompt=messages[1],
            stream=stream
        )
        return response["response"]
    
    async def ainvoke(self, messages: list, stream: bool = False, **kwargs):
        return await asyncio.to_thread(self.invoke, messages, stream)
    

class LLMvLLMWrapper:
    def __init__(self, llm, lora_model, model_name):
        self.llm = llm
        self.lora_model = lora_model
        self.model_name = model_name
        self._original_tqdm = tqdm.tqdm

    def _silent_tqdm(self, *args, **kwargs):
        kwargs["disable"] = True
        return self._original_tqdm(*args, **kwargs)

    def invoke(self, messages: list, **kwargs):
        prompt = [
            {"role": "assistant", "content": [{"type": "text", "text": messages[0]}]},
            {"role": "user",   "content": [{"type": "text", "text": messages[1]}]},
        ]

        json_schema_llm_response = read_json_single(kwargs.get("json_schema"))
        sampling_params = SamplingParams(temperature=0, max_tokens=16384, structured_outputs=StructuredOutputsParams(json=json_schema_llm_response))
        
        if self.lora_model is not None:
            if "BIO_QA_ITTF" in self.lora_model:
                # base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
                # lora_merged = PeftModel.from_pretrained(base_model, "JulenRM/gpt-oss-20b_BIO_QA_ITTF_LoRA", adapter_name="gpt-oss-20b_BIO_QA_ITTF_LoRA")
                # lora_merged.load_adapter(self.lora_model, adapter_name=self.lora_model.split("/")[1])
                # lora_merged.add_weighted_adapter(
                #     adapters=["gpt-oss-20b_BIO_QA_ITTF_LoRA", self.lora_model.split("/")[1]],
                #     weights=[1.0, 1.0],
                #     adapter_name="merged_lora",
                #     combination_type="linear"
                # )
                # lora_merged.set_adapter("merged_lora")
                # lora_merged.save_pretrained("tmp/merged_lora")
                lora_req = LoRARequest(self.lora_model.split("/")[1], 1, "tmp/merged_lora/merged_lora")
            else:
                lora_req = LoRARequest(self.lora_model.split("/")[1], 1, self.lora_model)

            tqdm.tqdm = self._silent_tqdm
            outputs  = self.llm.chat(prompt, sampling_params, use_tqdm=False, lora_request=lora_req)
            tqdm.tqdm = self._original_tqdm
        
        else:
            tqdm.tqdm = self._silent_tqdm
            outputs  = self.llm.chat(prompt, sampling_params, use_tqdm=False)
            tqdm.tqdm = self._original_tqdm

        return outputs[0].outputs[0].text
    
    async def ainvoke(self, messages: list, stream: bool = False, **kwargs):
        return await asyncio.to_thread(self.invoke, messages, stream)

#################################################################################################################

def load_llm(cfg: LLMConfig) -> ChatOpenAI | LLMTransformersWrapper | LLMOllamaWrapper | LLMvLLMWrapper:
    if (cfg.service == "openai"):
        return load_llm_openai_langchain_framework(cfg)
    elif (cfg.service == "ollama"):
        return load_llm_ollama_local_framework(cfg)
    elif (cfg.service == "vllm"):
        return load_llm_vllm_local_framework(cfg)
    else:
        return load_llm_transformers_local_framework(cfg)

#################################################################################################################

def load_llm_openai_langchain_framework(cfg: LLMConfig) -> ChatOpenAI:
    """
    Load OpenAI Largue Language Model from Langchain

    If any **4o models** are used, `reasoning={"effort": "low"}` is activated

    If any **o3 models** are used, `max_tokens=128000` are used

    Uses .env values

    Parameters
    ----------
        `cfg`: LLMConfig
            - Config of the model instantiated

    Returns
    -------
        `llm`: ChatOpenAI
            - Object from Langchain that is used as a llm
    """
    if "4o" in cfg.model: 
        llm = ChatOpenAI(
            model=cfg.model,
            reasoning={"effort": "low"},
            temperature=0,
            max_tokens=16384
        )
    elif "o3" in cfg.model: 
        llm = ChatOpenAI(
            model=cfg.model,
            temperature=0,
            max_tokens=128000
        )
    else:
        llm = ChatOpenAI(
            model=cfg.model,
            temperature=0,
            max_tokens=16384
        )

    return llm


def load_llm_transformers_local_framework(cfg: LLMConfig):
    """
    Load transformers Large Language Model locally with LangChain

    Parameters
    ----------
        `cfg`: LLMConfig
            - Config of the model instantiated

    Returns
    -------
        `llm`: LLMTransformersWrapper
            - Wrapper que permite ejecutar inferencias directamente con llm.invoke()
    """
    torch.set_num_threads(cfg.num_threads)
    torch.set_num_interop_threads(cfg.num_interop_threads)

    tok = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, 
        device_map="auto",
        quantization_config=Mxfp4Config(dequantize=True) if "gpt-oss" in cfg.model else None
    )

    if cfg.lora_model is not None:
        if "BIO_QA_ITTF" in cfg.lora_model:
            # Merfe the ITTF model prior to the IE LoRA
            model = PeftModel.from_pretrained(
                model,
                "JulenRM/gpt-oss-20b_QA_ITTF_LoRA",
                device_map="auto",
            )
            model = model.merge_and_unload()
        
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model,
            device_map="auto"
        )

    return LLMTransformersWrapper(model, tok, cfg.device)

def load_llm_ollama_local_framework(cfg):
    """
    Inicializa un modelo Ollama local y devuelve un objeto con .invoke()
    que permite generar texto como un LLM normal.

    Parameters
    ----------
        `cfg`: LLMConfig
            - Config of the model instantiated

    Returns
    -------
        `llm`: LLMOllamaWrapper
            - Wrapper que permite ejecutar inferencias directamente con llm.invoke()
    """

    try:
        _ = ollama.list()
    except Exception as e:
        raise RuntimeError(f"No se pudo conectar con Ollama. Asegúrate de que el servicio está activo.\nError: {e}")

    return LLMOllamaWrapper(cfg.model, cfg.lora_model)

def load_llm_vllm_local_framework(cfg):
    """
    Inicializa un modelo vLLM local y devuelve un objeto con .invoke()
    que permite generar texto como un LLM normal.

    Parameters
    ----------
        `cfg`: LLMConfig
            - Config of the model instantiated

    Returns
    -------
        `llm`: LLMvLLMWrapper
            - Wrapper que permite ejecutar inferencias directamente con llm.invoke()
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    llm = LLM(model=cfg.model, gpu_memory_utilization=0.6,
        enable_prefix_caching=True,
        limit_mm_per_prompt={
            "image": {"count": 0}, 
            "video": {"count": 0}
            },
        enable_sleep_mode=True,
        enable_lora=True if cfg.lora_model is not None else False
        ) 

    return LLMvLLMWrapper(llm, cfg.lora_model, cfg.model)