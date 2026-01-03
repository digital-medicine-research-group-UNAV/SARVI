import os
import torch
import ollama
import asyncio
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoImageProcessor
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from ..models import LLMConfig


#################################################################################################################
class LLMTransformersWrapper:
    def __init__(self, model_obj, tok, img_proc, device):
        self.model_obj = model_obj
        self.tok = tok
        self.img_proc = img_proc
        self.device = device

    def invoke(self, messages, max_new_tokens=8192, do_sample=False, temperature=0.0, **kwargs):
        messages_dict = [
            {"role": "system", "content": [{"type": "text", "text": messages[0]}]},
            {"role": "user",   "content": [{"type": "text", "text": messages[1]}]},
        ]
        
        prompt_text = self.tok.apply_chat_template(
            messages_dict,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tok(prompt_text, return_tensors="pt").to(self.model_obj.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            # temperature=temperature,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
        )

        with torch.no_grad():
            outputs = self.model_obj.generate(**inputs, **gen_kwargs)

        return self.tok.decode(outputs[0], skip_special_tokens=True)
    

    async def ainvoke(self, messages, max_new_tokens=8192, do_sample=False, temperature=0.0, **kwargs):
        return await asyncio.to_thread(self.invoke, messages, max_new_tokens, do_sample, temperature)


class LLMOllamaWrapper:
    def __init__(self, model):
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
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, messages: list, **kwargs):
        messages_dict = [
            {"role": "system", "content": [{"type": "text", "text": messages[0]}]},
            {"role": "user",   "content": [{"type": "text", "text": messages[1]}]},
        ]

        prompt_text = self.llm.get_tokenizer().apply_chat_template(
            messages_dict,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(temperature=0, structured_outputs=StructuredOutputsParams(json=kwargs.get("json_schema")))
        outputs  = self.llm.generate(prompt_text, sampling_params)
        return outputs[0].outputs[0].text
    
    async def ainvoke(self, messages: list, stream: bool = False, **kwargs):
        return await asyncio.to_thread(self.invoke, messages, stream)

#################################################################################################################

def load_llm(cfg: LLMConfig) -> ChatOpenAI | LLMTransformersWrapper | LLMOllamaWrapper:
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
    img_proc = AutoImageProcessor.from_pretrained(cfg.model, use_fast=True)
    model_obj = AutoModelForImageTextToText.from_pretrained(cfg.model, dtype=torch.float32).to(cfg.device)

    return LLMTransformersWrapper(model_obj, tok, img_proc, cfg.device)

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

    return LLMOllamaWrapper(cfg.model)

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

    llm = LLM(model=cfg.model, gpu_memory_utilization=0.4,
        enable_prefix_caching=True,
        limit_mm_per_prompt={
            "image": {"count": 0}, 
            "video": {"count": 0}
            },                                      
        runner="generate",
        enable_sleep_mode=True
        ) 

    return LLMvLLMWrapper(llm)