import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.datasetbuilder.classes.model_config import ModelConfig
from typing import Tuple
import configparser
from rich.console import Console

CONFIG = configparser.ConfigParser()
CONFIG.read("src/datasetbuilder/prompts.ini")

CHATML = "chatml"
LLAMA = "llama"
ZEPHYR = "zephyr"


def initialize_model_tokenizer(model_config: ModelConfig, console: Console) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    return None, None
    try:
        device = torch.device(model_config.device)
        console.log(f"Setting device [bold green]{str(device)}[/bold green]")
    except:
        console.log("Error with torch device, setting default")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.log(f"[bold red]ERROR[/bold red] - Setting default device [bold green]{str(device)}[/bold green]")
    
    console.log("Loading model...")
    params = {
        "torch_dtype": model_config.dtype,
        "trust_remote_code": model_config.trust_remote_code,
        "low_cpu_mem_usage": model_config.low_cpu_mem_usage,
        "quantization_config": model_config.quantization_config
    }
    params = {k: v for k, v in params.items() if v is not None}

    if model_config.use_flash_attn:
        console.log("Using [bold pink]Flash Attention[/bold pink]")
        params["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(model_config.hf_model_id, **params).eval()
    console.log("[bold green]Model succesfully loaded![/bold green]")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_tokenizer_id, )

    return model, tokenizer


def generate_answer(prompt: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, chat_template: str, max_new_tokens: int=100, device: str="cuda") -> str:
    return "answer"
    if device == "cuda":
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    elif device == "cpu":
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cpu()
    else:
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    if chat_template == CHATML:
        return decoded[0].split("<|im_start|> assistant")[1].replace("<|im_end|>", "")
    if chat_template == LLAMA:
        return decoded[0].split("[/INST]")[1].replace("</s>", "")
    if chat_template == ZEPHYR:
        return decoded[0].split("<|assistant|>")[1]


def format_json_prompt(context: str, chat_template: str, language: str, num_questions: int) -> str:
    #return '{"q1": "domanda 1", "q2": "domanda 2", "q3": "domanda3"}'
    strings = [f"q{i}" for i in range(1, num_questions + 1)]
    json_fields = ", ".join(strings)
    if chat_template == CHATML:
        return "<|im_start|>user\n" + CONFIG[language.upper()]["json_generation"].format(context, json_fields) + "<|im_end|>\n<|im_start|>assistant"
    if chat_template == LLAMA:
        return "[INST]" + CONFIG[language.upper()]["json_generation"].format(context, json_fields) + "[/INST]"
    if chat_template == ZEPHYR:
        return "<|user|>\n" + CONFIG[language.upper()]["json_generation"].format(context, json_fields) + "\n<|assistant|>"


def format_answer_prompt(context: str, question: str, chat_template: str, language: str) -> str:
    if chat_template == CHATML:
        return "<|im_start|>user\n" + CONFIG[language.upper()]["answer_generation"].format(context, question) + "<|im_end|>\n<|im_start|>assistant"
    if chat_template == LLAMA:
        return "[INST]" + CONFIG[language.upper()]["answer_generation"].format(context, question) + "[/INST]"
    if chat_template == ZEPHYR:
        return "<|user|>\n" + CONFIG[language.upper()]["answer_generation"].format(context, question) + "\n<|assistant|>"

