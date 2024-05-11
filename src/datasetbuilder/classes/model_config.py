from transformers import BitsAndBytesConfig
import torch
import json

class ModelConfig():
    def __init__(self, hf_model_id: str, 
                 hf_tokenizer_id: str=None, 
                 dtype: torch.dtype=torch.bfloat16, 
                 use_flash_attn: bool=True, 
                 device: str="cuda", 
                 quantization_config: BitsAndBytesConfig=None, 
                 chat_template: str="chatml",
                 trust_remote_code: bool=False,
                 low_cpu_mem_usage: bool=False,
                 max_new_token: int=2048
                 ):
        self.hf_model_id = hf_model_id
        if hf_tokenizer_id == None:
            self.hf_tokenizer_id = hf_model_id
        else:
            self.hf_tokenizer_id = hf_tokenizer_id
        self.dtype = dtype
        self.use_flash_attn = use_flash_attn
        self.device = device
        self.quantization_config = quantization_config
        self.chat_template = chat_template
        self.trust_remote_code = trust_remote_code
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.max_new_token = max_new_token
    
