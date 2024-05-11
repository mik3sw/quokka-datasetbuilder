from src.datasetbuilder.classes.chunking_config import ChunkingConfig
import json

class LoaderConfig:
    def __init__(self, dataset: str, 
                 is_hf_dataset: bool=True, 
                 text_column:str="text", 
                 chunking_config:ChunkingConfig=None,
                 HF_token:str=None,
                 trust_remote_code: bool = True,
                 split:str = "train"):
        
        self.dataset = dataset
        self.is_hf_dataset = is_hf_dataset
        self.text_column = text_column
        self.chunking_config = chunking_config
        self.HF_token = HF_token
        self.trust_remote_code = trust_remote_code
        self.split = split
        