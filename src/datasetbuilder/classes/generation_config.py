from src.datasetbuilder.classes.loader_config import LoaderConfig
from src.datasetbuilder.classes.model_config import ModelConfig
import json


LANGUAGES = ["en", "it"]

class GenerationConfig:
    def __init__(self, loader_config: LoaderConfig, model_config: ModelConfig, num_rows: int=10000, rows_per_chunk: int=3, language: str="en", agents: int=1):
        if num_rows <= 0:
            raise ValueError("Error, num_rows is expected to be greater than 0")
        
        if rows_per_chunk <= 0:
            raise ValueError("Error, rows_per_chunk is expected to be greater than 0")
        
        if language not in LANGUAGES:
            raise ValueError("Error, language is expected to be one of these: " + LANGUAGES)
        
        if agents <= 0:
            raise ValueError("Error, agents is expected to be greater than 0")
        
        self.loader_config = loader_config
        self.model_config = model_config
        self.num_rows = num_rows
        self.rows_per_chunk = rows_per_chunk
        self.language = language
        self.agents = agents



        
        