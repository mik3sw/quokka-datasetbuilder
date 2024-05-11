import json

class ChunkingConfig:
    def __init__(self, method: str, max_lenght: int):
        if max_lenght <= 0:
            raise ValueError("Error, max_lenght is expected to be greater than 0")
        
        self.method = method
        self.max_lenght = max_lenght

        