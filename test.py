from src.datasetbuilder.classes.generation_config import GenerationConfig
from src.datasetbuilder.classes.loader_config import LoaderConfig
from src.datasetbuilder.classes.model_config import ModelConfig
from src.datasetbuilder.dataset_builder import DatasetBuilder
import multiprocessing

loader_config = LoaderConfig(dataset="./test")
model_config = ModelConfig(hf_model_id="./model", trust_remote_code=True, use_flash_attn=True, device="cuda", chat_template="chatml")
generation_config = GenerationConfig(
    loader_config=loader_config,
    model_config=model_config,
    num_rows=100000,
    rows_per_chunk=3,
    language="IT",
    agents=3
)



if __name__ == '__main__':
    multiprocessing.freeze_support()
    db = DatasetBuilder(generation_config=generation_config)
    db.build()  