
![](quokka-banner.png)

This software can help you easily create instruct datasets from a custom domain


## Table of content
- [Installation](#installation)
- [Dataset](#installation)
- [Pipeline](#pipeline)
- [Config](#config)
- [Example](#example)


## Installation
```bash
# clone this repo
git clone http://...

cd quokka-datasetbuilder
pip install -r requirements.txt
```


## Dataset
This repo is still in a **work in progress state**.

For now the only supported format for the input dataset is:

```json
[
    {"text_field": "here the text", ...},
    ...
]
```

The output format is:

```json
{"question": "...", "context": "...", "answer": "..."}
{"question": "...", "context": "...", "answer": "..."}
{"question": "...", "context": "...", "answer": "..."}
...
{"question": "...", "context": "...", "answer": "..."}
```

## Pipeline

### How does it works?

The pipeline is pretty simple and composed by a few steps:
- take a row from the input dataset
- generate N questions based on the context
- generate an answer for each question generated (passing the row as context)
- save the generations

> NOTE:
> I do not have both questions and answers generated in the same step because of two complications:
> - Smaller LLMs have limited capabilities and can generate hallucinations if the prompt is too complex
> - In my personal experiments, I have noticed that the answers are more accurate and qualitatively better


## Config
The main functions takes a custom configuration class as input:

```python
model_config = ModelConfig(
    hf_model_id="./testmodel", 
    trust_remote_code=True, 
    use_flash_attn=True, 
    device="gpu", # or cpu, mps, ...
    chat_template="chatml", # can be llama and zephyr also!
    dtype = torch.bfloat16,
    quantization_config = BitsAndBytesConfig(...),
    trust_remote_code = True
    low_cpu_mem_usage = True
    max_new_token = 2048
)

loader_config = LoaderConfig(
    dataset="./test" # dataset path or huggingface Id
    is_hf_dataset = True, # LEAVE TRUE
    text_column = "text", # the text column you are going to use
    chunking_config = None, # NOT YET IMPLEMENTED
    trust_remote_code = True,
    split = "train" # the dataset split you are going to use
)


generation_config = GenerationConfig(
    loader_config = loader_config, # dataset loader configuration
    model_config = model_config, # model configuration
    num_rows = num_rows, # total number of rows of the dataset
    rows_per_chunk = rows_per_chunk, # number of questions generated for each "context"
    language = "it", # language of the prompts, you can create your custom prompts in "prompts.ini"
    agents = "3", # number of processes, it depends on your harware capabilities
)
```

## Example

```python
from src.datasetbuilder.classes.generation_config import GenerationConfig
from src.datasetbuilder.classes.loader_config import LoaderConfig
from src.datasetbuilder.classes.model_config import ModelConfig
from src.datasetbuilder.dataset_builder import DatasetBuilder
import multiprocessing

loader_config = LoaderConfig(dataset="./test")
model_config = ModelConfig(hf_model_id="./testmodel", trust_remote_code=True, use_flash_attn=False, device="cpu", chat_template="llama")
generation_config = GenerationConfig(
    loader_config=loader_config,
    model_config=model_config,
    num_rows=10000,
    rows_per_chunk=3,
    language="IT",
    agents=2
)



if __name__ == '__main__':
    multiprocessing.freeze_support()
    db = DatasetBuilder(generation_config=generation_config)
    db.build()  
```

