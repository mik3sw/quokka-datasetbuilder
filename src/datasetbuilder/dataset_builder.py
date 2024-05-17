import json
import jsonlines
import regex
from src.datasetbuilder.classes.generation_config import GenerationConfig
from src.datasetbuilder.loader import load_data
from src.datasetbuilder.model import initialize_model_tokenizer, generate_answer, format_answer_prompt, format_json_prompt
import multiprocessing
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from concurrent.futures import ThreadPoolExecutor
import threading


import time
import random


console = Console()


def from_string_to_json(string: str):
    '''
    Given a JSON string this function will return a JSON obj
    '''
    try:
        json_obj = json.loads(string)
        return json_obj
    except json.JSONDecodeError as e:
        console.print("[bold red]ERROR[/bold red] - failed decoding generated JSON")
        return None

def extract_json_from_text(text: str) -> str:
    '''
    Given a text with a JSON obj in it, this function 
    will return only the json OBJ in string format
    '''
    pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
    return pattern.findall(text)[0]

def split_list(lst: list, num_splits: int):
    '''
    This function splits list in chunks
    '''
    num_elements = len(lst)
    elements_per_split = num_elements // num_splits
    remainder = num_elements % num_splits

    splits = []
    start_index = 0

    for i in range(num_splits):
        end_index = start_index + elements_per_split + (1 if i < remainder else 0)
        splits.append(lst[start_index:end_index])
        start_index = end_index

    return splits


class DatasetBuilder:
    def __init__(self, generation_config: GenerationConfig, output_dir: str="generated_dataset.jsonl"):
        self.generation_config = generation_config
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("[Elapsed time: "),
            TimeElapsedColumn(),
            TextColumn("] [Remaining time: "),
            TimeRemainingColumn(),
            TextColumn("]"),
            #"[green]Generating dataset... [progress.percentage]{task.percentage:>3.1f}% ({task.completed}/{task.total})",
              auto_refresh=True,
              console=console)
        self.output_dir = output_dir

    
    def build(self, ):
        '''
        This function will start the generation process
        '''
        msg = """
  /$$$$$$                      /$$       /$$                
 /$$__  $$                    | $$      | $$                
| $$  \ $$ /$$   /$$  /$$$$$$ | $$   /$$| $$   /$$  /$$$$$$ 
| $$  | $$| $$  | $$ /$$__  $$| $$  /$$/| $$  /$$/ |____  $$
| $$  | $$| $$  | $$| $$  \ $$| $$$$$$/ | $$$$$$/   /$$$$$$$
| $$/$$ $$| $$  | $$| $$  | $$| $$_  $$ | $$_  $$  /$$__  $$
|  $$$$$$/|  $$$$$$/|  $$$$$$/| $$ \  $$| $$ \  $$|  $$$$$$$
 \____ $$$ \______/  \______/ |__/  \__/|__/  \__/ \_______/
      \__/                                                             
"""
        console.print(msg, style="bold green")
        console.log(f"Loading dataset {self.generation_config.loader_config.dataset}")
        dataset = load_data(self.generation_config.loader_config)
        self._buld_multiagent(dataset)
        console.print(f"Dataset succesfully created and saved in [bold purple]{self.output_dir}[/bold purple]", style="bold green")

        
    
    def _buld_multiagent(self, dataset: list):
        '''
        This function will start multiple generation 
        processes splitting the dataset in N chunks.
        '''
        num_processes = self.generation_config.agents
        subsets = split_list(dataset, num_processes)

        counter = multiprocessing.Value('i', 0)
        
        with self.progress:
            task = self.progress.add_task("[green]Generating dataset...", total=self.generation_config.num_rows)

            with ThreadPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                print(f"Lunghezza Subsets: {len(subsets)}")
                for subset in subsets:
                    time.sleep(10)
                    print("Starting Thread")
                    future = executor.submit(self._generate_qa, subset, counter, task)
                    futures.append(future)

                for future in futures:
                    future.result()  # Attendere che tutte le operazioni siano completate

    def _generate_qa(self, dataset: list, counter, task):
        '''
        Given a dataset (list) this function will 
        iterate each row and create a new set of QA
        '''
        console.log(f"Loading Model [bold cyan]{self.generation_config.model_config.hf_model_id}[/bold cyan] and Tokenizer [bold cyan]{self.generation_config.model_config.hf_tokenizer_id}[/bold cyan]")
        model, tokenizer = initialize_model_tokenizer(self.generation_config.model_config, console=console)

        for item in dataset:
            if counter.value >= self.generation_config.num_rows:
                break
            context = item[self.generation_config.loader_config.text_column]
            try:
                questions_obj = self._get_json_from_context(context, model, tokenizer)
                for value in questions_obj.values():
                    if counter.value >= self.generation_config.num_rows:
                        break
                    counter.value += 1 
                    value = value.strip()
                    if self._checkquestion(value):
                        answer = self._get_answer_from_context(context, value, model, tokenizer).strip()
                        dataset_obj = {
                            "question": value,
                            "context": context,
                            "answer": answer
                        }
                        self._save_to_dataset(dataset_obj)
                        self.progress.update(task, advance=1) 
                        console.log(f"Thread {threading.current_thread().ident} - row generated")
            except:
                console.log("An error occurred while generating")  

    def _get_json_from_context(self, context, model, tokenizer):
        '''
        This function given a context will generate a set of new questions
        '''
        formatted_prompt = format_json_prompt(context, self.generation_config.model_config.chat_template, self.generation_config.language, self.generation_config.rows_per_chunk)
        answer = generate_answer(formatted_prompt, tokenizer, model, self.generation_config.model_config.chat_template, max_new_tokens=self.generation_config.model_config.max_new_token, device=self.generation_config.model_config.device)
        return from_string_to_json(extract_json_from_text(answer))
    

    def _get_answer_from_context(self, context, question, model, tokenizer) -> str:
        '''
        This function given a context and a question will generate the answer
        '''
        formatted_prompt = format_answer_prompt(context, question, self.generation_config.model_config.chat_template, self.generation_config.language)
        return generate_answer(formatted_prompt, tokenizer, model, self.generation_config.model_config.chat_template, max_new_tokens=self.generation_config.model_config.max_new_token, device=self.generation_config.model_config.device) 

    def _checkquestion(self, text: str) -> bool:
        '''
        This function will check if the 
        generated question is really a question
        '''
        if "?" not in text:
            return False
        else:
            return True
        
    def _save_to_dataset(self, obj):
        '''
        Save a new json line to the dataset
        '''
        with jsonlines.open(self.output_dir, mode="a") as writer:
            writer.write(obj)
