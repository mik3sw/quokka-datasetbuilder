from datasets import load_dataset, Dataset
from src.datasetbuilder.classes.loader_config import LoaderConfig

def preprocess_documents(dataset) -> list:
    '''
    Questa funzione data una path si occuperà di trasformare tutti 
    i documenti in chunk e salverà un dataset in formato HF.
    Ritorna la path dove è presente quel dataset
    '''
    raise NotImplementedError
    return []
    


def load_data(loader_config: LoaderConfig) -> list:
    if not loader_config.is_hf_dataset:
        return preprocess_documents(loader_config.dataset)
    else:
        dataset = load_dataset(loader_config.dataset)
        dataset_split = dataset[loader_config.split]

    return Dataset.to_list(dataset_split)
    
    
