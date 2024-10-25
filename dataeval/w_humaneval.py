import os
import datasets
import pandas as pd
from datasets import Dataset

from utils.dataset import HumanEvalDataset

DATASET_ROOT="data/"
LANGUAGE="python"

def _save_dataset(sft=False):
    save_path = f"{DATASET_ROOT}/{LANGUAGE}" if not sft else f"{DATASET_ROOT}/{LANGUAGE}_sft"
    if not os.path.exists(save_path):
        data = HumanEvalDataset(root=DATASET_ROOT, language=LANGUAGE, issft=sft)
        dataset = {}
        dataset["prompt"] = []
        dataset["task_id"] = []
        dataset["original_prompt"] = []
        dataset["canonical_solution"] = []
        dataset["stopwords"] = []
        
        for sample in data:
            dataset["prompt"].append(sample["prompt"])
            dataset["task_id"].append(sample["task_id"])
            dataset["original_prompt"].append(sample["original_prompt"])
            dataset["canonical_solution"].append(sample["canonical_solution"])
            dataset["stopwords"].append(sample["stopwords"])
        
        data_df = pd.DataFrame.from_dict(dataset)
        dataset = Dataset.from_pandas(data_df)
        
        dataset.save_to_disk(save_path)
    return save_path

# _save_dataset(sft=False)

def get_dataset(tokenizer, sft=False):
    dataset = datasets.load_from_disk(_save_dataset(sft))

    def encode_humaneval(example):
        prompt = example['prompt']
        return tokenizer(prompt, truncation=False, padding=False)

    dataset = dataset.map(encode_humaneval, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset