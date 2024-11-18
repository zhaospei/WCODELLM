from datasets import load_dataset
import pandas as pd
import datasets
from datasets import Dataset
import os
import _settings
import json
DATASET_ROOT= os.path.join(_settings.DATA_FOLDER, "RepoEval", "datasets")
STOP_WORDS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"]

def load_jsonl(fname):
    with open(fname, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines


def _save_dataset(language, sft=False):
    save_path = f"{DATASET_ROOT}/{language}" if not sft else f"{DATASET_ROOT}/{language}_sft"
    if not os.path.exists(save_path):
        data_path = os.path.join(DATASET_ROOT, "function_level_completion_2k_context_codex.test.jsonl")
        lines = load_jsonl(data_path)
        dataset = {}
        dataset["prompt"] = []
        dataset["task_id"] = []
        dataset["original_prompt"] = []
        dataset["canonical_solution"] = []
        dataset["stopwords"] = []
        
        for sample in lines:
            dataset["prompt"].append(sample["prompt"])
            dataset["task_id"].append(sample["metadata"]["task_id"])
            dataset["original_prompt"].append(sample["prompt"])
            dataset["canonical_solution"].append(sample["metadata"]["ground_truth"])
            dataset["stopwords"].append(STOP_WORDS)
        
        data_df = pd.DataFrame.from_dict(dataset)
        dataset = Dataset.from_pandas(data_df)
        
        dataset.save_to_disk(save_path)
    return save_path

# _save_dataset(sft=False)

def get_dataset(tokenizer, language, sft=False):
    dataset = datasets.load_from_disk(_save_dataset(language, sft))

    def encode_humaneval(example):
        prompt = example['prompt']
        return tokenizer(prompt, truncation=False, padding=False)

    dataset = dataset.map(encode_humaneval, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset

def postprocess_by_function(generation, target):
    first_token = target.split()[0]
    function_indent = target.split(first_token)[0]
    generation_lines = []
    for line in generation.split('\n'):
        if line.split() and line.split()[0]!='#':
            first_token = line.split()[0]
            indent = line.split(first_token)[0]
            if len(indent) < len(function_indent):
                break
            generation_lines.append(line)
    return generation_lines