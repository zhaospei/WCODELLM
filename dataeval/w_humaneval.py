import os
import datasets
import pandas as pd
from datasets import Dataset

from benchmark.HumanEval.utils.dataset import HumanEvalDataset

DATASET_ROOT="/drive2/tuandung/WCODELLM/benchmark/HumanEval/data"
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

def _generate_config(tokenizer):
    return dict()

def cleanup_code(
    code: str,
    language_type: str = None,
    dataset: str = None,
    issft: bool = False,
    stop_words = []
):
    """
    Cleans up the generated code.
    """

    if language_type.lower() == "python":
        if issft:
            code = _clean_python_code_for_sft(code)
        stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
        code = _truncate_code_at_stopwords(code, stop_words)
    elif language_type.lower() == "ts":
        code = _truncate_code_at_stopwords(code, stop_words + ["\nexport", "\nimport", "\nexport default", "\nimport default", "\nconsole.log"])
    else:
        code = _truncate_code_at_stopwords(code, stop_words)

    return code

def _clean_python_code_for_sft(code):
    code = code.replace("\r", "")
    if "```python" in code:
        code_start_idx = code.index("```python")
        code = code[code_start_idx:].replace("```python", "").strip()
        end_idx = code.find("```") if "```" in code else len(code)
        code = code[:end_idx].strip()

    return code

def _truncate_code_at_stopwords(code, stop_words):
    min_stop_idx = len(code)
    for stop_word in stop_words:
        stop_index = code.find(stop_word)
        if 0 <= stop_index < min_stop_idx:
            min_stop_idx = stop_index
    return code[:min_stop_idx]