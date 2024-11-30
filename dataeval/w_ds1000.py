from datasets import load_dataset
import pandas as pd
import datasets
from datasets import Dataset
import os
import _settings
DATASET_ROOT= os.path.join(_settings.DATA_FOLDER, "ds1000", "data")

def _save_dataset(language, sft=False):
    save_path = f"{DATASET_ROOT}/{language}" if not sft else f"{DATASET_ROOT}/{language}_sft"
    if not os.path.exists(save_path):
        ds1000 = list(load_dataset("xlangai/DS-1000")["test"])
        prompts = [p["prompt"] for p in ds1000]
        # prompts = [ds1000[-1]["prompt"]]

        # specifically for id 156, too long, > 2048 tokens
        prompts[156] = "write a hello world in python"
        # specifically for id 156, too long, > 2048 tokens
        dataset = {}
        dataset["prompt"] = []
        dataset["task_id"] = []
        dataset["original_prompt"] = []
        dataset["canonical_solution"] = []
        dataset["stopwords"] = []
        
        for id, prompt in enumerate(prompts):
            dataset["prompt"].append(prompt)
            dataset["task_id"].append(id)
            dataset["original_prompt"].append(prompt)
            dataset["canonical_solution"].append("<NO_SOLUTION>")
            dataset["stopwords"].append(["</code>", "# SOLUTION END"])
        
        data_df = pd.DataFrame.from_dict(dataset)
        dataset = Dataset.from_pandas(data_df)
        
        dataset.save_to_disk(save_path)
    return save_path

# _save_dataset(sft=False)

def get_dataset(tokenizer, language, sft=False, instruction=False):
    dataset = datasets.load_from_disk(_save_dataset(language, sft, instruction))

    def encode_humaneval(example):
        prompt = example['prompt']
        if instruction:
            prompt = "Write a short code following the given format and indentation. Place the executable code between <code> and </code> tags, without any other non-executable things.\n" + prompt
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": f'{prompt}'}
                ], tokenize=False, add_generation_prompt=True)
        
        return tokenizer(prompt, truncation=False, padding=False)

    dataset = dataset.map(encode_humaneval, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset