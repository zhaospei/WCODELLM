from datasets import load_dataset
import pandas as pd
import datasets
from datasets import Dataset
import os
import _settings
import json

DATASET_ROOT= os.path.join(_settings.DATA_FOLDER, "EvoCodeBench", "prompt")
STOP_WORDS = []
TEMPLATE = """\
Please complete the {function_name} function based on the contexts above the function.

The contexts above the function are:
```Python
{contexts_above}
```

The code to be completed is:
```Python
{input_code}
```

Completed code:
"""

def load_jsonl(fname):
    with open(fname, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines

def _save_dataset(tokenizer,  max_seq_len, max_gen_len , instruction=False):
    save_path = f"{DATASET_ROOT}/{tokenizer.name_or_path}_{max_seq_len}_{max_gen_len}_{instruction}"
    if not os.path.exists(save_path):
        data_path = os.path.join(DATASET_ROOT, "prompt_elements.jsonl")
        lines = load_jsonl(data_path)

        dataset = {}
        dataset["prompt"] = []
        dataset["task_id"] = []
        dataset["original_prompt"] = []
        dataset["canonical_solution"] = []
        dataset["stopwords"] = []
        if instruction:
            for idx, sample in enumerate(lines):
                input_ids = tokenizer.encode(sample['input_code'])
                max_context_length = max_seq_len - len(input_ids) - max_gen_len
                context_above_ids = tokenizer.encode(sample['contexts_above'])
                context_above = ''
                if len(context_above_ids) > max_context_length:
                    context_above_ids = context_above_ids[-max_context_length:]
                    context_above = tokenizer.decode(context_above_ids)
                prompt = TEMPLATE.format(
                    function_name=sample['function_name'],
                    contexts_above=context_above,
                    input_code=sample['input_code']
                )
                dataset["prompt"].append(prompt)
                dataset["task_id"].append(sample["namespace"])
                dataset["original_prompt"].append(sample["input_code"])
                dataset["canonical_solution"].append("<NO_SOLUTION>")
                dataset["stopwords"].append(STOP_WORDS)
        else:
            for idx, sample in enumerate(lines):
                input_ids = tokenizer.encode(sample['input_code'])
                max_context_length = max_seq_len - len(input_ids) - max_gen_len
                context_above_ids = tokenizer.encode(sample['contexts_above'])
                context_above = ''
                if len(context_above_ids) > max_context_length:
                    context_above_ids = context_above_ids[-max_context_length:]
                    context_above = tokenizer.decode(context_above_ids)
                prompt = context_above + "\n" + sample['input_code']
                dataset["prompt"].append(prompt)
                dataset["task_id"].append(sample["namespace"])
                dataset["original_prompt"].append(sample["input_code"])
                dataset["canonical_solution"].append("<NO_SOLUTION>")
                dataset["stopwords"].append(STOP_WORDS)
            
        data_df = pd.DataFrame.from_dict(dataset)
        dataset = Dataset.from_pandas(data_df)
        
        dataset.save_to_disk(save_path)
    return save_path

# _save_dataset(sft=False)
def get_dataset(tokenizer, language='python', sft=False, instruction=False, max_seq_len=4096, max_gen_len=500):
    dataset = datasets.load_from_disk(_save_dataset(tokenizer, max_seq_len, max_gen_len, instruction))

    def encode_humaneval(example):
        prompt = example['prompt']
        if instruction:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": f'{prompt}'}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, truncation=False, padding=False)
        return inputs

    dataset = dataset.map(encode_humaneval, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset