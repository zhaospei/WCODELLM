import os
import datasets
import pandas as pd
from datasets import Dataset
import torch

from benchmark.HumanEval.utils.dataset import HumanEvalDataset
from benchmark.MBPP.utils.dataset import MBPPDataset

from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

import _settings
DATASET_ROOT= os.path.join(_settings.DATA_FOLDER, "MBPP", "data")

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_str, tokenizer):
        StoppingCriteria.__init__(self)
        self.current_context = []
        self.tokenizer = tokenizer
        self.keywords_str = keywords_str
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.current_context.append(input_ids[0][-1].item())
        current_context = self.tokenizer.decode(self.current_context)
        for word in self.keywords_str:
            if word in current_context:
                return True
        return False

def _save_dataset(tokenizer, max_seq_len, max_gen_len):
    save_path = f"{DATASET_ROOT}/{tokenizer.name_or_path}_{max_seq_len}_{max_gen_len}"
    if not os.path.exists(save_path):
        mpbb_data = MBPPDataset(root=DATASET_ROOT)
        dataset = {}
        dataset["prompt"] = []
        dataset["task_id"] = []
        dataset["canonical_solution"] = []
        dataset["prompt_length"] = []
        dataset["original_prompt"] = []
        
        for j in range(len(mpbb_data)):
            data = mpbb_data[j]
            prompt = mpbb_data.prompt
            prompt1 = data["prompt"]
            tests = "\n".join(data["test"])
            # test_list.append(data["test"])
            prompt_curr = f"You are an expert Python programmer, and here is your task: {prompt1} Your code should pass these tests:\n\n{tests}\n[BEGIN]"
            fprompt = ""
            for i in range(len(prompt) - 1, -1, -1):
                finalprompt = prompt[i] + prompt_curr
                curr_seq_len = len(tokenizer.encode(finalprompt))
                if curr_seq_len >= max_seq_len - max_gen_len:
                    continue
                else:
                    fprompt = finalprompt
                    break
            if fprompt == "":
                fprompt = prompt_curr
                encodelist = tokenizer.encode(fprompt)
                while True:
                    try:
                        fprompt = tokenizer.decode(encodelist[:max_seq_len - max_gen_len])
                        break
                    except:
                        encodelist.pop(-1)
            dataset["prompt"].append(fprompt)
            dataset["canonical_solution"].append(data['code'])
            dataset["prompt_length"].append(len(fprompt))
            dataset["task_id"].append(data["task_id"])
            dataset["original_prompt"].append(data["prompt"])
            
        data_df = pd.DataFrame.from_dict(dataset)
        dataset = Dataset.from_pandas(data_df)
        
        dataset.save_to_disk(save_path)
    return save_path

# _save_dataset(sft=False)

def get_dataset(tokenizer, language='python', max_seq_len=4096, max_gen_len=500):
    dataset = datasets.load_from_disk(_save_dataset(tokenizer, max_seq_len, max_gen_len))
    
    def encode_mbpp(example):
        tokenized_prompt = tokenizer(example['prompt'], truncation=False, padding=False)
        inputids = tokenized_prompt["input_ids"][-max_seq_len:]
        attenion_mask = tokenized_prompt["attention_mask"][-max_seq_len:]
        return dict(input_ids=inputids, attention_mask=attenion_mask)
        # return tokenized_prompt

    dataset = dataset.map(encode_mbpp, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset

def _generate_config(tokenizer):
    stop_criteria = KeywordsStoppingCriteria(["[DONE]"], tokenizer)
    return dict(stopping_criteria=StoppingCriteriaList([stop_criteria]))

def _stop_word_list(language):
    return "[DONE]"

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