import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

import datasets
import dataeval.w_mbpp as mbpp
import dataeval.w_repoeval as repo_eval
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="deepseek-ai/deepseek-coder-1.3b-instruct",
    help="which results to run",
)
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="where to resume inference",
)
args = parser.parse_args()
model_name = args.model

model = AutoModelForCausalLM.from_pretrained(
    model_name, resume_download=True, trust_remote_code=True
)
model.to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Need to set the padding token to the eos token for generation
if tokenizer.eos_token:
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer.add_special_tokens({
        "pad_token": "<pad>"
    })

import random
random.seed(37)

# Generate unique numbers within the range
train_task_ids = random.sample(range(11, 510), 450)

df = pd.read_parquet("/drive2/tuandung/WCODELLM/mbpp_result/mbpp_deepseek_6.7b_last_token_-1_-2_with_label.parquet")

df_test_dict = df[~df['task_id'].isin(train_task_ids)].to_dict(orient='records')

def build_prompt(Query):
    return f"Are you capable of providing an accurate response to the query given below? Respond only to this question with ’yes’ or ’no’ and do not address the content of the query itself. The query start with 'Query:' and your respone after 'Answer'. \n\n Query: {Query}\n\n. Are you capable of providing an accurate response to the query given above without more information? Respond only to this question with yes or no. \nAnswer: "
# \n\nQuery: Write a function to find the similar elements from the given two tuple lists. \nAnswer: yes
repoeval_ds = repo_eval.get_dataset(tokenizer, "python", sft=False, instruction=True)

# Apply padding on the left since we are doing generation
padding_side_default = tokenizer.padding_side
tokenizer.padding_side = "left"

baseline_prompts = [build_prompt("\nComplete the body of the unfinished function: \n" + prompt) for prompt in repoeval_ds['prompt']]

# df_test = pd.DataFrame(df_test_dict)

# Tokenize each batch
# Put back the original padding behavior
tokenizer.padding_side = padding_side_default

model_name = model_name.replace('/', '-')
generation_config = {
    "do_sample": False,
    "max_new_tokens": 32,
    "num_beams": 1
}

generated_texts = []

# with tqdm(total=len(df_test)) as pbar:
for prompt in tqdm(baseline_prompts):
    # prompt = build_prompt(p)
    tokenized_prompt = tokenizer(prompt, padding=True,
                                    return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **tokenized_prompt,
            **generation_config
        )
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # pbar.update(1)
    print(generated_text)
    generated_texts.append(generated_text[0])

with open("abc.txt", "w") as f:
    for text in generated_texts:
        f.write(text + "\n")
