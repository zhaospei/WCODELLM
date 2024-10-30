import pandas as pd
from tqdm import tqdm
import random

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from transformers import AutoTokenizer
import json
import os
from benchmark.MBPP.human_eval.evaluation import evaluate_functional_correctness_each_sample

data_root = "/drive2/tuandung/WCODELLM/benchmark/MBPP/data"
continue_from = '/drive2/tuandung/WCODELLM/0.pkl'
label_continue_from = '/drive2/tuandung/WCODELLM/0_label.pkl'
kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)
model_name = 'deepseek-ai/deepseek-coder-1.3b-base'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 

sequences = pd.read_pickle(continue_from)
# completion_id = '11_0'
label_sequences = pd.read_pickle(label_continue_from)
# print(label_sequences)
# print(type(label_sequences))
# passed = label_sequences[label_sequences['completion_id'] == completion_id]['label'].values[0]
# print(passed)
print(f'Loaded {len(sequences)} indices')
print(f'Loaded {len(label_sequences)} labels')
batch_size = 2

results = pd.DataFrame(columns=["task_id", "completion_id", "num_tokens", "last_token_middle_layer", "label"])

for idx in tqdm(range(0, len(sequences), batch_size)):
    # batch_lines = []
    for sequence in sequences[idx:idx + batch_size]:
        for j in range(len(sequence['generations_ids'])):
            task_id = sequence['id'].numpy().tolist()
            completion_id = str(task_id) + '_' + str(j)
            numtoken = sequence['num_tokens'][j]
            random_token = random.randint(0, numtoken - 1)
            random_token_middle_layer = sequence['middle_layer_embeddings'][j][random_token]
            last_token_middle_layer = sequence['middle_layer_embeddings'][j][numtoken-2]
            input_token_middle_layer = sequence['middle_layer_embeddings'][j][0]
            passed = label_sequences[label_sequences['completion_id'] == completion_id]['label'].values[0]
            results = results._append({
                "task_id": task_id, 
                "completion_id": completion_id, 
                "num_tokens": numtoken, 
                "input_token_middle_layer": input_token_middle_layer,
                "last_token_middle_layer": last_token_middle_layer,
                "random_token_middle_layer": random_token_middle_layer,
                "label": passed
            }, 
            ignore_index=True)

# results.to_pickle(os.path.join(data_root, 'mbpp_data_random_token_classify_dataset.pkl'))
results.to_parquet(os.path.join(data_root, 'mbpp_data_ramdom_first_last_classify_dataset.parquet'))
if __name__ == '__main__':
    print("Done!")