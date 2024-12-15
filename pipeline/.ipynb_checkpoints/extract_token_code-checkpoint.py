import os
import pickle
import argparse
import tqdm
import torch
import pandas as pd

import models
import dataeval.w_humaneval as human_eval
import dataeval.w_mbpp as mbpp
import dataeval.w_ds1000 as ds1000
import dataeval.w_repoeval as repo_eval
import dataeval.w_evocodebench as evocodebench
import dataeval.w_repoexec as repo_exec
from dataeval.w_humaneval import extract_generation_code as human_eval_egc
from dataeval.w_mbpp import extract_generation_code as mbpp_eval_egc
from dataeval.w_ds1000 import extract_generation_code as ds1000_eval_egc
from dataeval.w_evocodebench import extract_generation_code as evocodebench_eval_egc
from dataeval.w_repoeval import extract_generation_code as repoeval_eval_egc

from func.metric import *


parser = argparse.ArgumentParser()
parser.add_argument('--generate_dir', type=str, default='generate')
# parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--dataset', type=str, default='human_eval')
parser.add_argument('--model_name', type=str, default='opt-13b')
parser.add_argument("--language", default="python", type=str,)
parser.add_argument("--layers", default=[-1], type=int, nargs='+')
args = parser.parse_args()

def get_dataset_fn(data_name):
    if data_name == 'human_eval':
        return human_eval.get_dataset
    if data_name == 'mbpp':
        return mbpp.get_dataset
    if data_name == 'ds1000':
        return ds1000.get_dataset
    if data_name == 'repo_eval':
        return repo_eval.get_dataset
    if data_name == 'evocodebench':
        return evocodebench.get_dataset
    if data_name == 'repoexec':
        return repo_exec.get_dataset
    raise ValueError(f"Unknown dataset {data_name}")

def extract_generation_code_fun(data_name):
    if data_name == 'human_eval':
        return human_eval_egc
    if data_name == 'mbpp':
        return mbpp_eval_egc
    if data_name == 'ds1000':
        return ds1000_eval_egc
    if data_name == 'repo_eval':
        return repoeval_eval_egc
    if data_name == 'evocodebench':
        return evocodebench_eval_egc

def main():
    tokenizer = models.load_tokenizer(args.model_name)
    if 'chat' or 'instruct' in args.model_name.lower():
        instruction = True
    else:
        instruction = False
    dataset = get_dataset_fn(args.dataset)(tokenizer, language=args.language, instruction=instruction)
    dataset_egc = extract_generation_code_fun(args.dataset)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    output_dir = args.generate_dir.replace('temp', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for layer in args.layers:
        results = pd.DataFrame(columns=[
            "task_id", 
            "completion_id", 
            "num_tokens", 
            "generation", 
            "first_token_embedding", 
            "last_token_embedding",
            "first_token_code_embedding",
            "last_token_code_embedding",
            "has_error"
        ])
        for example in tqdm.tqdm(dataset, total=len(dataset)):
            has_error = False
            task_id_path =  str(example['task_id']).replace('/','_').replace('[','_').replace(']','_')
            if args.dataset == 'mbpp' or args.dataset == 'ds1000':
                task_id_path = f'tensor({task_id_path})'
            # if task_id_path != 'HumanEval_140_fix_spaces':
                # continue
            task_generation_seqs_path = f'generation_sequences_output_{task_id_path}.pkl'
            task_generation_seqs_path = os.path.join(args.generate_dir, task_generation_seqs_path)
            print(task_generation_seqs_path)
            if not os.path.exists(task_generation_seqs_path):
                print(f'File {task_id_path} not found. Skipping...')
                continue
            
            print(f'Found {task_id_path}. Processing...')
            
            with open(task_generation_seqs_path, 'rb') as f:
                task_generation_seqs = pickle.load(f)
            
            clean_generations_range = []
            for generated_ids in task_generation_seqs['generations_ids']:
                gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
                clean_generation_decoded = dataset_egc(example, gen, args.language)
                start_ind, end_ind = getCleanGenerationRange(generated_ids.tolist(), clean_generation_decoded, tokenizer)
                if start_ind is None or end_ind is None:
                    has_error = True
                    # print("gen:", gen)
                    # print("clean_generation_decoded:", clean_generation_decoded)
                    print(f'Cannot find clean generation range for {task_id_path}')
                    clean_generations_range.append(getGenerationRange(generated_ids.tolist(), tokenizer))
                else:
                    clean_generations_range.append((start_ind, end_ind))
            
            task_embedding_path = f'all_token_embedding_{task_id_path}_{layer}.pkl'
            task_embedding_path = os.path.join(args.generate_dir, task_embedding_path)
            if not os.path.exists(task_embedding_path):
                print(f'File {task_id_path} {layer} not found. Skipping...')
                continue
            
            with open(task_embedding_path, 'rb') as f:
                task_embedding = pickle.load(f)
            
            task_last_token_embedding = []
            for j in range(len(task_generation_seqs['generations'])):
                task_id = example['task_id']
                completion_id = str(task_id) + '_' + str(j)
                # num_tokens = task_generation_seqs['num_tokens'][j]
                generation = task_generation_seqs["generations"][j]
                generated_ids = task_generation_seqs["generations_ids"][j]
                start_code_ind, end_code_ind = clean_generations_range[j]
                start_ind, end_ind = getGenerationRange(generated_ids.tolist(), tokenizer)
                num_tokens = end_ind - start_ind
                layer_embedding = task_embedding['layer_embeddings'][j]
                # print(f'{start_ind} {end_ind} {start_code_ind} {end_code_ind}')
                try:
                    first_token_embedding = layer_embedding[start_ind].tolist()
                    last_token_embedding = layer_embedding[end_ind - 1].tolist()
                    first_token_code_embedding = layer_embedding[start_code_ind].tolist()
                    last_token_code_embedding = layer_embedding[end_code_ind - 1].tolist()
                    extracted_code = tokenizer.decode(generated_ids.tolist()[start_code_ind:end_code_ind], skip_special_tokens=True)
                    results = results._append({
                        "task_id": task_id, 
                        "completion_id": completion_id,
                        "num_tokens": num_tokens,
                        "generation": generation, 
                        "first_token_embedding": first_token_embedding, 
                        "last_token_embedding": last_token_embedding,
                        "first_token_code_embedding": first_token_code_embedding,
                        "last_token_code_embedding": last_token_code_embedding,
                        "has_error": has_error,
                        "extracted_code": extracted_code,
                    }, 
                    ignore_index=True)
                    # print(repr(extracted_code))
                except:
                    print(f'Error in {task_id} {completion_id}')
                    continue
        model_name = args.model_name.replace('/', '_')
        results.to_parquet(os.path.join(output_dir, f'LFCLF_embedding_{args.dataset}_{model_name}_{layer}.parquet'))
            
    return

if __name__ == '__main__':
    task_runner = main()
