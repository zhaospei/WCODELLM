import argparse
import dataeval.w_humaneval as human_eval
import dataeval.w_mbpp as mbpp
import dataeval.w_ds1000 as ds1000
import dataeval.w_repoeval as repo_eval
import models

parser = argparse.ArgumentParser()
parser.add_argument('--generate_dir', type=str, default='generate')
parser.add_argument('--dataset', type=str, default='human_eval')
parser.add_argument('--model_name', type=str, default='opt-13b')
parser.add_argument("--language", default="python", type=str,)
parser.add_argument("--layers", default=-1, type=int,)
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

def main():
    tokenizer = models.load_model_and_tokenizer(args.model_name)
    if 'chat' or 'instruct' in args.model_name.lower():
        instruction = True
    else:
        instruction = False
    dataset = get_dataset_fn(args.dataset)(tokenizer, language=args.language, instruction=instruction)
    
    for task_id in dataset['task_id']:
        
       
    return

if __name__ == '__main__':
    task_runner = main()