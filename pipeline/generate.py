import argparse
import glob
import json
import os
import copy
import time

import pandas as pd
import torch
import tqdm
import transformers
from sentence_transformers import SentenceTransformer
from torchmetrics.text.bert import BERTScore

import _settings
import dataeval.w_humaneval as human_eval
import dataeval.w_mbpp as mbpp
from dataeval.w_humaneval import cleanup_code as human_eval_cleanup_code
import models
import utils
from func.metric import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-13b-hf')
parser.add_argument('--dataset', type=str, default='human_eval')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=10)
parser.add_argument('--max_new_tokens', type=int, default=500)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=0.99)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument('--project_ind', type=int, default=0)
parser.add_argument("--layers", nargs='*', default=[-1], type=int,
                        help="List of layers of the LM to save embeddings from indexed negatively from the end")


args = parser.parse_args()
print(args.model.replace('/', '_'))
logInfo = open("./data/output/logInfo_{}_{}.txt".format(args.model.replace('/', '_'), args.dataset), mode="w",encoding="utf-8")


# _UNUSED_TOKENIZER = models.load_tokenizer()
def get_dataset_fn(data_name):
    if data_name == 'human_eval':
        return human_eval.get_dataset
    if data_name == 'mbpp':
        return mbpp.get_dataset

def get_clean_up_code_fn(data_name):
    if data_name == 'human_eval':
        return human_eval_cleanup_code

def get_num_tokens(generation, tokenizer):
    if args.dataset == 'human_eval':
        return humaneval_get_num_tokens(generation, tokenizer)
    if args.dataset == 'mbpp':
        return mbpp_get_num_tokens(generation, tokenizer)


def get_generation_config(tokenizer, data_name):
    if data_name == 'human_eval':
        generation_config = human_eval._generate_config(tokenizer)
    if data_name == 'mbpp':
        generation_config = mbpp._generate_config(tokenizer)
    return generation_config


@torch.no_grad()
def get_generations(model_name:str, args, seed=1, old_sequences=None, max_num_gen_once=args.num_generations_per_prompt):
    device = args.device
    model, tokenizer = models.load_model_and_tokenizer(model_name, args.device)
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Shape: {param.shape} | Parameters: {param.numel()}")
    # SenSimModel = SentenceTransformer('./data/weights/nli-roberta-large')
    # bertscore = BERTScore(model_name_or_path="./data/weights/bert-base/", device="cuda")

    utils.seed_everything(seed)
    dataset = get_dataset_fn(args.dataset)(tokenizer)
    cleanup_code = get_clean_up_code_fn(args.dataset)
    generation_config = get_generation_config(tokenizer, args.dataset)
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['task_id']: _ for _ in old_sequences}
    # print('Checking task id: ', dataset[0]['task_id'])
    # print(dataset[0])
    sequences = []
    time_start=time.time()
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # print(batch.keys())
        if batch['task_id'][0] in old_sequences:
            sequences.append(old_sequences[batch['task_id'][0]])
            continue
        # print(f"Batch {batch_idx} | Task ID: {batch['task_id']}")
        # print(batch)
        input_ids = batch['input_ids'].to(device)
        # print(f"input_ids: {input_ids}")
        # print(f"input_ids shape: {input_ids.shape}")
        # print(f"attention_mask: {batch['attention_mask']}")
        # print(f"attention_mask shape: {batch['attention_mask'].shape}")
        input_length = input_ids.shape[1]
        torch.cuda.empty_cache()
        generations = []
        num_gens = args.num_generations_per_prompt
        while num_gens > 0:
            dict_outputs =  model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                            num_beams=1, max_new_tokens=args.max_new_tokens, num_return_sequences=min(max_num_gen_once, num_gens),
                            do_sample=True, top_p=args.top_p, top_k=args.top_k,
                            temperature=args.temperature, 
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id,
                            # **generation_config,
                            output_hidden_states = True, return_dict_in_generate=True, output_scores=True
                            )

            generation = dict_outputs.sequences[:, input_length:].cpu()
            print(f"Generation shape: {generation.shape}")
            num_tokens = get_num_tokens(generation, tokenizer)
            for gen, num_token in zip(generation, num_tokens):
                generations.append(gen[:num_token])
            hidden_states = dict_outputs.hidden_states
            middle_layer_embeddings = getMiddleLayerEmbeddingEachToken(hidden_states, num_tokens)
            num_gens -= len(generation)
            

        # # remember the data
        # print(batch['task_id'][0])
        curr_seq = dict(
            prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
            id=batch['task_id'][0],
            problem=batch['original_prompt'][0],
        )
        curr_seq.update(
            dict(
                middle_layer_embeddings = middle_layer_embeddings,
                generations_ids=generations,
                num_tokens=num_tokens
            )
        )
        
        print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True))
        print("Problem:", batch['original_prompt'][0])
        print("AnswerGT:", batch['canonical_solution'][0])
        print("MostLikelyAns:", tokenizer.decode(curr_seq['generations_ids'][0], skip_special_tokens=True))

        sequences.append(curr_seq)
        torch.cuda.empty_cache()
    return sequences


def find_sublist(gen_tensor_ids, stop_word_ids):
    gen_tensor_ids = gen_tensor_ids.to('cuda')
    stop_word_tensor_ids = torch.tensor(stop_word_ids).to('cuda')
    len_gen = gen_tensor_ids.size(0)
    len_stop_word = stop_word_tensor_ids.size(0)
    if len_stop_word > len_gen:
        first_index = -1
    else:
        windows = gen_tensor_ids.unfold(0, len_stop_word, 1)
        matches = (windows == stop_word_tensor_ids).all(dim=1)
        first_index = torch.where(matches)[0][0].item() if matches.any() else -1
    
    return first_index

def mbpp_get_num_tokens(generation, tokenizer):
    stop_words = ["[DONE]"]
    tokenizer_stop_words = [tokenizer.encode(_)[1:] for _ in stop_words] + [[tokenizer.eos_token_id]]
    num_tokens = []
    for ids in generation:
        min_stop_idx = len(ids)
        for stop_word in tokenizer_stop_words:
            stop_index = find_sublist(ids, stop_word)
            if 0 <= stop_index < min_stop_idx:
                min_stop_idx = stop_index
        num_tokens.append(min_stop_idx)
    return num_tokens

def humaneval_get_num_tokens(generation, tokenizer):
    stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
    tokenizer_stop_words = [tokenizer.encode(_)[1:] for _ in stop_words] + [[tokenizer.eos_token_id]]
    num_tokens = []
    for ids in generation:
        min_stop_idx = len(ids)
        for stop_word in tokenizer_stop_words:
            stop_index = find_sublist(ids, stop_word)
            if 0 <= stop_index < min_stop_idx:
                min_stop_idx = stop_index
        num_tokens.append(min_stop_idx)
    return num_tokens


def main(overwrite=False, continue_from=None, parallel:int=None):
    if continue_from:
        fname = os.path.basename(continue_from)
        args.__dict__ = utils.jload(continue_from.replace(fname, 'args'+fname.replace("_partial.pkl", ".json")))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace("_partial.pkl", ""))
        model_name = args.model
    else:
        old_sequences = []
        model_name = args.model
        # if '/' in model_name:
        #     model_name = model_name.replace('/', '_')
        cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.project_ind}')
        os.makedirs(cache_dir, exist_ok=True)
        old_results = glob.glob(os.path.join(cache_dir, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        # if len(old_results) > 0 and not overwrite:
        #     print(f'Found {len(old_results)} generations in {cache_dir}.')
        #     return
        run_id = len(old_results)
        with open(os.path.join(cache_dir, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
    print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    print(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    sequences = get_generations(model_name, args, seed=args.seed, old_sequences=old_sequences)
    print(f'Writing {len(sequences)} generations to {cache_dir}...')
    pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}.pkl'))
    return

if __name__ == '__main__':
    task_runner = main(parallel=args.nprocess)
