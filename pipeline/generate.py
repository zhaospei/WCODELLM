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
import dataeval.coqa as coqa
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
import dataeval.SQuAD as SQuAD
import dataeval.w_humaneval as human_eval
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


args = parser.parse_args()
print(args.model.replace('/', '_'))
logInfo = open("./data/output/logInfo_{}_{}.txt".format(args.model.replace('/', '_'), args.dataset), mode="w",encoding="utf-8")


# _UNUSED_TOKENIZER = models.load_tokenizer()
def get_dataset_fn(data_name):
    if data_name == 'human_eval':
        return human_eval.get_dataset

def get_clean_up_code_fn(data_name):
    if data_name == 'human_eval':
        return human_eval_cleanup_code


# def get_generation_config(input_ids, tokenizer, data_name):
#     assert len(input_ids.shape) == 2
#     max_length_of_generated_sequence = 256
#     if data_name == 'triviaqa':
#         generation_config = triviaqa._generate_config(tokenizer)
#     if data_name == 'coqa':
#         generation_config = coqa._generate_config(tokenizer)
#     if data_name == 'nq_open':
#         generation_config = nq_open._generate_config(tokenizer)
#     if data_name == 'SQuAD':
#         generation_config = SQuAD._generate_config(tokenizer)
#     generation_config['max_new_tokens'] = max_length_of_generated_sequence
#     generation_config['early_stopping'] = True
#     # https://jaketae.github.io/study/gpt2/#setup
#     generation_config['pad_token_id'] = tokenizer.eos_token_id
#     return generation_config


@torch.no_grad()
def get_generations(model_name:str, args, seed=1, old_sequences=None, max_num_gen_once=args.num_generations_per_prompt):
    device = args.device
    model, tokenizer = models.load_model_and_tokenizer(model_name, args.device)
    # SenSimModel = SentenceTransformer('./data/weights/nli-roberta-large')
    # bertscore = BERTScore(model_name_or_path="./data/weights/bert-base/", device="cuda")

    utils.seed_everything(seed)
    dataset = get_dataset_fn(args.dataset)(tokenizer)
    cleanup_code = get_clean_up_code_fn(args.dataset)
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['task_id']: _ for _ in old_sequences}

    sequences = []
    time_start=time.time()
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # print(batch.keys())
        if batch['task_id'][0] in old_sequences:
            sequences.append(old_sequences[batch['task_id'][0]])
            continue

        input_ids = batch['input_ids'].to(device)
        input_length = input_ids.shape[1]
        # generation_config = get_generation_config(input_ids, tokenizer, args.dataset)
        # generation_config = transformers.GenerationConfig(**generation_config)
        if args.decoding_method == 'beam_search':
            raise NotImplementedError()
        elif args.decoding_method == 'greedy':
            # generate the code
            # if args.temperature != 0:       
            #     dict_outputs = model.generate(
            #         input_ids=input_ids,
            #         max_new_tokens=self.max_gen_len,
            #         do_sample=True,
            #         eos_token_id=self.tokenizer.eos_token_id,
            #         temperature=self.temperature,
            #         top_p=self.top_p,
            #         pad_token_id=self.tokenizer.eos_token_id,
            #     )
            # else:
            #     dict_outputs = model.generate(
            #         input_ids=input_ids,
            #         max_new_tokens=self.max_gen_len,
            #         do_sample=False,
            #         eos_token_id=self.tokenizer.eos_token_id,
            #         pad_token_id=self.tokenizer.eos_token_id,
            #     )
            dict_outputs = model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                        num_beams=1,
                                        do_sample=False,
                                        max_new_tokens=args.max_new_tokens,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.eos_token_id,
                                        output_hidden_states = True,
                                        return_dict_in_generate=True,
                                        output_scores=True)

            scores = dict_outputs.scores    #([logits],[logits],[logits])
            perplexity = get_perplexity_score(scores)
            energy_score = get_energy_score(scores)
            most_likely_generations = dict_outputs.sequences.cpu()[0, input_length:]

        torch.cuda.empty_cache()
        generations = []
        num_gens = args.num_generations_per_prompt
        while num_gens > 0:
            print("num_gens: ", num_gens)
            dict_outputs =  model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                            num_beams=1, max_new_tokens=args.max_new_tokens, num_return_sequences=min(max_num_gen_once, num_gens),
                            do_sample=True, top_p=args.top_p, top_k=args.top_k,
                            temperature=args.temperature, 
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id,
                            output_hidden_states = True, return_dict_in_generate=True, output_scores=True
                            )

            generation = dict_outputs.sequences[:, input_length:].cpu()
            generations.append(generation)
            num_tokens = get_num_tokens(generation)
            # scores = dict_outputs.scores
            # predictive_entropy = get_lenghthNormalized_entropy(scores, num_tokens) 
            hidden_states = dict_outputs.hidden_states
            # eigenIndicator, eigenValue = getEigenIndicator_v0(hidden_states, num_tokens)
            # print(len(dict_outputs['hidden_states']))
            # print(len(dict_outputs['hidden_states'][0]))
            # print(len(dict_outputs['hidden_states'][0][0]))
            num_gens -= len(generation)

        # generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
        # generations = generations.reshape(-1, generations.shape[-1])[:args.num_generations_per_prompt]
        # best_generated_text = tokenizer.decode(most_likely_generations, skip_special_tokens=True)
        # generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generations]
        # lexical_similarity = getLexicalSim(generated_texts)
        # sent_bertscore = getAvgBertScore(bertscore, best_generated_text, generated_texts)
        # eigenIndicatorOutput, eigenValue_O = getEigenIndicatorOutput(generated_texts, SenSimModel)


        # # remember the data
        curr_seq = dict(
            prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
            id=batch['task_id'][0],
            problem=batch['original_prompt'][0],
            answer=batch['canonical_solution'][0],
            additional_answers=[],
        )
        curr_seq.update(
            dict(
                most_likely_generation_ids = most_likely_generations,
                generations_ids=generations,
                hidden_states=hidden_states,
                num_tokens=num_tokens
            )
        )
        # curr_seq.update(
        #     dict(
        #         most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
        #         generations=generated_texts,
        #     )
        # )
        # curr_seq.update(
        #     dict(
        #         perplexity=perplexity
        #     )
        # )
        # curr_seq.update(
        #     dict(
        #         energy=energy_score
        #     )
        # )
        # curr_seq.update(
        #     dict(
        #         lexical_similarity=lexical_similarity
        #     )
        # )
        # curr_seq.update(
        #     dict(
        #         sent_bertscore=sent_bertscore
        #     )
        # )
        # curr_seq.update(
        #     dict(
        #         entropy=predictive_entropy
        #     )
        # )
        # curr_seq.update(
        #     dict(
        #         eigenIndicator=eigenIndicator
        #     )
        # )
        # curr_seq.update(
        #     dict(
        #         eigenIndicatorOutput=eigenIndicatorOutput
        #     )
        # )
        # if args.dataset == 'coqa' or args.dataset == "TruthfulQA":
        #     curr_seq['additional_answers'] = [x[0] for x in batch['additional_answers']]

        sequences.append(curr_seq)
        torch.cuda.empty_cache()
        ########## 信息打印 #########
        # print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True))
        # print("Problem:", batch['original_prompt'][0])
        # print("Answer:", batch['canonical_solution'][0])
        # print("MostLikelyAns:", cleanup_code(tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True)))
        # print("Batch_Generations:", generated_texts)
        # print("Perplexity:", perplexity)
        # print("Energy:", energy_score)
        # print("NormalizedEntropy: ", predictive_entropy)
        # print("LexicalSimilarity: ", lexical_similarity)
        # print("EigenScore: ", eigenIndicator)
        # print("EigenValue:", eigenValue)
        # print("EigenScore-Output: ", eigenIndicatorOutput)

        print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True), file=logInfo)
        print("Problem:", batch['original_prompt'][0], file=logInfo)
        print("Answer:", batch['canonical_solution'][0], file=logInfo)
        print("BestAns:", tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True), file=logInfo)
        print("\n","\n","\n", file=logInfo)
    return sequences


def get_num_tokens(generation):  # generation: num_seq x max(num_tokens)
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2:
                count+=1
        num_tokens.append(count+1)
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
        if len(old_results) > 0 and not overwrite:
            print(f'Found {len(old_results)} generations in {cache_dir}.')
            return
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
