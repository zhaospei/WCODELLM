#!/bin/bash

cd /home/trang-n/WCODELLM_MULTILANGUAGE

python3 -m pipeline.save_hidden_states --model deepseek-ai/deepseek-coder-1.3b-instruct --dataset ds1000 --num_generations_per_prompt 10 --fraction_of_data_to_use 1 --project_ind 0 --layer 0 1 2 11 12 13 22 23 24 --max_new_tokens 1000 --language python