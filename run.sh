# torchrun --nproc_per_node 2 -m pipeline.generate --model deepseek-ai/deepseek-coder-1.3b-base --dataset mbpp --num_generations_per_prompt 5 --fraction_of_data_to_use 0.01 --project_ind 0
# python -m pipeline.generate --model deepseek-ai/deepseek-coder-1.3b-base --dataset mbpp --num_generations_per_prompt 5 --fraction_of_data_to_use 0.01 --project_ind 0
# CUDA_VISIBLE_DEVICES=0,1 python3 -m pipeline.generate --model deepseek-ai/deepseek-coder-1.3b-base --dataset mbpp --num_generations_per_prompt 10 --fraction_of_data_to_use 1 --project_ind 0 --layer -1 -2 --max_new_tokens 256
# python3 -m pipeline.generate --model deepseek-ai/deepseek-coder-1.3b-base --dataset human_eval --num_generations_per_prompt 10 --fraction_of_data_to_use 1 --project_ind 0 --layer -1 --language python
# CUDA_VISIBLE_DEVICES=0,1 python3 -m pipeline.generate --model deepseek-ai/deepseek-coder-1.3b-base --dataset human_eval --num_generations_per_prompt 5 --fraction_of_data_to_use 1 --project_ind 0 --layer -1 --language java
# python3 -m pipeline.generate --model deepseek-ai/deepseek-coder-1.3b-base --dataset human_eval --num_generations_per_prompt 10 --fraction_of_data_to_use 1 --project_ind 0 --layer -1 --language cpp
# CUDA_VISIBLE_DEVICES=0,1 python3 -m pipeline.generate --model codellama/CodeLlama-7b-hf --dataset ds1000 --num_generations_per_prompt 5 --fraction_of_data_to_use 1 --project_ind 0 --layer -1 -2 --max_new_tokens 1024
# python3 -m pipeline.generate --model deepseek-ai/deepseek-coder-1.3b-base --dataset ds1000 --num_generations_per_prompt 5 --fraction_of_data_to_use 1 --project_ind 0 --layer -1 -2 --max_new_tokens 1024
# python3 -m pipeline.generate --model deepseek-ai/deepseek-coder-6.7b-base --dataset human_eval --num_generations_per_prompt 10 --fraction_of_data_to_use 1 --project_ind 0 --layer -1 -2 --max_new_tokens 256 --language rs 
# CUDA_VISIBLE_DEVICES=0,1 python3 -m pipeline.generate --model deepseek-ai/deepseek-coder-1.3b-base --dataset ds1000 --num_generations_per_prompt 5 --fraction_of_data_to_use 1 --project_ind 0 --layer -1 -2
# python3 -m pipeline.generate --model deepseek-ai/deepseek-coder-1.3b-base --dataset ds1000 --num_generations_per_prompt 5 --fraction_of_data_to_use 1 --project_ind 0 --layer -1 -2 --max_new_tokens 1000
# python3 -m pipeline.generate --model deepseek-ai/deepseek-coder-6.7b-base --dataset human_eval --num_generations_per_prompt 10 --fraction_of_data_to_use $FRACTION_OF_DATA --project_ind 0 --layer -1 -2 --max_new_tokens 256
# CUDA_VISIBLE_DEVICES=0,1 python3 -m pipeline.save_hidden_states --model deepseek-ai/deepseek-coder-6.7b-instruct --dataset human_eval --num_generations_per_prompt 10 --fraction_of_data_to_use 1 --project_ind 0 --layer 0 1 2 11 12 13 22 23 24 --max_new_tokens 1000 --language python
# 0 1 2 15 16 17 30 31 32
# python3 -m pipeline.save_hidden_states --model deepseek-ai/deepseek-coder-1.3b-instruct --dataset mbpp --num_generations_per_prompt 10 --fraction_of_data_to_use 1 --project_ind 0 --layer 0 1 2 11 12 13 22 23 24 --max_new_tokens 1000 --language python
python3 -m pipeline.save_hidden_states --model deepseek-ai/deepseek-coder-1.3b-instruct --dataset ds1000 --num_generations_per_prompt 2 --fraction_of_data_to_use 0.01 --project_ind 0 --layer 0 1 2 11 12 13 22 23 24 --max_new_tokens 1000 --language python
# python3 -m pipeline.save_hidden_states --model deepseek-ai/deepseek-coder-6.7b-instruct --dataset human_eval --num_generations_per_prompt 10 --fraction_of_data_to_use 1 --project_ind 0 --layer 0 1 2 11 12 13 22 23 24 --max_new_tokens 400
# 0 1 2 15 16 17 30 31 32
# python3 -m pipeline.save_hidden_states --model deepseek-ai/deepseek-coder-6.7b-instruct --dataset human_eval --num_generations_per_prompt 10 --fraction_of_data_to_use 1 --project_ind 0 --layer 23 24 --max_new_tokens 400
