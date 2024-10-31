torchrun --nproc_per_node 2 -m pipeline.generate --model deepseek-ai/deepseek-coder-1.3b-base --dataset mbpp --num_generations_per_prompt 5 --fraction_of_data_to_use 0.01 --project_ind 0
# python -m pipeline.generate --model deepseek-ai/deepseek-coder-1.3b-base --dataset mbpp --num_generations_per_prompt 5 --fraction_of_data_to_use 0.01 --project_ind 0
