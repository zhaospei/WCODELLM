# python3 -m pipeline.extract_token_code --model codellama/CodeLlama-7b-Instruct-hf --dataset dev_eval --layers 16 20 24 28 32 --generate_dir /home/trang-n/WCODELLM_MULTILANGUAGE/output/codellama_CodeLlama-7b-Instruct-hf_dev_eval_python_1_4_8_12_16_20_24_28_32/temp2 --type last_line
# python3 -m pipeline.extract_token_code --model deepseek-ai/deepseek-coder-1.3b-instruct --dataset ds1000 --layers 0 1 2 11 12 13 22 23 24 --generate_dir /drive2/tuandung/WCODELLM/output/deepseek-ai_deepseek-coder-1.3b-instruct_ds1000_python_0_1_2_11_12_13_22_23_24/temp2
# python3 -m pipeline.extract_token_code --model Qwen/Qwen2.5-Coder-3B-Instruct --dataset mbpp --layers 0 1 2 17 18 19 34 35 36 --generate_dir /drive2/tuandung/WCODELLM/output/Qwen_Qwen2.5-Coder-3B-Instruct_mbpp_python_0_1_2_17_18_19_34_35_36/temp2
# python3 -m pipeline.extract_token_code --model Qwen/Qwen2.5-Coder-3B-Instruct --dataset human_eval --layers 1 12 18 24 28 32 36 --generate_dir /home/trang-n/WCODELLM_MULTILANGUAGE/output/Qwen_Qwen2.5-Coder-3B-Instruct_human_eval_python_1_12_18_24_28_32_36/temp2
# python3 -m pipeline.extract_token_code --model Qwen/Qwen2.5-Coder-3B-Instruct --dataset human_eval --layers 0 --generate_dir /drive2/tuandung/WCODELLM/output/Qwen_Qwen2.5-Coder-3B-Instruct_human_eval_python_0_1_2_17_18_19_34_35_36/temp2
# python3 -m pipeline.extract_token_code --model Qwen/Qwen2.5-Coder-3B-Instruct --dataset mbpp --layers 0 1 2 17 18 19 34 35 36 --generate_dir /root/WCODELLM/output/Qwen_Qwen2.5-Coder-3B-Instruct_human_eval_python_0_1_2_17_18_19_34_35_36/temp2
# python3 -m pipeline.extract_token_code --model codellama/CodeLlama-7b-Instruct-hf --dataset dev_eval --layers 1 4 8 12 16 20 24 28 32 --generate_dir /home/trang-n/WCODELLM_MULTILANGUAGE/output/codellama_CodeLlama-7b-Instruct-hf_dev_eval_python_1_4_8_12_16_20_24_28_32/temp2
python3 -m pipeline.extract_token_code --model deepseek-ai/deepseek-coder-1.3b-instruct --dataset mbpp --layers 22 24 --generate_dir /drive2/tuandung/WCODELLM/output/softmax_scores_deepseek-ai_deepseek-coder-1.3b-instruct_mbpp_python_22_24/temp2 --type min_prob_token_line