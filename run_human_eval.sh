#deepseek-ai/deepseek-coder-6.7b-instruct
#Qwen/Qwen2.5-Coder-3B-Instruct
#codellama/CodeLlama-7b-Instruct-hf
#pip install hf_transfer
#pip install "huggingface_hub[hf_transfer]"
# MODEL=$1
LANGUAGE=$1
MODEL='deepseek-ai/deepseek-coder-6.7b-instruct'
MODEL_NAME='deepseek-ai_deepseek-coder-6.7b-instruct'
#HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download $MODEL

# echo "MODEL: ${MODEL}"
#echo "LANGUAGE: ${LANGUAGE}"

#echo "running save_hidden_states"
#CUDA_VISIBLE_DEVICES=0,1 python3 -m pipeline.save_hidden_states --model ${MODEL} --dataset human_eval --num_generations_per_prompt 10 --fraction_of_data_to_use 1 --project_ind 0 --layer 1 16 20 24 28 32 --max_new_tokens 1000 --language ${LANGUAGE}
for L in "cpp" "cs" "java" "js" "php" "python" "sh" "ts" 
do
#echo "running extract_token_code"
echo "mv output/${MODEL_NAME}_human_eval_${L}_1_4_8_12_16_20_24_28_32/temp2 output/ds67${L}"
#python3 -m pipeline.extract_token_code --model ${MODEL} --dataset human_eval --language ${L} --layers 1 4 8 12 16 20 24 28 32 --generate_dir  output/${MODEL_NAME}_human_eval_${L}_1_4_8_12_16_20_24_28_32/temp2 
echo "mv output/ds67${L} output/${MODEL_NAME}_human_eval_${L}_1_4_8_12_16_20_24_28_32/temp2"
done
