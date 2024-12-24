#deepseek-ai/deepseek-coder-6.7b-instruct
#Qwen/Qwen2.5-Coder-3B-Instruct
#codellama/CodeLlama-7b-Instruct-hf
#pip install hf_transfer
#pip install "huggingface_hub[hf_transfer]"
LANGUAGE=$1
MODEL='deepseek-ai/deepseek-coder-6.7b-instruct'
MODEL_NAME='deepseek-ai_deepseek-coder-6.7b-instruct'
# echo "MODEL: ${MODEL}"
echo "LANGUAGE: ${LANGUAGE}"

echo "running extract_token_code"
python3 -m pipeline.extract_token_code --model ${MODEL} --dataset human_eval --language ${LANGUAGE} --layers 1 4 8 12 16 20 24 28 32 --generate_dir  /data/thanhvt/QALLMCODE/multilingual/ds67${LANGUAGE}/temp2 