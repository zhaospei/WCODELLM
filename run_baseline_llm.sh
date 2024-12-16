export PYTHONPATH=`pwd`
pip install -r req.txt
pip install hf_transfer
pip install "huggingface_hub[hf_transfer]"
# MODEL=$1
LANGUAGE=$1
MODEL='deepseek-ai/deepseek-coder-6.7b-instruct'
MODEL_NAME='deepseek-ai_deepseek-coder-6.7b-instruct'
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download $MODEL



for L in "cpp" "cs" "java" "js" "php" "python" "sh" "ts" # "cross_task"
do 
LANG=$L
TYPE="output"
python3 pipeline/baseline_generation_2.py \
 --model deepseek-ai/deepseek-coder-6.7b-instruct \
 --source_file data/ds67$LANG/output2/LFCLF_embedding_human_eval_deepseek-ai_deepseek-coder-6.7b-instruct_1_label.parquet \
 --prompt_file data/data_for_classify/${TYPE}_${LANG}.json \
 --problem_file benchmark/HumanEval/data/humaneval-${LANG}.jsonl \
 --mode evaluate \
 --type $TYPE

done

