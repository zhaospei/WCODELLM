for L in "cpp" "cs" "java" "js" "php" "python" "sh" "ts" # "cross_task"
do 
LANG=$L
TYPE="output"
python3 pipeline/baseline_generation_2.py \
 --model deepseek-ai/deepseek-coder-6.7b-instruct \
 --source_file data/ds67$LANG/output2/LFCLF_embedding_human_eval_deepseek-ai_deepseek-coder-6.7b-instruct_1_label.parquet \
 --prompt_file data/data_for_classify/${TYPE}_${LANG}.json \
 --problem_file benchmark/HumanEval/data/humaneval-${LANG}.jsonl \
 --mode gen_prompt \
 --type $TYPE

done

