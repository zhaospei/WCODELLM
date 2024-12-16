#done sh cs js java python php 
#error ts cpp 
#ndone  
for L in "cs" "java" "js" "php" "python" "sh" "ts" 
do 
LANG=$L
python -m datalabel.label_human_eval_instruct2 \
--data_root benchmark/HumanEval/data \
--file data/ds67$LANG/output2/LFCLF_embedding_human_eval_deepseek-ai_deepseek-coder-6.7b-instruct_1.parquet \
--model_name deepseek-ai/deepseek-coder-6.7b-instruct \
--lang $LANG
done