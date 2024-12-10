import pandas as pd

df = pd.read_parquet('/drive2/tuandung/WCODELLM/jaist/codellama/LFCLF_embedding_mbpp_codellama_CodeLlama-7b-Instruct-hf_1.parquet')

print(df.iloc[16]['extracted_code'])