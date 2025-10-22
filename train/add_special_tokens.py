"""
添加特殊 token 到 tokenizer
"""
import os
from transformers import AutoTokenizer

# 路径配置
MODEL_PATH = "/root/llada/train/models/LLaDA-V"
OUTPUT_PATH = "/root/llada/train/models/LLaDA-V-grounding"

# 加载 tokenizer
print(f"Loading tokenizer from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 定义要添加的特殊 token
special_tokens_dict = {
    'additional_special_tokens': ['<COUNT>', '</COUNT>', '<COORD>', '</COORD>']
}

# 添加特殊 token
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added_tokens} special tokens")
print(f"New vocab size: {len(tokenizer)}")

# 保存扩展后的 tokenizer
os.makedirs(OUTPUT_PATH, exist_ok=True)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"Tokenizer saved to {OUTPUT_PATH}")

# 验证
print("\nVerifying new tokens:")
for token in ['<COUNT>', '</COUNT>', '<COORD>', '</COORD>']:
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"{token}: {token_id}")