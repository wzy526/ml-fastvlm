import torch
import transformers
from transformers import AutoTokenizer
# 假设你的 LlavaForCausalLM 模型类在这里
from llava.model import LlavaLlamaForCausalLM 
from typing import Dict
import os

# --- 复制你提供的函数 ---
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    # ... (你上面提供的完整函数代码) ...
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
# --- 函数结束 ---


# --- 修复脚本的主体 ---
model_path = "/home/coder/work/llava-v1.5-7b" # 你的原始模型路径
fixed_model_path = "/home/coder/work/llava-v1.5-7b-fixed" # 修复后模型的保存路径

if not os.path.exists(fixed_model_path):
    os.makedirs(fixed_model_path)

print(f"Loading model from {model_path}...")
# !! 你必须用你自己的方式加载模型 !!
# !! 这是一个示例，请替换为你加载 LlavaForCausalLM 的代码 !!
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = LlavaLlamaForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16
)

print("Checking for pad_token error...")
if tokenizer.pad_token == tokenizer.unk_token or tokenizer.pad_token is None:
    print("WARNING: pad_token error detected. Fixing...")

    tokenizer.pad_token = None
    tokenizer.pad_token_id = None

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]"), 
        tokenizer=tokenizer, 
        model=model
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Fix applied. New pad_token: [PAD]")
else:
    print("Pad token seems OK. No fix needed.")

print(f"Saving fixed model and tokenizer to {fixed_model_path}...")
model.save_pretrained(fixed_model_path)
tokenizer.save_pretrained(fixed_model_path)
print("Done.")