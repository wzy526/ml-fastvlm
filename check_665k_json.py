import json
# from llava.train.train_dat import preprocess_v1
# import llava.conversation as conversation_lib
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("/home/coder/work/llava-v1.5-7b", model_max_length=4096)
# conversation_lib.default_conversation = conversation_lib.conv_vicuna_v1
with open("/home/coder/work/llava-665k/llava_v1_5_mix665k.json", "r") as f:
    data_anno = json.load(f)

indexes = [245723, 247644]
gpt_val = ['beta-carotene', 'type a']

for idx,index in enumerate(indexes):
    old_val = data_anno[index]['conversations'][-1]['value']
    data_anno[index]['conversations'][-1]['value'] == gpt_val[idx]
    print(f"index = {index}, old_val = {old_val}, new_val = {gpt_val[idx]}")

with open("/home/coder/work/llava-665k/llava_v1_5_mix665k.json", "w") as f:
    json.dump(data_anno, f, indent=4)

print("Write back done")