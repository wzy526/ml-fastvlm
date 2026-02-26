import os
import json
import random
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# ================= 配置区 =================
SFT_DATA_DIR = "./sft_data"
TRAIN_SPLIT_DIR = os.path.join(SFT_DATA_DIR, "train_split")
BASE_LLAVA_JSON = os.path.join(SFT_DATA_DIR, "llava_v1_5_mix665k_filtered.json")
FINAL_OUTPUT_JSON = os.path.join(SFT_DATA_DIR, "llava_v1_5_mix665k_shuffled.json")

# 确保子目录存在
for sub_dir in ["docvqa", "infovqa", "synthdog", "refcoco"]:
    os.makedirs(os.path.join(TRAIN_SPLIT_DIR, sub_dir), exist_ok=True)

# RefCOCO 提示词模板
REFCOCO_PROMPTS = [
    "Provide the bounding box for the region that this text describes: <expr>",
    "Where is <expr> located in the image? Please output the bounding box.",
    "Find the region matching the description '<expr>' and give its coordinates."
]
# ==========================================

def normalize_bbox(coco_bbox, img_w, img_h):
    """将 [x_min, y_min, w, h] 转换为 [x1, y1, x2, y2] 的 0~1 浮点数"""
    x_min, y_min, w, h = coco_bbox
    x1 = max(0.0, min(1.0, x_min / img_w))
    y1 = max(0.0, min(1.0, y_min / img_h))
    x2 = max(0.0, min(1.0, (x_min + w) / img_w))
    y2 = max(0.0, min(1.0, (y_min + h) / img_h))
    return f"[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]"

def process_hf_vqa_dataset(dataset_name, split, subset_name, save_dir_name, limit=None):
    """通用的 Hugging Face VQA 数据集下载与转换逻辑"""
    print(f"Loading {dataset_name} ({subset_name})...")
    ds = load_dataset(dataset_name, subset_name, split=split)
    
    if limit:
        # 随机降采样 (应对 SynthDog 这类 50万条的巨无霸)
        indices = random.sample(range(len(ds)), limit)
        ds = ds.select(indices)
        
    augmented_data = []
    save_dir = os.path.join(TRAIN_SPLIT_DIR, save_dir_name)
    
    for idx, item in enumerate(tqdm(ds, desc=f"Processing {save_dir_name}")):
        img = item['image']
        # 生成唯一的文件名并保存原图
        img_filename = f"{save_dir_name}_{idx}.jpg"
        img_path = os.path.join(save_dir, img_filename)
        
        # 存图到本地 train_split/ 目录
        if not os.path.exists(img_path):
            img.convert("RGB").save(img_path)
            
        # 组装 LLaVA 格式
        # 注意：DocVQA 字段是 'question', 'answers'；SynthDog 是 'text' (没问题，我们动态适配)
        question = item.get('question', "What is written in this image?")
        answers = item.get('answers', [item.get('text', '')])
        answer = answers[0] if isinstance(answers, list) else answers
        
        llava_sample = {
            "id": f"{save_dir_name}_{idx}",
            "image": f"{save_dir_name}/{img_filename}",  # 相对路径
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": answer}
            ]
        }
        augmented_data.append(llava_sample)
        
    return augmented_data

def process_refcoco_from_hf():
    """从 Hugging Face 处理 RefCOCO，并映射到 train2017"""
    print("Loading RefCOCO...")
    # 这里使用社区处理好的扁平化 RefCOCO 集合
    ds = load_dataset("PaDT-MLLM/RefCOCO", split="train")
    augmented_data = []
    
    coco_train2017_dir = os.path.join(TRAIN_SPLIT_DIR, "coco", "train2017")
    
    for idx, item in enumerate(tqdm(ds, desc="Processing RefCOCO")):
        # 1. 提取 ID 并映射到 train2017 文件名
        image_id = item['image_id']
        img_filename = f"{int(image_id):012d}.jpg"
        img_full_path = os.path.join(coco_train2017_dir, img_filename)
        
        # 因为极少数(不到4%)的 COCO2014 图被分到了 val2017，如果在 train2017 找不到就跳过
        if not os.path.exists(img_full_path):
            continue
            
        # 2. 读取图像宽高进行归一化
        with Image.open(img_full_path) as img:
            img_w, img_h = img.size
            
        norm_bbox_str = normalize_bbox(item['bbox'], img_w, img_h)
        prompt = random.choice(REFCOCO_PROMPTS).replace("<expr>", item['sentences'][0])
        
        llava_sample = {
            "id": f"refcoco_{idx}",
            "image": f"coco/train2017/{img_filename}",
            "conversations": [
                {"from": "human", "value": f"<image>\n{prompt}"},
                {"from": "gpt", "value": norm_bbox_str}
            ]
        }
        augmented_data.append(llava_sample)
        
    return augmented_data

def main():
    random.seed(42)
    all_new_data = []
    
    # 1. 下载并处理 DocVQA (~39k)
    all_new_data.extend(process_hf_vqa_dataset("HuggingFaceM4/DocVQA", "train", "DocVQA", "docvqa"))
    
    # 2. 下载并处理 InfoVQA (~24k)
    all_new_data.extend(process_hf_vqa_dataset("HuggingFaceM4/infovqa", "train", "infovqa", "infovqa"))
    
    # 3. 下载并处理 SynthDoG (官方有50万，这里我们随机降采样保留 20万)
    all_new_data.extend(process_hf_vqa_dataset("naver-clova-ix/synthdog-en", "train", "synthdog-en", "synthdog", limit=200000))
    
    # 4. 处理 RefCOCO (自动映射现有 COCO 图片)
    all_new_data.extend(process_refcoco_from_hf())
    
    print(f"\n[Stats] Generated {len(all_new_data)} augmented samples.")
    
    # 5. 合并基础数据
    print(f"Loading base dataset from {BASE_LLAVA_JSON}...")
    with open(BASE_LLAVA_JSON, 'r') as f:
        llava_base_data = json.load(f)
        
    final_merged_data = llava_base_data + all_new_data
    
    # 6. 全局打乱 (必须做！)
    print("Shuffling final dataset...")
    random.shuffle(final_merged_data)
    
    # 7. 保存终极 JSON
    with open(FINAL_OUTPUT_JSON, 'w') as f:
        json.dump(final_merged_data, f)
        
    print(f"\n🎉 ALL DONE! Saved {len(final_merged_data)} samples to {FINAL_OUTPUT_JSON}.")

if __name__ == "__main__":
    main()