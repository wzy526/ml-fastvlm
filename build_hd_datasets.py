import os
import json
import random
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# ================= 配置区 =================
SFT_DATA_DIR = "/mnt/ephemeral/sft_data"
TRAIN_SPLIT_DIR = os.path.join(SFT_DATA_DIR, "train_split")
OUTPUT_JSONL = os.path.join(SFT_DATA_DIR, "hd_supplements.jsonl")
# 中间文件：每个数据集处理完追加写入，支持 resume
STAGING_JSONL = os.path.join(SFT_DATA_DIR, "hd_supplements_staging.jsonl")
DONE_MARKER = os.path.join(SFT_DATA_DIR, "hd_supplements_done.txt")

# 确保子目录存在
for sub_dir in ["docvqa", "infovqa", "synthdog", "chartqa", "refcoco"]:
    os.makedirs(os.path.join(TRAIN_SPLIT_DIR, sub_dir), exist_ok=True)

# RefCOCO 提示词模板
REFCOCO_PROMPTS = [
    "Provide the bounding box for the region that this text describes: <expr>",
    "Where is <expr> located in the image? Please output the bounding box.",
    "Find the region matching the description '<expr>' and give its coordinates."
]
# ==========================================

def _load_done_stages():
    """读取已完成的阶段列表"""
    if os.path.exists(DONE_MARKER):
        with open(DONE_MARKER, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def _mark_done(stage_name):
    """标记一个阶段完成"""
    with open(DONE_MARKER, 'a') as f:
        f.write(stage_name + '\n')

def _append_jsonl(data_list):
    """将样本追加写入 staging JSONL"""
    with open(STAGING_JSONL, 'a') as f:
        for sample in data_list:
            f.write(json.dumps(sample) + '\n')

def normalize_bbox(coco_bbox, img_w, img_h):
    """将 [x_min, y_min, w, h] 转换为 [x1, y1, x2, y2] 的 0~1 浮点数"""
    x_min, y_min, w, h = coco_bbox
    x1 = max(0.0, min(1.0, x_min / img_w))
    y1 = max(0.0, min(1.0, y_min / img_h))
    x2 = max(0.0, min(1.0, (x_min + w) / img_w))
    y2 = max(0.0, min(1.0, (y_min + h) / img_h))
    return f"[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]"

def process_hf_vqa_dataset(dataset_name, split, subset_name, save_dir_name,
                           limit=None, question_key="question", answer_key="answers",
                           default_question="What is written in this image?"):
    """通用的 Hugging Face VQA 数据集下载与转换逻辑

    Args:
        question_key: 问题字段名 (DocVQA='question', ChartQA='query', SynthDog 无则用默认)
        answer_key: 答案字段名 (DocVQA='answers', ChartQA='label', SynthDog='text')
        default_question: 当样本中不存在 question_key 时的默认问题
    """
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

        # 组装 LLaVA 格式 — 通过 key 参数适配不同数据集的字段名
        question = item.get(question_key, default_question)
        answers = item.get(answer_key, '')
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

REFCOCO_DATASETS = [
    ("jxu124/refcoco",     "refcoco"),
    ("jxu124/refcocoplus", "refcoco+"),
    ("jxu124/refcocog",    "refcocog"),
]

def process_refcoco_from_hf():
    """从 Hugging Face 处理 RefCOCO / RefCOCO+ / RefCOCOg，并映射到 train2017"""
    coco_train2017_dir = os.path.join(TRAIN_SPLIT_DIR, "coco", "train2017")
    augmented_data = []
    sample_idx = 0

    for hf_name, short_name in REFCOCO_DATASETS:
        print(f"Loading {short_name} from {hf_name}...")
        ds = load_dataset(hf_name, split="train")

        for item in tqdm(ds, desc=f"Processing {short_name}"):
            image_id = item['image_id']
            img_filename = f"{int(image_id):012d}.jpg"
            img_full_path = os.path.join(coco_train2017_dir, img_filename)

            # 极少数 COCO2014 图被分到了 val2017，找不到就跳过
            if not os.path.exists(img_full_path):
                continue

            with Image.open(img_full_path) as img:
                img_w, img_h = img.size

            norm_bbox_str = normalize_bbox(item['bbox'], img_w, img_h)

            # jxu124 格式: sentences 是 [{"sent": "...", ...}, ...]
            # 展开每条 sentence 为独立样本，增加多样性
            for sent_obj in item['sentences']:
                expr = sent_obj['sent']
                prompt = random.choice(REFCOCO_PROMPTS).replace("<expr>", expr)

                llava_sample = {
                    "id": f"{short_name}_{sample_idx}",
                    "image": f"coco/train2017/{img_filename}",
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{prompt}"},
                        {"from": "gpt", "value": norm_bbox_str}
                    ]
                }
                augmented_data.append(llava_sample)
                sample_idx += 1

    print(f"RefCOCO total: {len(augmented_data)} samples "
          f"(from {len(REFCOCO_DATASETS)} subsets, sentences expanded)")
    return augmented_data

def main():
    random.seed(42)
    done_stages = _load_done_stages()
    total_new = 0

    # ---- 各阶段：已完成则跳过，否则处理后追加写入 ----

    # 1. DocVQA (~39k)
    if "docvqa" not in done_stages:
        data = process_hf_vqa_dataset(
            "HuggingFaceM4/DocumentVQA", "train", None, "docvqa")
        _append_jsonl(data)
        _mark_done("docvqa")
        total_new += len(data)
        print(f"  -> docvqa: {len(data)} samples written")
    else:
        print("[SKIP] docvqa already done")

    # 2. InfoVQA (~24k)
    if "infovqa" not in done_stages:
        data = process_hf_vqa_dataset(
            "Ahren09/InfoVQA", "train", None, "infovqa",
            answer_key="answer")
        _append_jsonl(data)
        _mark_done("infovqa")
        total_new += len(data)
        print(f"  -> infovqa: {len(data)} samples written")
    else:
        print("[SKIP] infovqa already done")

    # 3. SynthDoG (降采样 20万)
    if "synthdog" not in done_stages:
        data = process_hf_vqa_dataset(
            "naver-clova-ix/synthdog-en", "train", None, "synthdog", limit=200000)
        _append_jsonl(data)
        _mark_done("synthdog")
        total_new += len(data)
        print(f"  -> synthdog: {len(data)} samples written")
    else:
        print("[SKIP] synthdog already done")

    # 4. ChartQA (~28k)
    if "chartqa" not in done_stages:
        data = process_hf_vqa_dataset(
            "HuggingFaceM4/ChartQA", "train", None, "chartqa",
            question_key="query", answer_key="label")
        _append_jsonl(data)
        _mark_done("chartqa")
        total_new += len(data)
        print(f"  -> chartqa: {len(data)} samples written")
    else:
        print("[SKIP] chartqa already done")

    # 5. RefCOCO + RefCOCO+ + RefCOCOg
    if "refcoco_all" not in done_stages:
        data = process_refcoco_from_hf()
        _append_jsonl(data)
        _mark_done("refcoco_all")
        total_new += len(data)
        print(f"  -> refcoco_all: {len(data)} samples written")
    else:
        print("[SKIP] refcoco_all already done")

    # ---- 最终：读取 staging → shuffle → 写入最终 JSONL ----
    print(f"\n[Stats] This run added {total_new} new samples.")
    print(f"Reading staging file {STAGING_JSONL}...")
    all_data = []
    with open(STAGING_JSONL, 'r') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))

    print(f"Total {len(all_data)} samples. Shuffling...")
    random.shuffle(all_data)

    with open(OUTPUT_JSONL, 'w') as f:
        for sample in all_data:
            f.write(json.dumps(sample) + '\n')

    print(f"\nDONE! Saved {len(all_data)} samples to {OUTPUT_JSONL}.")

if __name__ == "__main__":
    main()
