#!/usr/bin/env python3
"""
基于LLaVA官方实现的TextVQA评估脚本
适配您训练的DAT-LLaVA-1.5模型
"""

import os
import json
import argparse
import torch
from PIL import Image
from tqdm import tqdm
import re
from collections import defaultdict

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model import LlavaLlamaForCausalLM, LlavaQwen2ForCausalLM
import transformers


def load_model_and_tokenizer(model_path, device_map="auto"):
    """加载模型和分词器 - 基于官方实现"""
    disable_torch_init()
    
    # 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    
    # 修复配置中的 decoder_config 问题
    config = transformers.AutoConfig.from_pretrained(model_path)
    if hasattr(config, 'decoder_config') and isinstance(config.decoder_config, dict):
        print("修复 decoder_config 配置...")
        if 'model_type' in config.decoder_config:
            decoder_config = transformers.AutoConfig.from_dict(config.decoder_config)
            config.decoder_config = decoder_config
        else:
            from transformers import PretrainedConfig
            decoder_config = PretrainedConfig.from_dict(config.decoder_config)
            config.decoder_config = decoder_config
    
    # 加载模型
    model_name = get_model_name_from_path(model_path)
    if 'qwen' in model_name.lower():
        print("Loading LlavaQwen2ForCausalLM model (Qwen2 backbone)")
        model = LlavaQwen2ForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        print("Loading LlavaLlamaForCausalLM model (LLaMA backbone)")
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    # 初始化视觉编码器
    print("初始化视觉编码器...")
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and not vision_tower.is_loaded:
            print("加载视觉编码器...")
            vision_tower.load_model()
            print("视觉编码器加载完成")
    
    # 加载图像处理器
    try:
        image_processor = transformers.CLIPImageProcessor.from_pretrained(model_path)
        print("使用模型路径的图像处理器")
    except OSError:
        print("模型路径没有图像处理器，使用vision_tower路径")
        vision_tower_path = "/home/zhuofan.xia/gsva_pretrains/clip-vit-large-patch14-336"
        image_processor = transformers.CLIPImageProcessor.from_pretrained(vision_tower_path)
    
    return model, tokenizer, image_processor


def prompt_processor(prompt):
    """官方LLaVA的prompt处理函数"""
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def load_textvqa_data(question_path, annotation_path=None, max_samples=None):
    """加载TextVQA数据 - 完全按照官方格式"""
    print(f"加载TextVQA问题: {question_path}")
    
    with open(question_path, 'r') as f:
        question_data = json.load(f)
    
    # 加载答案（如果有annotations文件）
    answers = {}
    if annotation_path and os.path.exists(annotation_path):
        print(f"加载TextVQA答案: {annotation_path}")
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
        
        # 构建答案映射 - 获取完整的10个答案
        for item in annotation_data['annotations']:
            if 'answers' in item and item['answers']:
                # 提取答案字符串列表（从字典格式中提取answer字段）
                answer_strings = [ans['answer'] for ans in item['answers'] if 'answer' in ans]
                answers[item['question_id']] = answer_strings
            else:
                answers[item['question_id']] = []
    
    samples = []
    # TextVQA格式：data['questions'] 是列表
    for item in question_data['questions']:
        question_id = item['question_id']
        sample = {
            'questionId': question_id,
            'imageId': item['image_id'],
            'question': item['question'],
            'answers': answers.get(question_id, [])  # 获取完整的答案列表
        }
        samples.append(sample)
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"加载了 {len(samples)} 个样本")
    return samples


def evaluate_single_sample(model, tokenizer, image_processor, sample, image_folder, conv_mode="llava_v1"):
    """评估单个样本"""
    try:
        # 加载图像
        image_id = sample['imageId']
        # TextVQA图像文件名格式可能不同，尝试多种格式
        possible_paths = [
            os.path.join(image_folder, f"{image_id}.jpg"),
            os.path.join(image_folder, f"{image_id}.png"),
            os.path.join(image_folder, f"n{image_id}.jpg"),
            os.path.join(image_folder, f"n{image_id}.png")
        ]
        
        image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if not image_path:
            return None, f"Image not found for ID {image_id}. Tried: {possible_paths[:2]}"
        
        image = Image.open(image_path).convert('RGB')
        
        # 构建对话 - 使用官方TextVQA格式
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        
        # 官方TextVQA prompt格式
        inp = f"Question: {sample['question']} Short answer:"
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        
        conv.append_message(roles[0], inp)
        conv.append_message(roles[1], None)
        prompt = conv.get_prompt()
        
        # 处理输入
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda()
        
        # 处理图像
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        # 生成回答
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=16,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # 提取回答
        if prompt in outputs:
            outputs = outputs.split(prompt)[-1].strip()
        else:
            outputs = outputs.strip()
        
        if outputs:
            lines = outputs.split('\n')
            if lines:
                outputs = lines[0].strip()
        
        return outputs, None
        
    except Exception as e:
        print(f"推理错误: {e}")
        return None, str(e)


def normalize_answer(answer):
    """官方TextVQA答案标准化处理"""
    if not answer:
        return ""
    
    # 转换为小写
    answer = answer.lower()
    
    # 移除标点符号
    import re
    answer = re.sub(r'[^\w\s]', '', answer)
    
    # 移除多余空格
    answer = ' '.join(answer.split())
    
    return answer


def calculate_anls_score(predictions, ground_truths):
    """使用官方TextVQA评估器计算ANLS分数 - 完全按照官方脚本"""
    try:
        from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
        
        # 构建官方评估器需要的格式 - 完全按照官方脚本
        pred_list = []
        for pred, gt in zip(predictions, ground_truths):
            pred_list.append({
                "pred_answer": pred,
                "gt_answers": gt  # gt已经是完整的答案列表
            })
        
        evaluator = TextVQAAccuracyEvaluator()
        accuracy = evaluator.eval_pred_list(pred_list)
        return accuracy
        
    except ImportError:
        print("警告: 无法导入官方评估器，使用简化版本")
        return calculate_anls_score_simple(predictions, ground_truths)


def calculate_anls_score_simple(predictions, ground_truths):
    """简化版ANLS计算（备用）"""
    def anls_score(pred, gt):
        if not pred or not gt:
            return 0.0
        
        # 标准化答案
        pred = normalize_answer(pred)
        gt = normalize_answer(gt)
        
        if pred == gt:
            return 1.0
        
        # 计算编辑距离
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(pred, gt)
        max_len = max(len(pred), len(gt))
        
        if max_len == 0:
            return 1.0
        
        # ANLS计算：基于编辑距离的相似度
        similarity = 1.0 - (distance / max_len)
        
        # 如果相似度低于0.5，返回0（官方TextVQA阈值）
        if similarity < 0.5:
            return 0.0
        
        return similarity
    
    total_score = 0.0
    for pred, gt in zip(predictions, ground_truths):
        total_score += anls_score(pred, gt)
    
    return total_score / len(predictions) if predictions else 0.0


def eval_single_official(annotation_file, result_file):
    """完全按照官方脚本的评估函数"""
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    
    # 加载annotations - 匹配实际JSON格式
    annotation_data = json.load(open(annotation_file))
    annotations = annotation_data['annotations']
    
    # 构建question_id到annotation的映射（更简单的方式）
    annotation_dict = {ann['question_id']: ann for ann in annotations}
    
    # 加载结果
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        question_id = result['question_id']
        if question_id in annotation_dict:
            annotation = annotation_dict[question_id]
            # 提取答案字符串列表（从字典格式中提取answer字段）
            gt_answers = [ans['answer'] for ans in annotation['answers'] if 'answer' in ans]
            pred_list.append({
                "pred_answer": result['text'],
                "gt_answers": gt_answers,
            })
        else:
            print(f"警告: 找不到question_id {question_id} 的annotation")

    evaluator = TextVQAAccuracyEvaluator()
    accuracy = evaluator.eval_pred_list(pred_list)
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * accuracy))
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="TextVQA评估")
    parser.add_argument("--model-path", default="/perception-hl/zhuofan.xia/vlm_exps/textdat/tdat-7b-l0d32-s12g8z3")
    parser.add_argument("--question-file", default="/perception-hl/zhuofan.xia/data/textvqa/val_questions.json")
    parser.add_argument("--annotation-file", default="/perception-hl/zhuofan.xia/data/textvqa/val_annotations.json")
    parser.add_argument("--image-folder", default="/perception-hl/zhuofan.xia/data/textvqa/train_images")
    parser.add_argument("--output-file", default="/perception-hl/zhuofan.xia/vlm_exps/textdat/textvqa_val_pred.jsonl")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--chunks", type=int, default=1, help="总的分块数")
    parser.add_argument("--chunk-idx", type=int, default=0, help="当前分块索引")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer, image_processor = load_model_and_tokenizer(args.model_path)
    
    # 加载数据
    samples = load_textvqa_data(args.question_file, args.annotation_file, args.max_samples)
    
    # 分块处理
    if args.chunks > 1:
        print(f"分块处理: {args.chunk_idx}/{args.chunks}")
        chunk_size = len(samples) // args.chunks
        start_idx = args.chunk_idx * chunk_size
        if args.chunk_idx == args.chunks - 1:
            # 最后一个分块处理剩余的所有样本
            end_idx = len(samples)
        else:
            end_idx = start_idx + chunk_size
        samples = samples[start_idx:end_idx]
        print(f"分块 {args.chunk_idx} 处理样本 {start_idx}-{end_idx-1} (共 {len(samples)} 个样本)")
    
    # 评估
    predictions = []
    ground_truths = []
    
    print("开始评估...")
    for i, sample in enumerate(tqdm(samples)):
        prediction, error = evaluate_single_sample(
            model, tokenizer, image_processor, sample, args.image_folder, args.conv_mode
        )
        
        if error:
            print(f"样本 {i} 错误: {error}")
            prediction = ""
        
        predictions.append(prediction)
        ground_truths.append(sample['answers'])  # 使用完整的答案列表
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 个样本")
    
    # 计算ANLS分数
    anls_score = calculate_anls_score(predictions, ground_truths)
    print(f"ANLS分数: {anls_score:.4f}")
    
    # 保存结果 - 使用官方格式
    with open(args.output_file, 'w') as f:
        for i, (pred, gt, sample) in enumerate(zip(predictions, ground_truths, samples)):
            result = {
                'question_id': sample['questionId'],
                'image_id': sample['imageId'],
                'question': sample['question'],
                'prompt': f"Question: {sample['question']} Short answer:",
                'text': pred,
                'ground_truth': gt,
                'anls_score': calculate_anls_score_simple([pred], [gt[0]] if gt else [''])
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"结果已保存到: {args.output_file}")
    print(f"最终ANLS分数: {anls_score:.4f}")


if __name__ == "__main__":
    main()
