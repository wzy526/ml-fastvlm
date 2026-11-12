# Qwen2.5-VL-7B：where-head / crop + learned DAT

本入口移植当前 `ldat_eval/infer.py` / `scripts/offset_feat_train.py` 路线：低分辨率图像与问题 → LLM 中间层 where-head → 自适应窗口 → learned density → inverse-CDF 像素采样 → mosaic 经原生 ViT 重编码 → global M-RoPE → 回答。

它使用官方 `Qwen2.5-VL-7B-Instruct` 或其合并后的 SFT checkpoint。`scripts/qwen2_5vl/` 和 `modeling_qwen2_5vl_dat.py` 是此前的 attention 注入路线，本入口不调用它们。不要把 `qwen2_5_vl_dat` 旧 checkpoint 当作本入口的 backbone。

## 模型差异与实现

- `ldat_eval/backbone.py` 适配模型加载、ViT 输出、语言模型层和 M-RoPE。尺寸读取 config / processor；Qwen2.5-VL 每个 merged token 是 `14 × 2 = 28` 像素，640 LR tokens 对应 `501760` max_pixels。
- `llava/model/where_dat.py` 共享定位头和窗口函数；32 像素默认值保留原 Qwen3.5 行为。
- `where_head.py --model_family qwen2_5_vl --head_prefix train` 和 sampler / 推理使用相同的 PIL resize、问题 + WHERE_HINT 前缀及中间层。没有答案或 GT box 输入推理定位头。
- Qwen2.5 使用普通 KV cache；不应用 Qwen3.5 的 Gated DeltaNet patch。`FAST_PREFIX=True` 截止 where 层并复用 LR ViT，`PREFIX_CACHE=True` 则复用完整 prefix cache；两者同时开启时会运行完整第一遍。
- 新头权重旁保存 `.pt.json` 元数据，同时保留既有 `.pt` 格式。加载检查模型族、通道数、视觉尺寸和 where 层。旧 Qwen3.5 head 不可用于 7B；必须重新训练。

上游核对依据：[7B 官方配置](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json)、[Transformers 5.6.1 实现](https://github.com/huggingface/transformers/blob/v5.6.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)。实现按仓库 `requirements-lock.txt` 中的 Transformers 5.6.1 测试；该版本 Qwen2.5 的 RoPE 也使用 `mm_token_type_ids`。

## 环境与单卡检查

使用已配置的训练环境，在仓库根目录执行；不用更改全局 Python。需要 PyTorch、torchvision、Transformers、accelerate、Pillow、NumPy、einops；可选 SFT 还需要 PEFT / DeepSpeed，评测需要你目前使用的 lmms-eval checkout。SDPA 路线不要求 FlashAttention。

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH=/absolute/path/to/Qwen2.5-VL-7B-Instruct

python scripts/qwen2_5_ldat/smoke.py --model_path "$MODEL_PATH"
# 进一步检查答案 CE 是否穿过冻结 ViT 回传到密度头；显存占用高于纯推理。
python scripts/qwen2_5_ldat/smoke.py --model_path "$MODEL_PATH" --backward
```

未传头权重时，smoke 只使用临时随机头测试接口，生成文本不代表准确率。可以传 `--where_head_pt`、`--ldat_pt` 检查已训练的头。7B 的正式训练显存与吞吐需在实际 GPU 上测量，本地小模型测试不提供这一保证。

## 训练

默认冻结官方 7B，先训练定位头，再训练 density head。复用已有数据格式：`image`、`conversations`、归一化 `bbox` / `bboxes`。`DATA_JSON` 是 boxed Visual-CoT，`REF_JSON` / `COCO_JSON` 给定位头，`DAT_JSON` 是答案监督数据（默认等于 DATA_JSON，也可换 SA-1B QA / 自己的混合数据）。

```bash
export OUT="$HOME/vldat_experiments/qwen25_7b_ldat"
export DATA_JSON=/absolute/path/to/viscot_bbox.train.json
export REF_JSON=/absolute/path/to/refcoco_bbox.json
export COCO_JSON=/absolute/path/to/coco_small_bbox.json
export IMAGE_ROOT=/absolute/path/to/train_split

DRY_RUN=1 bash scripts/qwen2_5_ldat/train_7b.sh
NPROC=8 bash scripts/qwen2_5_ldat/train_7b.sh
```

默认 where 层为 8，LR 上限 640 tokens，单窗口上限 400 tokens（随窗口长宽比向下取整）；where 1500 步、DAT 2000 步。DAT 使用 `ldat_in=native, pe=global, alpha=0.8, ratio=0, k=0.6, src_scale=4, supersample=2`。损失为答案 CE + `0.1 × box KL + 0.1 × trust-region KL`，`HINGE_W=0`。这是可配置的起始配方，不声称等同于某个已有实验的全部超参；用 `DAT_JSON`、`DAT_STEPS`、`BOX_W`、`TR_W`、`HINGE_W` 对齐要复现的实验。

输出 `$OUT/where_lm8.pt` 和 `$OUT/ldat.pt`（以及各自 JSON）。`STAGES=where` / `STAGES=dat` 可单独运行阶段。脚本会执行所选阶段，不会因为同名文件存在而静默跳过；如要新实验请设置新的 OUT。

若要复用目前 4B 实验的“先 Visual-CoT 答案 SFT，再训两个头”设置：

```bash
STAGES='sft where dat' NPROC=8 bash scripts/qwen2_5_ldat/train_7b.sh
# 后续评测必须指向 SFT 合并模型，不能回到官方未微调模型。
export MODEL_PATH="$OUT/sft-merged"
```

SFT 使用现有 `train_qwen_dat.py --use_dat False`，q/k/v/o LoRA 和 merger 微调；合并脚本不会转成旧 DAT attention 模型。合并目录非空时会停止，避免覆盖其他模型。SFT、head、DAT 的数据划分仍由你控制：建议按原始图像划分并预先去除 benchmark 重叠；现有两个头训练器的 held-out 是行级拆分，不保证同图隔离。默认训练不读取 V* benchmark (`--vstar_n 0`)。

7B 每个 torchrun rank 持有完整冻结模型。DAT 默认每次处理 1 张图、`DAT_ACCUM=1`；这不会做张量并行或自动分片。多窗口训练和评测都必须显式设置相同的 `N_WIN`、`GRID`，例如 `N_WIN=2 GRID=20` 表示每窗 400 tokens。改变 NPROC / accumulation 会改变有效 batch，正式对比需要记录。

## 评测与比较

```bash
export RUN_ROOT="$OUT"
export LMMS_EVAL_DIR=/absolute/path/to/your/lmms-eval
DRY_RUN=1 LIMIT=8 bash scripts/qwen2_5_ldat/eval_7b.sh
NPROC=1 LIMIT=8 bash scripts/qwen2_5_ldat/eval_7b.sh
NPROC=8 bash scripts/qwen2_5_ldat/eval_7b.sh

# 同一 backbone 的纯原生基线；用 BASE_TOK 选择输入预算。
MODE=base BASE_TOK=640 NPROC=8 bash scripts/qwen2_5_ldat/eval_7b.sh
```

默认任务为 `vstar_bench hrbench4k hrbench8k`，可以用 TASKS 覆盖。模型注册名是 `qwen2_5_ldat`，保留 lmms-eval 的 task prompt、生成长度和 scorer，保存逐样本结果。baseline 使用现有 lmms-eval 的 `qwen2_5_vl`。请使用同一 lmms-eval commit / task 版本运行 baseline 和 DAT。

`offset_feat_train.py --eval_dataset` 仍是旧的选项首 token logits 评分，不应和 lmms-eval 完整生成的分数混合。截图缺少列标题、完整表格和评测协议，仅凭图片不能确定所有 RL / OPD 分数的指标。正式表格至少注明 backbone、数据和监督（包括 bbox）、prompt、解码预算、LR/mosaic token 数以及延迟测量方式；相同 7B backbone 本身不保证训练数据、推理预算或监督等价。

当前 inference 支持 density-only head；如 checkpoint 额外包含未合并 LoRA 或 window-scale head，会明确报错，不会丢掉这些权重后输出一个看似可比较的分数。

## 本地验证

```bash
python -m unittest discover -s tests -p 'test_ldat*.py'
bash -n scripts/qwen2_5_ldat/train_7b.sh scripts/qwen2_5_ldat/eval_7b.sh
```

测试用本地构建的 tiny Qwen2.5-VL（真实 HF 模型和 processor），覆盖像素几何、梯度、cache、fast-prefix、多窗口、两个训练 CLI 的一步训练及保存重载、SFT LoRA/merger 合并、smoke 和 shell dry-run。测试不下载 7B 权重，不产生正式 benchmark 成绩。

2026-11-12 本次验证：14 项新增测试通过；shell 语法、dry-run、Python 编译和 `git diff --check` 通过。独立代码审查发现的共享 helper 缺失 `bisect` 导入已由失败用例复现并修复；默认训练的 `crop,dat,ldat` 三路现均进入 CLI 测试。还用修改前 HEAD 的函数核对了 3 组 Qwen3.5 默认 pixel-DAT 输入，输出图像字节和几何一致。

原仓库 `python -m unittest discover -s scripts -p 'test_*.py'` 运行 28 项，出现 1 failure + 9 errors。将修改前 HEAD `2512ca2` 解包到临时目录，在同一环境重跑，得到相同失败；本次未修改这些旧测试：

- `test_dat_experiments.ExperimentTests.test_fixed_coarse_zero_origin_and_reader_freeze_after_step`、`test_multi_image_multi_answer_readers_have_distinct_keys_and_400_tokens`、`test_per_group_override_ce_gradient_and_finite_difference`、`test_trainer_replays_reference_through_checkpoint_backward_then_cleans_up`：旧 fixture 缺 `glob_grid`。
- `test_dat_loading_geometry.LoadingTests.test_auto_and_bf16_preserve_all_66_original_tensors`：断言仍写 66 个 tensor，当前为 72。
- `test_hd_shuffle.HDShuffleTests.setUpClass`：引用已移动到 archive 的 `_test_lr_drop_leverage.py` 旧路径。
- `test_probe_report.ProbeReportTest.test_constant_coordinates_have_undefined_correlation`、`test_full_report_keeps_accuracy_and_serializes`、`test_hd_diagnostics_without_global_offset`、`test_small_hd_sample_skips_correlation`：旧 fixture 缺 `dump_ground`。

未验证范围：真实 7B 权重、CUDA/bfloat16、分布式训练、实际 lmms-eval task/plugin 运行和完整 Qwen3.5 模型回归。正式启动前先在目标 GPU 环境跑上面的 smoke 和 `LIMIT=8` 评测。
