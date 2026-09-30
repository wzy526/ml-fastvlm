# Offset 学习实验计划

初版日期：2026-09-29；修复记录更新：2026-09-30。初版代码基线：`262a26c`（`main`）。本文区分代码实现、回归测试与真实模型实验；修复状态见下文，GPU 实验结果仍需运行记录。

目标：固定每张图、每个 DAT 层的 `20×20=400` 个采样 token，让采样位置随问题和图像证据改变，并稳定降低答案损失。暂不做自适应点数，也不增加定位辅助 loss。8 个 offset group 是一个 token 内的分组采样，不应把预算记成 3200 个 reader token；不同 HD reader 分支的实际开销另记。

## 1. 执行顺序与当前状态

| 阶段 | 要回答的问题 | 当前状态 | 进入下一阶段的条件 |
| --- | --- | --- | --- |
| E0 环境与基线 | 当前 checkpoint 能否读取 HD 内容？ | 已修报告崩溃与不同图配对；第 3 节命令可运行 | 环境检查通过；记录 real/off/shuffle/oracle 基线及限制 |
| E1 query 路由 | 首个答案 token 是否使用问题条件 HD？ | 已独立修复训练路由；真实张量/GPU 检查待运行 | 单标签、短答案、多轮、首答案 query 与 prefill 对齐测试通过 |
| E2 条件与几何 | local query、采样位置、位置编码是否一致？ | 已增加 query/slot/冻结参考粗网格控制；真实模型验收待运行 | 因素分开控制，oracle 与 learned 使用相同编码约定 |
| E3 CE 梯度 | 答案损失能否给局部移动提供有效方向？ | 已增加逐组坐标梯度与有限差分；CPU 回归通过，GPU 待运行 | 坐标梯度数值检查通过；直接优化坐标能获得收益 |
| E4 局部头 | HD 邻域是否比当前 LR 输入更适合预测修正？ | HD 邻域 refiner 尚未实现 | sampler-only 短训在验证集优于零残差与 LR-local |
| E5 因果验收 | 提升是否来自问题相关的位置选择？ | 需新增干预与结果汇总 | 固定预算反事实对照、独立测试集、重复种子结果支持结论 |

先跑 E0 留档；正式训练以 E1–E3 通过为前提。已实现的实验控制与 CLI 见第 9 节；其他待实现功能仍需开发后才能启动实验。

### 2026-09-30 修复与回归命令

- FP32 加载：Qwen3.5 DAT 声明 HF 的 FP32 保留模块，并从原始 checkpoint 直接恢复这些参数为 FP32；`torch_dtype='auto'` 和显式 bf16 均不再经过 bf16 中转。reader 的 k/v_hd 仍沿用模型精度。
- LR-first：训练、probe 和 TTFT benchmark 用 `patch_size` 将未合并 `image_grid_thw` 换算为像素，尺寸对齐仍用 `patch_size * merge_size`；消除重复乘 merge size 导致的面积四倍误差。
- 初始化：训练入口解析参数后立即调用 `transformers.set_seed(training_args.seed)`，先于模型和 DAT 参数构造。

```bash
python scripts/test_dat_loading_geometry.py
```

此回归用临时 safetensors 检查 66 个非 bf16 可表示的 FP32 参数逐位保留，兼顾矩形图和构造前 seed 顺序；需要 PyTorch/safetensors。真实 checkpoint 的张量数由层配置决定。

- E0 报告：原 `_corr` 只在有 `GLOBC` 时定义，但无 global offset 也可能有 HD diag12 统计；现已无条件定义，避免报告阶段 `UnboundLocalError` 导致 JSON 未生成。
- shuffle：probe 原来只比较前 4096 字节、搜索 8 个候选；leverage 原来直接使用半个数据集之外的行。现两者共用完整解码 RGB 像素与尺寸的 SHA256、全候选确定性搜索，并在模型加载前排除无不同图 donor 的输入。结果保存配对以便审计。
- E1：训练答案 query 修为 `[s−1,e)`，question-HD 边界同步；`labels[0]` 不参与 shifted CE，解析时排除。只改路由，不改 local query、位置编码和 offset head。

```bash
python scripts/test_probe_report.py
python scripts/test_hd_shuffle.py
python scripts/test_qwen35_dat_routing.py
```

前两项需要 NumPy/Pillow，无需模型或 GPU。路由测试的区间与缓存早退检查只需标准库；4 项真实张量解析检查在有 PyTorch 时执行，否则明确跳过。本地已通过 4 项报告、8 项 shuffle、6 项路由检查；4 项 PyTorch 检查因本地未安装而跳过。公司机器需补跑它们及真实模型的 exact merge/前向反向检查。

## 2. 内网机器同步与固定实验条件

在公司电脑的仓库根目录、`main` 分支上执行：

```bash
git status --short --branch
# 工作区干净且没有本地分叉时执行；有改动先保留和核对，不要 reset/clean。
git pull --ff-only origin main
git log -3 --oneline
```

公司电脑只需 pull，不需要 push。代码与计划从可推送的电脑同步；数据、模型和运行结果留在运行机器。本文不要求从内网导出或上传结果。

实验路径在 Bash 中设置，下面三个 `/absolute/path/...` 必须改成运行机器上的真实路径。`MODEL_PATH` 应是自包含的完整 HF checkpoint，而不是单独的 LoRA adapter。沿用公司机器已经可用的训练环境，无需为本计划重新安装环境。

```bash
set -euo pipefail
export MODEL_PATH='/absolute/path/to/full-hf-checkpoint'
export HELDOUT_JSON='/absolute/path/to/viscot_bbox.heldout.json'
export IMAGE_FOLDER='/absolute/path/to/train_split'
export RUN_ROOT="$HOME/offset_runs/20260929"
mkdir -p "$RUN_ROOT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DAT_EXACT_MERGE_GRAD=1
export DAT_ATTN_BACKEND=auto
export CUDA_VISIBLE_DEVICES=0
git rev-parse HEAD > "$RUN_ROOT/code_commit.txt"
git diff > "$RUN_ROOT/local_changes.patch"
```

固定以下条件，变化必须在结果中单独标注：

- 同一 checkpoint、DAT 层集合、全局中心/尺度来源、reader 配置和预处理；不得让某个实验重新初始化 reader。
- `grid_size=20`；probe/leverage 从 checkpoint 读取此值，二者没有 `--grid_size` 参数。不要靠改 JSON 假装 checkpoint 已适配新预算。
- 第一轮 LR 上限 256 token，`min_pixels=28224`，HD cap `5017600`，`hr_scale=3`，与下面命令一致。它们是诊断设置，不代表原训练的 resize 分布；另做部署分辨率复核。
- 同一组样本、同一顺序、同一 tokenizer 和答案截断规则；训练/验证/测试按图像划分，避免同图不同问泄漏。
- 有 bbox 的验证数据先取 64 例做检查，再取最多 500 例评估；同图不同目标的问题对单列。英文短答案/OCR 适合现有 synth 评分，其他任务需要对应的评分器。
- 梯度诊断关闭随机 dropout，固定随机种子；同时分别记录答案内容 token 和结束符/模板 token 的 CE，不能仅看后者下降。

下面检查 checkpoint 的预算，并保存诊断子集、输入顺序、数据哈希与配置。它不修改原 checkpoint 或原数据。

```python
import hashlib
import json
import os
from pathlib import Path

model = Path(os.environ["MODEL_PATH"])
data_path = Path(os.environ["HELDOUT_JSON"])
image_root = Path(os.environ["IMAGE_FOLDER"])
out = Path(os.environ["RUN_ROOT"])
config = json.loads((model / "config.json").read_text())
dat = config.get("dat_extra_args") or {}
assert dat.get("grid_size") == 20, dat
docs = json.loads(data_path.read_text())
subset = docs[:500]
assert len(subset) >= 2, "shuffle 对照至少需要两张不同图像"
paths = []
for doc in subset:
    assert isinstance(doc.get("image"), str), "第一轮仅使用单图样本"
    box = doc.get("bbox")
    assert box and len(box) == 4
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    assert len(doc.get("conversations", [])) >= 2
    path = image_root / doc["image"]
    assert path.is_file(), path
    paths.append(str(path))
assert len(set(paths)) >= 2
(out / "eval_subset.json").write_text(json.dumps(subset, ensure_ascii=False, indent=2))
(out / "checkpoint_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
manifest = {
    "source": str(data_path),
    "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    "subset_sha256": hashlib.sha256((out / "eval_subset.json").read_bytes()).hexdigest(),
    "ordered_images": paths,
    "n": len(subset),
    "dat_extra_args": dat,
}
(out / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(f"Saved {len(subset)} samples; checkpoint grid_size=20")
```

将此 Python 块保存为运行目录中的脚本后执行，或直接用 `python - <<'PY'` 在终端执行。多个问题可以来自同一图像；shuffle 现已按全图指纹排除相同内容，并输出实际 donor 配对。

## 3. E0：现有脚本可直接运行的检查

### 3.1 坐标约定与 exact merge

```bash
python scripts/verify_sampling_coordinates.py \
  2>&1 | tee "$RUN_ROOT/sampling_geometry.log"

python scripts/_test_exact_merge_grad.py \
  2>&1 | tee "$RUN_ROOT/exact_merge_grad.log"
```

第一项检查矩形图的 `(x,y)` 约定；第二项需要 CUDA 和可用的 FlashAttention backward，对照拼接 KV 的 eager attention 检查前向及 Q/K/V 梯度。查看日志中的误差和检查结果，不能把“命令结束”单独当作通过。

`DAT_EXACT_MERGE_GRAD=1` 并不保证实际模型每次都走 exact 路径：当前模型还要求 `hd_gate is None`。E3 要记录实际选择的分支。这两个脚本均不覆盖答案 query 路由，也不覆盖端到端坐标梯度。

### 3.2 real / shuffle / oracle / 错窗基线

```bash
for source in real shuffle oracle oracle_rand; do
  python scripts/probe_whd_qwen35.py \
    --model_path "$MODEL_PATH" \
    --dataset synth \
    --synth_json "$RUN_ROOT/eval_subset.json" \
    --synth_image_root "$IMAGE_FOLDER" \
    --max_samples 500 --max_new_tokens 32 \
    --tok_budget 256 --min_pixels 28224 \
    --hd_cap 5017600 --hr_scale 3 \
    --oracle_min_cells 20 \
    --hd_source "$source" --hd_bias 0 \
    --out "$RUN_ROOT/probe_${source}.json" \
    2>&1 | tee "$RUN_ROOT/probe_${source}.log"
done
```

每次调用也会运行 HD-off，无需传不存在的 `--hd_source off`。初次检查可把 `--max_samples` 改成 64，输出文件另起名字。`--dataset synth` 实际兼容 Visual-CoT 的 LLaVA 格式 bbox 数据；bbox 必须为 `[x0,y0,x1,y1]` 的图像比例坐标。

解释结果时必须保留这些限制：

- synth 使用归一化英文/数字字符串的包含匹配，不是通用 VQA 评分；长答案、中文或否定句不能只依赖它的 accuracy。需要保留 raw prediction 供检查。
- 当前 learned 与 oracle 的位置编码不同，real-vs-oracle 不能单独归因于 offset，见 E2。
- `oracle_min_cells=20` 是窗口最小边长所对应的 HD 特征单元数，不是采样点数；oracle 窗可能大于原 bbox。
- `oracle_rand` 尝试避开 GT，失败时使用最远候选，仍可能重叠；正式错窗对照要统计实际重叠率。
- probe 按文件顺序取前 N 个样本；输出 `sample_index`，shuffle 还输出目标/来源图像指纹与 `shuffle_source_index`，应和 `input_manifest.json` 一起保存。
- probe shuffle 保留目标 LR 输入、问题和 HD 尺寸，但仍重新计算 offset 与隐藏状态。这是 HD 替换对照，尚不是固定坐标重放对照。全图指纹只保证原始解码像素不同，不保证语义、缩放后特征或答案不同。donor 可以重复使用；它不是保证一一对应的行排列。
- diag12 的 `ib_s/2`、`ib_shd` 是候选网格落框率的离线几何计算，并未重采样生成答案；`covered` 子集只要求已有采样点命中框。它还平均了 group 坐标，不等于逐组真实覆盖。

### 3.3 HD 内容是否影响答案 CE

```bash
python scripts/_test_lr_drop_leverage.py \
  --model_path "$MODEL_PATH" \
  --data_json "$RUN_ROOT/eval_subset.json" \
  --image_folder "$IMAGE_FOLDER" \
  --n 64 --seed 0 --ratio 0.75 \
  --tok_budget 256 --min_pixels 28224 \
  --hd_cap 5017600 --hr_scale 3 \
  --oracle_min_cells 20 --max_answer_tokens 128 \
  --question_hd \
  --out "$RUN_ROOT/readout_leverage.json" \
  2>&1 | tee "$RUN_ROOT/readout_leverage.log"
```

优先查看：`S−O`（LR drop 下错误图相对 oracle 的 CE 差）、`D−P`（full LR 下 HD-off 相对 oracle 的差），以及对应 k/v_hd 梯度。它们是读出能力诊断。

该脚本不训练，不调用 optimizer.step；`g_off` 只统计部分 offset 卷积参数，不是逐点坐标梯度。它将 A/B/C/D/O/S/P 的 question reader 关掉、HD bias 置零；`--question_hd` 仅另加 E 条件，不能把全部结果当成原 checkpoint 的部署行为。脚本使用 train mode，且未全面关闭 checkpoint 的辅助梯度 hook；其梯度数值不能证明“纯 CE 在训练 offset”。E3 必须使用专门的纯 CE 诊断。

O/S 共享目标 oracle 窗口、对应位置编码、HD 尺寸和 dropout seed，只替换 HD 图像。输出的 `shuffle_pairs` 是完整预选配对表；`per_sample` 中第 j 条结果对应 `sample_indices[j]`，再用该索引查配对，以兼容被跳过的样本。leverage 会按 seed 重排/过滤输入，所以它与 probe 的候选池、次序和 donor 不一定相同。

## 4. E1：答案 query 路由修复及验收

实现位置：[modeling_qwen3_5_dat.py](../llava/model/language_model/modeling_qwen3_5_dat.py) 的 `compute_image_range_list` 和 `Qwen3_5AttentionDAT.forward`。

答案标签的端点 `[s,e]` 都包含在监督内；修复前问题条件 HD 的训练 query 为 `[s,e)`，遗漏预测首答案 token 的 `s−1` 行。现已通过 `_dat_answer_query_range` 统一为 `[max(s−1,0),e)`，并用 `_dat_question_ranges` 同步切分 question-HD，保证正常多轮各 segment 不重叠。

prefill 继续使用现有 `[intention_idx+1,Nq)`，cached decode 继续走原早退路径。此次修复使训练与 prefill 的首答案预测行都使用问题条件 HD，但没有声称整个 assistant 前缀逐行路由或 logits 完全相同；更早的模板行仍有原来的路由差异。文本 span 元数据也保持原含义。

必须验证：

1. 单个有效标签 `s=e` 时仍有一个答案 query；普通短答案、结束符与换行的监督位置逐 token 打印。
2. 多轮对话的每个答案段均正确；question-HD 开/关都检查；padding 和截断不产生负索引或越界。
3. 同一完整 prompt 下，训练预测首答案 token 的位置与推理 prefill 均归入问题条件 HD；模型级验证不预设所有前缀 logits 完全相同。
4. 只对首个答案内容 token 的 CE backward，确认其对应坐标有计算图；零梯度需区分 reader 没有利用 HD 与路由断开。

此处先做实现正确性修复，不与新 head、LoRA 或新 loss 混在一个性能实验里。修复前基线只用于定位问题，正式消融共用修复后实现。

## 5. E2：分别核对问题条件、局部证据与位置编码

### 5.1 local query 与 global query 独立控制

`glob_query_pos=ans_prev` 只影响全局定位。局部头现在由独立的 `local_query_pos` 控制：`im_start` 取 `ar[2]`，`ans_prev` 取训练答案前一行或 prefill 最后一行；对应 CLI 为 `--dat_local_query_pos`，默认保持 `im_start`。局部 gate 和 spatial guide 共用这一选择，global query 保持独立。选择写入 checkpoint 配置，训练与推理均读取它。

用该实验路径对比 `im_start` 与 `ans_prev=s−1`。保持同一 coarse grid、reader、局部头容量、初始化、样本和步数。对旧 checkpoint 直接换 query 的结果只作即时干预诊断，不代表重训后的能力上限。

同图问题交换时，仅交换送给采样器的问题表示，reader 继续接收原问题。观察采样是否朝各自证据移动，以及原问题对应的坐标是否更有利于原答案。

### 5.2 头的输入应对应最终采样附近

当前 offset 来自原 LR 网格卷积特征，随后直接加到 `c+s*r`。它尚未在粗定位后的 HD 位置重新读取邻域。这是待验证的设计限制，不是已经实现的 HD refiner。

第一轮固定 coarse grid：`x0_i = c + s * r_i`，由独立冻结参考模型生成并缓存/回放 `c/s` 或 `x0`；只冻结当前模型的主干权重不足以保证粗网格不变。明确 checkpoint 原有 local offset 是否被置零，不能一组叠加旧残差而另一组替换旧残差。新局部头使用问题表示、`x0_i` 附近的 HD 特征、邻域相对坐标，输出残差。

### 5.3 先统一 position encoding，再解释 oracle 上限

当前 `_construct_hd_position_ids` 的普通路径用固定全图 slot 网格，强制采样路径却按 forced 坐标生成位置。最小对照先实现：**learned、零残差、oracle、随机窗口均使用同一固定 slot 编码**，只改变采样内容。

在这个对照上，先确认 oracle 相对错图/错窗和 coarse grid 的收益。如果要比较随坐标变化的 RoPE，单开实验：

- 8 个 group 各有坐标，拼接后才投影成一个 token，需要明确该 token 的代表位置；group 均值只是约定，不等于每组真实采样位置。
- 若希望位置编码也对坐标可微，不能无说明地 `round().long()`；整数化会截断这条位置梯度，但不等于 grid_sample 的内容梯度也消失。
- E3 先用固定 slot 编码，单独判断内容采样带来的梯度，避免一次改变两条路径。

编码对照已通过 `--dat_hd_position_mode slot` 实现；默认 `legacy` 保持旧 checkpoint 行为。`DATSamplingControl` 的作用域也强制采用 slot 编码，learned、零残差、共享 oracle 网格和逐组 override 均共用这一位置约定。center-only、scale-only 的真实模型干预仍需另行运行。

## 6. E3：证明纯 next-token CE 能指导局部移动

### 6.1 配置与冻结边界

以下是现有训练入口的配置对照表，不是一条 sampler-only 启动命令：

| 设置 | 第一轮诊断取值 | 原因 |
| --- | --- | --- |
| `TF_PROB` | `0` | 不生成用于 bbox 强制采样的窗口 |
| `OFF_SUP_WEIGHT` / `REL_SUP_WEIGHT` / `OFF_PENALTY` | 全部 `0` | 排除通过 backward hook 注入的额外梯度 |
| `--kd_on` | `False` | 只保留答案 CE |
| `LR_DROP_PROB` | `0` | 先测部署输入下的坐标信号；LR drop 另做读出诊断 |
| `HD_LSE_BIAS` / `HD_LSE_BIAS_DECAY` | 全部 `0` | 固定 HD 融合权重设置，不混入 bias curriculum |
| `LORA_ENABLE` / `TUNE_MM_MLP` | `False` | 第一轮固定 reader 与特征来源 |
| `FREEZE_BASE` | `True`，但还不充分 | 脚本仍会解冻 DAT reader 参数，必须额外限制训练参数 |
| `OFF_HEAD_TRUNK_GRAD` | 固定粗定位时可为 `0` | 冻结 trunk 输入不妨碍独立 local head 学习 |
| `DAT_EXACT_MERGE_GRAD` | `1`，并记录实际分支 | 显式 HD gate 等设置可能使模型改走 legacy 路径 |

[`exp_sft_qwen35_2b_dat_readers_bias_genvs.sh`](../scripts/qwen3_5_adl_0915/exp_sft_qwen35_2b_dat_readers_bias_genvs.sh) 默认 `OFF_PENALTY=1.0`，且训练入口最终会按 `DAT_KEYS_MATCH` 解冻 DAT 参数。`FREEZE_BASE=True` 不等于 sampler-only：新的冻结逻辑须在这些解冻操作之后、optimizer 创建之前生效，并打印最终可训练参数与 optimizer 参数清单。

sampler-only 时冻结 ViT、merger、LLM、k/v_hd、HD layernorm、HD gate/bias 和 coarse locator，只训练选定 local head。reader 参数 `requires_grad=False`，但 reader 前向仍须保留对输入的反向图。冻结并缓存 HD 特征可以；不要将采样或整个 reader 放进 `no_grad()`。

如果 coarse locator 是 parameter-free trunk attention，且 trunk 冻结或对应梯度缩放为零，offset loss 不会沿这条定位路径更新 trunk 参数。后续要训练此路径时，需明确可训练 Q/K/上游参数，另开实验。

但仅冻结主干权重不保证后层 `c/s` 数值恒定：较早层 DAT 的变化仍可能改变后层输入。需要严格固定粗定位时，使用同一冻结参考模型或缓存的 `c/s`，并记录其跨 step 是否漂移。冻结检查同时比较 optimizer.step 前后的参数变化，确认 reader 没有被其他解冻逻辑重新放开。

### 6.2 逐点梯度与局部数值验证

先用 16–32 个有明确答案证据的验证样本，保留最终采样坐标的梯度。记录每层/每组：坐标梯度范数、非零比例、clamp 前越界比例、边界点比例、残差范围、tanh 饱和比例，以及首答案 token 的独立 CE 梯度。

在少量远离边界与插值单元折点的坐标上，用中心有限差分检查 `dL/dx` 的符号与量级。扰动用 HD 单元定义，例如 `0.01/0.05/0.1` 个单元；检查多个步长，避免 bf16 舍入让微小 loss 差淹没。固定随机性和样本；在可行范围采用 fp32 的小规模参考计算。

再沿归一化负梯度移动少量坐标，与相同步长的正方向、随机方向比较 CE。只在光滑局部和数值精度允许的范围解读，不要求每个样本每一步都下降；观察配对均值、符号一致性与误差。

`scripts/check_dat_coordinate_grad.py` 已实现纯 CE、content/template/first 三类分项梯度和中心有限差分，具体命令见第 9 节。负梯度移动与直接坐标优化仍是后续真实模型实验，不能由 CPU 回归代替。

### 6.3 直接优化坐标，区分信号不足与 head 学不会

冻结全部模型权重，把每样本、每层的坐标残差暂时作为独立参数；固定 coarse grid、slot RoPE 和点数，用同一样本答案 CE 优化 10–30 步。先用 32 例，再到 128 例。步长和最大移动半径都以 HD 单元记录，比较 `1/2/4` 单元半径；保存起点与终点的 CE、坐标和答案。

需要专门的可微坐标 override。当前 `_dat_force_locs` 是所有 group/slot 共用的 `[Ns,2]`，且会替换原 learned 坐标路径；不能直接将它当作保留全部分组自由度的坐标优化器。若先做共享坐标诊断，须记录它与原模型逐组坐标的自由度差异。

这是使用答案标签的逐样本优化诊断，不是可部署方法，也不是 held-out 泛化成绩；其 loss 改善不能计入 E5 的模型效果。

| 观察 | 下一步 |
| --- | --- |
| 统一编码后 oracle 也没有稳定优势 | 优先检查 reader、数据可回答性、LR 是否已足够 |
| oracle 有优势，局部坐标优化不改善 | 检查可达范围、边界、数值梯度及局部 loss 地形；先排除路由问题 |
| 坐标优化改善，local head 短训不改善 | 检查输入、条件化、参数化与优化；尚不能只归因于头的容量 |
| 坐标和答案 CE 均改善 | 进入 E4，验证跨样本可学习的映射 |

## 7. E4：固定 400 点的局部头实验

### 7.1 第一版结构约束

- 使用同一冻结 coarse grid，先读 `x0_i` 周围 `5×5`、间距 1 个 HD 特征单元的邻域；保留二维相对坐标。观察范围和移动范围分别记录。
- 主候选输入为投影前 `X_HD`；以相近参数量的 `K_HD` 邻域头做后续对照，写清 K 是 pre-RoPE 还是 post-RoPE。diag12 不能证明 reader 的 K 一定更适合定位。
- 残差以 HD 单元参数化。当前 `align_corners=True`：`delta_x = 2*r_x/(W_HD-1)*tanh(u_x)`，`delta_y = 2*r_y/(H_HD-1)*tanh(u_y)`；维度为 1 时单独处理。第一版 `r_x=r_y=2`。
- 零初始化最后的残差投影，使起点等于粗网格；检查后续更新中上游层是否开始获得梯度，不能把首步上游零梯度直接判为断路。
- 记录从观察邻域可见、从允许移动范围可达的证据覆盖率。大范围粗定位失误单列，不能要求两单元局部头跨图寻找目标。
- 记录 clamp 和 tanh 饱和；无需先增加边界辅助 loss。仅缩小窗口把大量重复点挤进 bbox，不算有效覆盖改善。
- 最终 reader 仍接收 400 个 token；邻域读取有额外成本，应记录吞吐、显存和实际采样量，不能声称总计算量完全相同。

### 7.2 最小实验矩阵

E1 修复与 E2 几何约定先统一。第一轮各学习组只训练 local head，reader/coarse locator 固定；从相同零残差起点开始。下面都是计划实验名，不是已有 CLI。

| 实验 | local 输入/设置 | 用途 |
| --- | --- | --- |
| L0 | `delta=0`，不训练 | 同一 coarse grid 的固定采样基线 |
| L1 | 当前 LR-local 结构，`im_start` query | 现有局部头结构基线 |
| L2 | 与 L1 相同，仅换 `ans_prev` | 隔离问题表示选择 |
| L3 | `X_HD` 的 `5×5` 邻域 + `ans_prev` | 主候选，与 L2 比较局部 HD 证据 |
| L4 | 与 L3 相同容量的 `K_HD` 邻域头 | L3 有收益后再比较定位与读取投影 |

L1–L4 统一最终坐标范围、每组输入维度/参数量预算和训练步数，明确与原 checkpoint 局部头的权重关系。不要让 L1 用已训练残差、L3 用零残差，然后把差距全部解释为输入特征效果。

先用 128 个训练样本检查能否过拟合、梯度和参数更新；这不作为泛化结论。通过后在同一训练子集上短训，建议起点为 600 steps，每 100 steps 记录验证 CE。固定有效 batch size；最后比较按同一验证规则选择的 checkpoint，不能使用测试集挑选步数。

先跑 L0–L3 一个种子，只有出现稳定验证收益才扩为三个种子（建议 0/1/2）和 L4。现有 SFT shell 的 seed 硬编码为 42；重复种子实验需显式修改/参数化传给 trainer 的 `--seed`，只设 `SEED` 环境变量不会生效。

reader 共训放在 sampler-only 之后，使用独立的 2×2 对照：固定/学习采样 × 固定/训练 reader。学习位置相对同样训练 reader 的固定采样仍需有收益，才能区分读出适应与位置选择的贡献。

## 8. E5：反事实验收与记录

在最终 checkpoint 上，固定 reader、图像和原问题，逐样本做以下干预：

| 采样策略 | 控制方式 |
| --- | --- |
| learned local | 正常预测残差 |
| zero local | 仅把 local 残差置零，保持 coarse grid |
| wrong-question local | 同图换问产生残差，reader 仍回答原问题；粗定位先固定 |
| random local | 与 learned 相同幅度约束和预算的随机残差，多次采样取均值 |
| oracle / wrong-window | 相同编码约定、相同点数，记录真实窗口重叠率 |
| wrong-content HD | 保持坐标与位置编码，换入另一图的 HD 内容 |

真实配对问题的证据应有区分度；不同问题若共享同一目标，不能作为“错误问题”负例。随机或错图对照主要诊断内容依赖，负例退化本身不充分证明定位精度提升。

主指标：held-out 答案内容 CE、任务准确率；辅指标：首答案 token CE、GT 框内有效 HD 单元覆盖、重复率、边界率、按层/组的位移与梯度、吞吐/峰值显存。bbox 只用于诊断与 oracle，不进入主实验训练 loss。

同时报告全体和初始 coarse grid/邻域已覆盖证据的子集，子集由干预前状态定义。逐组统计坐标，不能只看 group 均值；单列多目标、细字和边界目标。短答案内容与模板 token 的 CE 分开。

初步通过标准：learned 在验证/测试上优于 zero-local，且正确问题的残差优于 wrong-question 与等幅随机残差；三个种子方向基本一致。对同一样本计算配对差值和置信区间，同图多问按图像聚类重采样。置信区间跨零或增益主要来自模板 token 时，记录为“证据不足”，不扩大长训。这里不预设保证有效的准确率百分点阈值。

现有 probe 不输出上述全部指标，需新增逐样本记录和汇总逻辑。新记录至少包含：样本 ID/图像 ID、实验组、seed、代码 SHA 与本地 diff、checkpoint、完整 dat 配置、最终训练参数清单、数据哈希、精度/backend、LR/HD 实际尺寸、CE 分项、预测答案、group 坐标、运行时间及显存。

结果汇总模板：

| 实验/seed | 代码版本 | reader/locator 是否训练 | 内容 CE ↓ | 首 token CE ↓ | 准确率 ↑ | 有效覆盖 ↑ | 重复/边界率 ↓ | 峰值显存/吞吐 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待运行 | | | | | | | | | |

## 9. 开始长训前的检查表

- [ ] E0 日志、样本顺序、配置与代码版本已保存。
- [ ] E1 的训练 query 为 `[s−1,e)`，question/answer 段不重叠，短答案测试通过。
- [ ] local 与 global query 来源分别确认，未把全局设置当作局部设置。
- [ ] oracle、learned 和随机网格的位置编码约定一致。
- [ ] 纯 CE 配置与实际 trainable/optimizer 参数已核对，reader 冻结但输入梯度保留。
- [ ] 实际 exact merge 分支与坐标有限差分通过，直接坐标优化的结果已解释。
- [ ] local 头读取实际粗位置附近的证据，观察范围与移动范围均以 HD 单元记录。
- [ ] 固定预算对照显示泛化收益，而不只是训练 loss 或框内点数改善。

主要代码入口：[模型与采样](../llava/model/language_model/modeling_qwen3_5_dat.py)、[训练与冻结](../llava/train/train_qwen_dat.py)、[推理 probe/diag12](../scripts/probe_whd_qwen35.py)、[读出 leverage](../scripts/_test_lr_drop_leverage.py)。

## 9. 已实现的 sampler-only 控制与坐标检查

CPU 回归读取生产采样方法，使用冻结 reader 和真实 CE 检查每组 override、梯度、三种有限差分步长、optimizer.step 的冻结边界、共同零残差起点、多图/多答案预算，以及 checkpoint 重算期间的粗网格回放：

```bash
python scripts/test_dat_loading_geometry.py
python scripts/test_dat_experiments.py
```

### 9.1 局部 query 两组短训

`--dat_sampler_only True` 从**完整初始 DAT checkpoint**继承 reader、全局定位、head 容量、DAT 层、HR scale 等配置，避免通用 CLI 的默认 grid/groups 改变已有模型。只切换 local query 和固定 slot 编码、关闭训练 curriculum；局部 readout `conv_off_proj` 清零，其他局部头参数保留初始 checkpoint 数值。原 local 残差被替换为零起点，不叠加旧残差。零 readout 的第一步只更新 readout，上游 head 从随后步骤获得梯度。

下面共用同一初始 checkpoint、seed、数据和顺序，分别训练两组。沿用第 2 节环境变量；若训练集与验证集另行划分，改用真实训练 JSON，不能把 heldout 验证集直接拿来训练。

```bash
export TRAIN_JSON='/absolute/path/to/train.json'
for qpos in im_start ans_prev; do
  torchrun --nproc_per_node=1 llava/train/train_qwen_dat.py \
    --model_name_or_path "$MODEL_PATH" --model_family qwen3_5 \
    --use_dat True --dat_sampler_only True --dat_local_query_pos "$qpos" \
    --data_path "$TRAIN_JSON" --image_folder "$IMAGE_FOLDER" \
    --dat_tf_prob 0 --dat_tf_force_prob 0 \
    --dat_off_penalty 0 --dat_off_sup_weight 0 --dat_rel_sup_weight 0 \
    --dat_lr_drop_prob 0 --dat_hd_lse_bias 0 --dat_hd_lse_bias_decay_steps 0 \
    --hd_content_gap_every 0 --lora_enable False --kd_on False \
    --tune_mm_mlp False --dat_warmup_steps 0 \
    --use_hr_first_resize False --use_decoupled_hr_lr False \
    --lr_min_pixels 28224 --lr_max_pixels 262144 --hd_max_pixels 5017600 \
    --bf16 True --seed 42 --dat_lr 1e-4 --learning_rate 1e-4 \
    --max_steps 100 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --model_max_length 4096 --logging_steps 1 --save_steps 100 \
    --report_to none --output_dir "$RUN_ROOT/local_${qpos}"
done
```

最终可训练参数只有 `conv_lr_dw/ln_1/conv_lr_proj/proj_intention/ln_2/conv_off_proj`；ViT、merger、LLM、k/v_hd、HD norm/gate 和 coarse locator 冻结。冻结 reference 是初始化模型的完整副本，其 local query 固定 `im_start`、local 残差固定零。每个 batch 先在 reference 中取得所有层、样本、图像、reader slot、group 的粗网格，再回放给 student；reference 与 student 共用固定 reader/预处理，后层粗网格不会随训练中的前层局部残差变化。

这条 opt-in 路径每个进程额外保留一份冻结模型并增加一次前向；支持 DDP/ZeRO-2，当前显式拒绝 ZeRO-3。最终冻结在通用 DAT 解冻之后、optimizer 创建之前执行，启动日志打印最终 trainable/optimizer 参数，输出 `dat_sampler_manifest.json` 防止向同一输出目录恢复不同设置。非 sampler-only 路径保持原有配置流程。

### 9.2 真实 checkpoint 的纯 CE 坐标检查

```bash
python scripts/check_dat_coordinate_grad.py \
  --model_path "$MODEL_PATH" \
  --data_json "$RUN_ROOT/eval_subset.json" --image_folder "$IMAGE_FOLDER" \
  --n 16 --seed 0 --local_query_pos ans_prev --residual zero \
  --tok_budget 256 --min_pixels 28224 --hd_cap 5017600 --hr_scale 3 \
  --fd_points 2 --fd_steps 0.01 0.05 0.1 \
  --coordinates_out "$RUN_ROOT/coordinates.json" \
  --out "$RUN_ROOT/coordinate_grad.json"
```

脚本冻结所有模型权重，并关闭 checkpoint 的 TF、offset/relevance 辅助梯度 hook、LR drop、HD bias 和 dropout；只对显式坐标叶张量求导。保留真实 grid_sample 输入的梯度，分别报告 total/content/template/first CE；逐层、reader slot、group 输出范数、非零比例、clamp 前越界、边界比例、raw/effective 残差范围与 tanh 饱和比例。实际 merge 分支必须为 exact，否则报错；`hd_gate` 存在或 exact backward 不可用的 checkpoint/环境需要单独处理，不能把 legacy 梯度当作数值验收。

有限差分按 HD 单元换算扰动，选择远离 clamp 边界和插值折点的坐标；输出每个步长的 analytic/FD、符号、误差和原始 CE 差。bf16 reader 量化可能吞掉小 CE 差，`unresolved_loss_delta` 会标记它，报告不把这种情况自动判为梯度错误或通过。CPU fp32 回归提供采样内容梯度的平滑参考；端到端 GPU 数值与收益仍需在公司机器运行。

检查已训练模型时用 `--model_path "$RUN_ROOT/local_ans_prev" --reference_model_path "$MODEL_PATH" --residual learned`；reference 必须是训练时的初始 checkpoint。脚本逐参数验证 reader/coarse 权重与 reference 相同，允许 local sampler 权重变化。固定 reference 的粗网格、slot 编码和预算后，才比较训练效果。常规 `generate` 不自动加载这个 reference；受控推理须沿用 `sampling_control` 回放，不能混为固定粗网格实验。

### 9.3 完整逐组坐标观察与 override

`--coordinates_out` 保存 `[reader_slots, groups, 20, 20, 2]` 的完整坐标，JSON 用 `sample_index` 和 `[layer,batch_row,image_index]` 定位。修改某一 group 后传入 `--coordinates_in`，保持同一数据、seed 和样本顺序；不平均 group，也不隐式广播。`--residual zero/learned` 控制未覆盖坐标的起点，显式 override 优先。坐标使用 `(x,y)`、`[-1,1]`，之后由真实 clamp/grid_sample 处理。

Python 接口在 `llava/model/dat_experiments.py`：先用 `DATSamplingControl(zero_residual=True)` 对冻结 reference 做 capture，再用 `capture.replay(observe=True, overrides={key: coordinates})` 对 student 执行 `sampling_control`。`records[key]['sampled']` 是实际 grid_sample 输入，保留计算图和 `.grad`；旧可视化返回的 detached 坐标继续保持原语义。使用 gradient checkpointing 时，replay 作用域必须覆盖 backward；训练入口已处理这一生命周期。

每个 reader slot/每张图仍输出 400 个 KV token。8 个 group 是同一 token 的通道分组；image reader 与多个 answer reader 的额外开销分别记账。新的控制与诊断不包含 HD 邻域 refiner、直接坐标优化或验证集收益结论。
