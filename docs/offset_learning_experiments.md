# Offset 学习实验计划

日期：2026-09-29。代码基线：`262a26c`（`main`）。本文是实验执行与后续实现清单，不表示实验已经完成；本次提交只新增文档。

目标：固定每张图、每个 DAT 层的 `20×20=400` 个采样 token，让采样位置随问题和图像证据改变，并稳定降低答案损失。暂不做自适应点数，也不增加定位辅助 loss。8 个 offset group 是一个 token 内的分组采样，不应把预算记成 3200 个 reader token；不同 HD reader 分支的实际开销另记。

## 1. 执行顺序与当前状态

| 阶段 | 要回答的问题 | 当前状态 | 进入下一阶段的条件 |
| --- | --- | --- | --- |
| E0 环境与基线 | 当前 checkpoint 能否读取 HD 内容？ | 第 3 节命令可直接运行 | 环境检查通过；记录 real/off/shuffle/oracle 基线及限制 |
| E1 query 路由 | 首个答案 token 是否使用问题条件 HD？ | 已确认存在错位，需先修实现 | 单标签、短答案、多轮、训练与 prefill 对齐测试通过 |
| E2 条件与几何 | local query、采样位置、位置编码是否一致？ | 需增加独立实验路径 | 因素分开控制，oracle 与 learned 使用相同编码约定 |
| E3 CE 梯度 | 答案损失能否给局部移动提供有效方向？ | 需新增诊断 | 坐标梯度数值检查通过；直接优化坐标能获得收益 |
| E4 局部头 | HD 邻域是否比当前 LR 输入更适合预测修正？ | HD 邻域 refiner 尚未实现 | sampler-only 短训在验证集优于零残差与 LR-local |
| E5 因果验收 | 提升是否来自问题相关的位置选择？ | 需新增干预与结果汇总 | 固定预算反事实对照、独立测试集、重复种子结果支持结论 |

先跑 E0 留档；正式训练以 E1–E3 通过为前提。后续新功能没有现成 CLI，本文不会用虚构的启动参数代替实现。

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

将此 Python 块保存为运行目录中的脚本后执行，或直接用 `python - <<'PY'` 在终端执行。多个问题可以来自同一图像；shuffle 的具体配对仍需确认不是同一张图。

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
- probe 按文件顺序取前 N 个样本；输出未完整保存输入标识，因此应和 `input_manifest.json` 一起保存。
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

## 4. E1：先修答案 query 路由

实现位置：[modeling_qwen3_5_dat.py](../llava/model/language_model/modeling_qwen3_5_dat.py) 的 `compute_image_range_list` 和 `Qwen3_5AttentionDAT.forward`。

当前答案标签的端点 `[s,e]` 都包含在监督内，但问题条件 HD 的训练 query 为 `[s,e)`。next-token CE 对应的 query 应为 `[s−1,e)`。需要同时调整 question-HD 的终点，保持各 segment 不重叠，满足 exact merge 的前提。不要只把一个起点减一。

必须验证：

1. 单个有效标签 `s=e` 时仍有一个答案 query；普通短答案、结束符与换行的监督位置逐 token 打印。
2. 多轮对话的每个答案段均正确；question-HD 开/关都检查；padding 和截断不产生负索引或越界。
3. 同一完整 prompt 下，训练预测首答案 token 的位置与推理 prefill 使用同一套问题条件 HD。
4. 只对首个答案内容 token 的 CE backward，确认其对应坐标有计算图；零梯度需区分 reader 没有利用 HD 与路由断开。

此处先做实现正确性修复，不与新 head、LoRA 或新 loss 混在一个性能实验里。修复前基线只用于定位问题，正式消融共用修复后实现。

## 5. E2：分别核对问题条件、局部证据与位置编码

### 5.1 local query 与 global query 独立控制

当前 `glob_query_pos=ans_prev` 只影响全局定位。局部 gate 的 `intention_indices` 仍取 `ar[2]`，即 assistant 的 `<|im_start|>`；不能由全局 query 设置推断局部头也已改用 prompt-end。

增加只控制局部头 query 来源的实验路径，对比 `im_start` 与 `ans_prev=s−1`。保持同一 coarse grid、reader、局部头容量、初始化、样本和步数。该对照要在训练与推理一致的条件下跑；对旧 checkpoint 直接换 query 的结果只作即时干预诊断，不代表重训后的能力上限。

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

以上编码对照当前没有 CLI；需实现后才可声称完成。可在此阶段另做 center-only、scale-only 干预，区分质心误差与窗口密度误差，避免只看一个 `in_box` 数字。

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

### 6.2 逐点梯度与局部数值验证（需新增诊断）

先用 16–32 个有明确答案证据的验证样本，保留最终采样坐标的梯度。记录每层/每组：坐标梯度范数、非零比例、clamp 前越界比例、边界点比例、残差范围、tanh 饱和比例，以及首答案 token 的独立 CE 梯度。

在少量远离边界与插值单元折点的坐标上，用中心有限差分检查 `dL/dx` 的符号与量级。扰动用 HD 单元定义，例如 `0.01/0.05/0.1` 个单元；检查多个步长，避免 bf16 舍入让微小 loss 差淹没。固定随机性和样本；在可行范围采用 fp32 的小规模参考计算。

再沿归一化负梯度移动少量坐标，与相同步长的正方向、随机方向比较 CE。只在光滑局部和数值精度允许的范围解读，不要求每个样本每一步都下降；观察配对均值、符号一致性与误差。

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
