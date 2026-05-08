# NeoVerse Control Latent Distillation 使用说明

这份文档只覆盖本仓库在原始 NeoVerse 之外新增的蒸馏训练流程，核心入口是：

```text
train_distill_control_latent.py
```

原始 NeoVerse 的 `app.py`、`inference.py`、`train.py` 用法请看 `README.md` 和 `USAGE_CN.md`。这里的目标不是复现原始 NeoVerse 训练，而是训练一个轻量 Student Adapter：让它从 reconstructor / VGGT-like backbone 的中间 tokens 直接预测 NeoVerse control branch 的 early condition embedding，减少对 4DGS degraded render teacher path 的依赖。

## 1. 这条线新增了什么

主要新增文件如下：

```text
configs/distill_control_latent.yaml
train_distill_control_latent.py
cli
tools/eval/replace_teacher_with_student.py
tools/diagnostics/compare_reconstruction_context.py
README_distill_adapter.md
scripts/launch/train_distill.sh
scripts/overfit/conv_gpu0.sh
scripts/overfit/cross_gpu1.sh
scripts/launch/compare_reconstruction_context.sh
hooks/extract_vggt_tokens.py
diffsynth/models/student_adapters.py
```

各文件职责：

| 文件 | 作用 |
| --- | --- |
| `train_distill_control_latent.py` | 蒸馏训练主入口。冻结 NeoVerse reconstructor、VAE、control branch，只训练 student adapter。 |
| `configs/distill_control_latent.yaml` | 默认蒸馏配置。控制数据、context 帧数、adapter 结构、loss、缓存、resume 等。 |
| `cli` | 推荐本地入口，统一训练、cache 构建、评估、诊断和 overfit 命令。 |
| `scripts/launch/train_distill.sh` | CLI 调用的训练 launcher。封装环境变量、accelerate、多 GPU、日志、自动 resume、frozen cache。 |
| `hooks/extract_vggt_tokens.py` | 从 reconstructor 中提取中间层 tokens，默认取 `[4, 11, 17, 23]` 层，并可追加 camera token。 |
| `diffsynth/models/student_adapters.py` | Student Adapter 实现。默认使用 `cross_attention_rope`，也保留 `conv` 对照。 |
| `tools/eval/replace_teacher_with_student.py` | 用训练好的 student 替换 teacher condition，跑 teacher / student 视频生成对比。 |
| `tools/diagnostics/compare_reconstruction_context.py` | 不跑 diffusion，只对比“81 帧全量输入重建”和“稀疏 context 输入重建”的 81 帧渲染视频。 |
| `README_distill_adapter.md` | 更偏实现细节的旧说明。本文档是更完整的上手手册。 |

## 2. 一句话流程

主训练：

```bash
cd /root/vepfs/diffsynth-dev/papers/neoverse/code

./cli train cache
```

训练后评估：

```bash
./cli eval last \
  configs/distill_control_latent.yaml \
  outputs/NeoVerseControlLatentDistill/YYYY-MM-DD/HH-MM-SS/adapter_last.pt
```

诊断用：看 81 帧全量重建 vs 20 帧 context 重建。

```bash
USE_CAMERA_ANNOTATIONS=true DATASET_INDEX=0 NUM_CONTEXT_VIEWS=20 \
  bash scripts/launch/compare_reconstruction_context.sh
```

## 3. 环境和模型

默认环境路径是：

```text
/root/vepfs/envs/neoverse
```

如果你的环境不在这个位置，可以覆盖：

```bash
VENV_PATH=/path/to/env ./cli train cache
```

模型默认放在：

```text
models/NeoVerse/
├── reconstructor.ckpt
├── Wan2.1_VAE.pth
├── diffusion_pytorch_model-00001-of-00006.safetensors
├── ...
├── diffusion_pytorch_model-00006-of-00006.safetensors
├── diffusion_pytorch_model.safetensors.index.json
├── models_t5_umt5-xxl-enc-bf16.pth
├── google/
└── loras/
    └── Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors
```

训练 `train_distill_control_latent.py` 默认不加载 Wan DiT 和 text encoder：

```yaml
load_dit: false
load_text_encoder: false
load_vae: true
```

原因是训练目标是 early condition embedding，不需要完整 diffusion denoising。真正做 `tools/eval/replace_teacher_with_student.py` 生成视频时才需要加载完整 NeoVerse diffusion 模型。

## 4. 数据格式

默认数据集是 `SpatialVID`：

```text
data/SpatialVID/
├── data/train/SpatialVID_HQ_metadata.csv
└── SpatialVid/HQ/
    ├── videos/...
    └── annotations/...
```

默认配置：

```yaml
height: 336
width: 560
num_views: 81
min_num_context_views: 10
max_num_context_views: 20
continuous_target_frames: true
force_first_context: true
timestamp_unit: seconds
```

也就是说，每个训练 sample 采 81 帧；其中 10 到 20 帧作为 context/source views，其余作为 target views。数据集返回顺序是：

```text
context views 在前，target views 在后
```

每个 view 至少需要这些字段：

```text
img
prompt
timestamp
is_target
dataset
video_name
image_name
```

如果存在相机标注，还会有：

```text
camera_poses
camera_intrs
```

### 关于 81 帧、重建轨迹和 GT camera annotation

这是最容易误解的地方。

主训练默认使用：

```yaml
use_camera_annotations: false
```

这时 pipeline 没有外部 GT camera，`WanVideoUnit_4DPreprocesser` 会沿用原 NeoVerse 路径，用 reconstructor 的输出作为 target trajectory：

```text
render_extrinsics = recon_output["rendered_extrinsics"]
render_intrinsics = recon_output["rendered_intrinsics"]
render_timestamps = recon_output["rendered_timestamps"]
```

也就是说，teacher 的 target RGB / depth / mask 是从 reconstructor 重建出的 4DGS 和它估计出的相机轨迹渲染出来的。这是更贴近原论文和原仓库训练流程的主设置。

student 的 tokens 默认仍然是 context-only：

```yaml
token:
  context_only: true
```

如果打开：

```bash
USE_CAMERA_ANNOTATIONS=true
```

则会变成一个诊断/对照设置：

1. reconstructor 只看 context/source views；
2. target 81 帧相机轨迹来自 `annotations/*/poses.npy` 和 `intrinsics.npy`；
3. student tokens 仍然只来自 context/source views；
4. teacher / student 都渲染或预测到同一条 81 帧 target trajectory。

这个设置适合检查“只给 context 输入时重建质量如何”或者做无 target-frame 视觉泄露的 sanity check，但不应该替代主训练路径，除非实验设计明确要使用外部 GT camera。

## 5. 训练目标

原 NeoVerse 的 control path 会从 degraded RGB / depth / mask / camera map 编码出 control condition，再进入 frozen control blocks 给 Wan DiT 提供 hints。

我们新增的蒸馏线训练一个 student adapter，直接从 reconstructor 中间 tokens 预测：

```text
control_branch.control_patch_embedding(...) 之后的 early condition embedding
```

默认蒸馏目标是：

```yaml
adapter:
  output_mode: condition_embedding

loss:
  l1_weight: 1.0
  stat_weight: 0.01
```

训练时冻结：

```text
reconstructor
VAE
control_branch
Wan DiT
text encoder
```

只更新：

```text
student adapter
```

默认 adapter 是 cross-attention + RoPE：

```yaml
adapter:
  type: cross_attention_rope
  hidden_dim: 512
  num_heads: 8
  num_blocks: 4
```

默认 token 设置适配当前 `models/NeoVerse/reconstructor.ckpt`：

```yaml
token:
  layer_indices: [4, 11, 17, 23]
  include_camera_token: auto
  token_dim: 2048
  token_groups: 5
```

如果换了 reconstructor 或 backbone，第一件事就是检查 `token_dim` 和 `token_groups` 是否还对。

## 6. 推荐训练方式

直接用这一条：

```bash
cd /root/vepfs/diffsynth-dev/papers/neoverse/code

./cli train cache
```

默认推荐从 frozen cache 训练：直接枚举 `outputs/NeoVerseControlLatentDistill/frozen_cache/*.pt`，并按稳定 hash 切 train/eval。在线训练用 `./cli train online`。

输出目录：

```text
outputs/NeoVerseControlLatentDistill/YYYY-MM-DD/HH-MM-SS/
```

如果只是检查命令，不启动训练：

```bash
DRY_RUN=1 ./cli train cache
```

如果只是做 1 step smoke test：

```bash
MAX_STEPS=1 NUM_WORKERS=0 ./cli train cache
```

指定 GPU 时加 `GPU_LIST=0` 或 `GPU_LIST=0,1`。后台长跑加 `LAUNCH_MODE=background`。

## 7. 提升训练效率：Frozen Cache

这条训练线最慢的部分不是 adapter，而是每个 step 都要跑 frozen reconstructor、VAE、control branch 来构造 teacher label 和 tokens。

cache 训练入口等价于打开：

```yaml
frozen_cache_read: true
train_from_frozen_cache: true
frozen_cache_write: false
preload_frozen_cache: false
preload_frozen_cache_device: cpu
preload_frozen_cache_batch_size: 1
save_optimizer_intermediate: false
```

默认 cache 目录：

```text
outputs/NeoVerseControlLatentDistill/frozen_cache/
```

如果要放到更大的盘：

```bash
FROZEN_CACHE_DIR=/path/to/frozen_cache ./cli train cache
```

### 不建议一上来全塞 GPU

`preload_frozen_cache_device` 支持 `cpu` 或 `cuda`。默认是 `cpu`，这是更稳的选择。

不建议默认设成 `cuda`，原因：

1. 每个 rank 的 frozen cache 可能很大；
2. adapter 本身很小，把 cache 常驻 GPU 会快速吃光显存；
3. CPU preload 已经能避免反复跑 frozen teacher，通常已经能显著提高 GPU 利用率；
4. 真要测试 GPU preload，应先用很小数据或单视频 overfit 验证显存。

可以在小实验中尝试：

```bash
PRELOAD_FROZEN_CACHE_DEVICE=cuda \
MAX_STEPS=100 \
./cli train cache
```

但正式长跑建议保持：

```bash
PRELOAD_FROZEN_CACHE_DEVICE=cpu
```

## 8. Resume 和 checkpoint

默认：

```yaml
auto_resume: true
resume_optimizer: true
```

如果 `OUTPUT_PATH` 指向已有 run，且里面有 `adapter_last.pt`，脚本会自动恢复。旧 run 里只有 `adapter_step_*.pt` 时也兼容恢复：

```bash
OUTPUT_PATH=outputs/NeoVerseControlLatentDistill/2026-04-30/12-00-00 \
./cli train cache
```

手动指定 checkpoint：

```bash
RESUME_FROM=outputs/.../adapter_last.pt \
GPU_LIST=0 \
./cli train cache
```

输出文件：

| 文件 | 说明 |
| --- | --- |
| `adapter_last.pt` | 最终或当前最新 adapter checkpoint，通常包含 optimizer。 |
| `config.yaml` | 解析后的完整配置。 |
| `visuals/*.png` | teacher / student / abs error heatmap。 |
| `events.out.tfevents.*` | TensorBoard scalar。 |

看 TensorBoard：

```bash
tensorboard --logdir outputs/NeoVerseControlLatentDistill
```

重点指标：

```text
loss
condition/l1
condition/stat
time/source_frames
time/target_frames
```

## 9. Overfit 调试

如果想快速判断 adapter 是否能学到东西，先用单视频 overfit。

卷积 adapter，默认 GPU 0：

```bash
MAX_STEPS=20000 ./cli overfit conv
```

Cross-attention + RoPE adapter，默认 GPU 1：

```bash
MAX_STEPS=20000 ./cli overfit cross
```

常用覆盖：

```bash
VIDEO_NAME=c93dd173-51dd-54f9-bead-b835a485db24.mp4 \
NUM_CONTEXT_VIEWS=16 \
MAX_STEPS=2000 \
./cli overfit conv
```

Overfit 脚本会打开：

```text
cache_train_batch=true
cache_frozen_outputs=true
num_workers=0
```

## 10. 替换评估：teacher / student

训练完后，用 `tools/eval/replace_teacher_with_student.py` 看 student 是否能替代 teacher condition。

```bash
/root/vepfs/envs/neoverse/bin/python tools/eval/replace_teacher_with_student.py \
  configs/distill_control_latent.yaml \
  outputs/NeoVerseControlLatentDistill/YYYY-MM-DD/HH-MM-SS/adapter_last.pt \
  --output_dir outputs/distill_eval/run0 \
  --dataset_index 0 \
  --modes teacher,student
```

输出：

```text
outputs/distill_eval/run0/
├── teacher.mp4
├── student.mp4
├── input_source_views.mp4
├── input_context_views.mp4
├── input_target_views_gt.mp4
├── rendered_degraded_rgb.mp4
├── rendered_degraded_depth.mp4
├── rendered_degraded_mask.mp4
├── target_plucker_dir.mp4
└── target_plucker_moment.mp4
```

模式含义：

| 模式 | 说明 |
| --- | --- |
| `teacher` | 原 NeoVerse degraded render condition path。 |
| `student` | 用 student 预测的 early condition embedding 替换 teacher condition。 |

默认评估逻辑：

1. 如果存在官方 LightX2V LoRA，则用 4-step 生成，`cfg_scale=1.0`；
2. 如果加 `--disable_lora`，则用 50-step full inference，`cfg_scale=5.0`；
3. 可以手动传 `--num_inference_steps` 和 `--cfg_scale` 覆盖。

示例：

```bash
/root/vepfs/envs/neoverse/bin/python tools/eval/replace_teacher_with_student.py \
  configs/distill_control_latent.yaml \
  outputs/.../adapter_last.pt \
  --output_dir outputs/distill_eval/fast \
  --dataset_index 0 \
  --modes teacher,student \
  --num_inference_steps 4 \
  --cfg_scale 1.0 \
  --enable_vram_management
```

## 11. 81 帧全量重建 vs 稀疏 context 重建对比

这个脚本不跑 diffusion，只回答一个问题：

```text
同一条 81 帧相机轨迹下，
用全部 81 帧输入重建出来的 render，
和只用 20 帧 context 输入重建出来的 render，
差别到底有多大？
```

推荐命令：

```bash
USE_CAMERA_ANNOTATIONS=true \
DATASET_INDEX=0 \
NUM_CONTEXT_VIEWS=20 \
bash scripts/launch/compare_reconstruction_context.sh
```

输出目录默认：

```text
outputs/reconstruction_context_compare/YYYYMMDD_HHMMSS/
```

主要文件：

| 文件 | 说明 |
| --- | --- |
| `index.html` | 浏览器打开，一页看所有视频。 |
| `side_by_side_all81_vs_context.mp4` | 左右对比：全 81 输入 vs context 输入。 |
| `render_all81.mp4` | 81 帧全量输入的重建渲染。 |
| `render_context.mp4` | 只用 context 输入、渲染到完整 81 帧 target timestamps；中间时间会走 4DGS transition。 |
| `render_context_nointerp.mp4` | 只用 context 输入、只渲染有对应 context GS 的 timestamp；不做时间插值。 |
| `absdiff_all81_vs_context.mp4` | 两者 RGB 绝对差异。 |
| `gt_81_sorted.mp4` | 数据集中采样出的 81 帧 GT。 |
| `context_input.mp4` | 实际喂给 sparse reconstruction 的 context 帧。 |
| `depth_all81.mp4` / `depth_context.mp4` / `depth_context_nointerp.mp4` | 深度对比。 |
| `alpha_all81.mp4` / `alpha_context.mp4` / `alpha_context_nointerp.mp4` | alpha 覆盖对比。 |
| `metadata.json` | dataset index、context indices、trajectory reference 等。 |

指定输出目录：

```bash
USE_CAMERA_ANNOTATIONS=true \
DATASET_INDEX=0 \
NUM_CONTEXT_VIEWS=20 \
OUTPUT_DIR=outputs/reconstruction_context_compare/run_dataset0_ctx20_gtcam \
bash scripts/launch/compare_reconstruction_context.sh
```

如果你要检查 claim，请优先看：

```text
side_by_side_all81_vs_context.mp4
absdiff_all81_vs_context.mp4
metadata.json
```

`metadata.json` 里应有：

```json
"trajectory_reference": "gt_camera_annotations"
```

这说明两路结果渲染到了同一条 GT 81 帧相机轨迹。

## 12. 常用参数速查

### 训练脚本环境变量

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `GPU_LIST` | 空 | 指定可见 GPU，如 `0` 或 `0,1`。 |
| `RUN_NAME` | `train` | 日志名。 |
| `OUTPUT_PATH` | 自动日期目录 | run 输出目录。 |
| `LAUNCH_MODE` | `foreground` | 可设为 `background`。 |
| `MAX_STEPS` | `20000` | 最大训练 step。 |
| `NUM_EPOCHS` | 同 `MAX_STEPS` | epoch 上限。 |
| `USE_CAMERA_ANNOTATIONS` | 配置默认 `false` | 主流程保持 `false`，只在 GT camera 诊断/对照实验中设为 `true`。 |
| `NUM_VIEWS` | `81` | 每个 sample 的总帧数。 |
| `MIN_NUM_CONTEXT_VIEWS` | `10` | 最少 context 帧。 |
| `MAX_NUM_CONTEXT_VIEWS` | `20` | 最多 context 帧。 |
| `LEARNING_RATE` | `1e-4` | adapter 学习率。 |
| `BATCH_SIZE` | `1` | 每进程 batch size。当前训练主要按 batch 1 设计。 |
| `NUM_WORKERS` | `1` | DataLoader workers。 |
| `PRINT_FREQ` | `10` | 打印间隔。 |
| `SAVE_FREQ` | `500` | 覆盖保存 `adapter_last.pt` 的间隔。 |
| `VIS_FREQ` | `500` | heatmap 可视化间隔。 |
| `FROZEN_CACHE_DIR` | `outputs/.../frozen_cache` | frozen cache 目录。 |
| `TRAIN_FROM_FROZEN_CACHE` | `true` via `./cli train cache` | 直接从 `.pt` cache 文件采样训练。 |

### OmegaConf 覆盖

除了环境变量，也可以在命令最后直接写 dotlist：

```bash
GPU_LIST=0 ./cli train cache \
  adapter.hidden_dim=768 \
  adapter.num_res_blocks=6 \
  loss.stat_weight=0.02 \
  pipeline_kwargs.mask_non_context_targets=false
```

## 13. 切换 adapter

默认是 cross-attention + RoPE adapter：

```yaml
adapter:
  type: cross_attention_rope
```

切到卷积 adapter：

```bash
ADAPTER_TYPE=conv ./cli train cache
```

调整 cross-attention + RoPE：

```bash
GPU_LIST=0 ./cli train cache \
  adapter.type=cross_attention_rope \
  adapter.token_dim=null \
  adapter.num_heads=8 \
  adapter.num_blocks=2 \
  adapter.source_pool_hw=[16,16] \
  adapter.max_source_tokens=32768 \
  adapter.query_chunk_size=4096 \
  adapter.use_rope=true \
  adapter.use_dit_state=false
```

或者直接用：

```bash
MAX_STEPS=2000 ./cli overfit cross
```

显存不够时优先降：

```text
adapter.source_pool_hw
adapter.max_source_tokens
adapter.query_chunk_size
adapter.hidden_dim
adapter.num_blocks
```

## 14. 训练是否正常的判断

先看日志中是否有 shape：

```text
token_shape=(B, N, L, h, w, C)
output_grid=(T, H, W)
teacher[condition]=(B, seq, 5120)
student[condition]=(B, seq, 5120)
```

当前默认 reconstructor 预期：

```text
C = 2048
L = 5
```

再看 loss：

```text
loss
condition/l1
condition/stat
```

Overfit 单视频时，`condition/l1` 应该能明显下降。如果 overfit 都不降，优先检查：

1. `token_dim` / `token_groups` 是否匹配；
2. `USE_CAMERA_ANNOTATIONS` 和数据 annotation 是否存在；
3. context 帧数量是否过少；
4. frozen cache 是否混用了不同配置；
5. learning rate 是否太大或太小。

## 15. 常见问题

### Q1：训练时是不是随机选 81 帧？

默认 `continuous_target_frames=true` 时，是从视频里随机截取连续 81 帧。context 帧在这 81 帧内部按 `linspace` 选出，默认 10 到 20 帧，并强制包含第 0 帧。

如果关掉 `continuous_target_frames`，则会按 `min_interval/max_interval` 从更长范围里采样。

### Q2：context 20 帧和 target 81 帧是什么关系？

81 是总 target trajectory 长度。context 是这 81 帧中的稀疏观测帧。数据返回时 context 在前，target 在后，但 timestamp 会保留原始时间位置。

例如 20 个 context 时，常见 index 类似：

```text
0, 4, 8, 12, 16, 21, ..., 80
```

### Q3：什么时候应该用 `USE_CAMERA_ANNOTATIONS=true`？

它不是主训练推荐项。主训练应该保持默认 `false`，使用 reconstructor 估计出来的 target trajectory，和原 NeoVerse 流程更一致。

`USE_CAMERA_ANNOTATIONS=true` 适合做诊断/对照：只给 reconstructor context 帧，然后用数据集 GT camera 渲染 81 帧目标轨迹，检查 sparse context reconstruction 本身的上限和失真。

### Q4：GPU 利用率低怎么办？

优先用 `./cli train cache`，它会直接从 frozen cache 文件采样，避免训练时重跑 frozen teacher。先用 `./cli cache inspect` 看 cache 数量、大小和 train/eval 切分。

### Q5：最新 checkpoint 没有 optimizer，能 resume 吗？

可以。脚本会加载 adapter；如果 checkpoint 里有 optimizer 且 `resume_optimizer=true`，就恢复 optimizer。周期保存的 `adapter_last.pt` 是否包含 optimizer 由 `SAVE_OPTIMIZER_INTERMEDIATE` 控制，最终保存会包含 optimizer。

### Q6：评估为什么比训练更吃显存？

训练 early condition embedding 时默认不加载 DiT/text encoder。`tools/eval/replace_teacher_with_student.py` 要真正跑 generation，所以会加载完整 Wan DiT、text encoder、VAE、control branch 和 reconstructor。显存不够就加：

```bash
--enable_vram_management
```

### Q7：我只想看重建差异，不想跑 diffusion。

用：

```bash
USE_CAMERA_ANNOTATIONS=true DATASET_INDEX=0 NUM_CONTEXT_VIEWS=20 \
  bash scripts/launch/compare_reconstruction_context.sh
```

然后打开输出目录下的 `index.html`。

## 16. 建议的实验记录方式

每次正式实验至少记录：

```text
git commit / diff
OUTPUT_PATH
config.yaml
run_config.env
adapter_last.pt
TensorBoard 曲线
tools/eval/replace_teacher_with_student.py 输出视频
tools/diagnostics/compare_reconstruction_context.py 输出视频和 metadata.json
```

如果要写论文或报告，建议至少做这些对比：

1. 原 teacher path vs student path；
2. context 帧数：10 / 16 / 20；
3. `USE_CAMERA_ANNOTATIONS=false` 主流程和 `true` 诊断流程的差异；
4. 81 帧全量重建 vs 20 帧 context 重建可视化；
5. conv adapter vs cross-attention RoPE adapter；
6. frozen cache 只作为训练加速，不作为方法贡献。

## 17. 训练后检查

训练后评估：

```bash
/root/vepfs/envs/neoverse/bin/python tools/eval/replace_teacher_with_student.py \
  configs/distill_control_latent.yaml \
  outputs/NeoVerseControlLatentDistill/YYYY-MM-DD/HH-MM-SS/adapter_last.pt \
  --output_dir outputs/distill_eval/exp_name_dataset0 \
  --dataset_index 0 \
  --modes teacher,student \
  --enable_vram_management
```

重建对比：

```bash
USE_CAMERA_ANNOTATIONS=true \
DATASET_INDEX=0 \
NUM_CONTEXT_VIEWS=20 \
OUTPUT_DIR=outputs/reconstruction_context_compare/exp_name_dataset0_ctx20 \
bash scripts/launch/compare_reconstruction_context.sh
```
