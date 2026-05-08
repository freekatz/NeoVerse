# NeoVerse 蒸馏训练说明

这份文档只覆盖当前仓库里新增的 NeoVerse control latent distillation 流程。原始 NeoVerse 的 app、inference、基础训练代码仍在仓库里，但当前主要工作线是训练一个 student adapter：用 reconstructor/VGGT-like backbone 的中间 tokens 直接预测 NeoVerse control branch 的 early condition embedding。

推荐入口只有一个：

```bash
cd /root/vepfs/diffsynth-dev/papers/neoverse/code
./cli help
```

## 1. 当前推荐流程

如果 frozen cache 已经建好，直接训练：

```bash
./cli train cache
```

如果还没有 cache，按顺序跑：

```bash
./cli cache build-camera
./cli cache build-frozen
./cli train cache
```

训练完评估：

```bash
./cli eval last \
  configs/distill_control_latent.yaml \
  outputs/NeoVerseControlLatentDistill/YYYY-MM-DD/HH-MM-SS/adapter_last.pt
```

看 cache 状态：

```bash
./cli cache inspect
```

## 2. CLI 命令

```text
./cli train cache
```

主训练入口。直接从 `frozen_cache/*.pt` 读取已经离线算好的 tokens、camera/time、teacher condition label。训练时不再在线跑 reconstructor、VAE、control teacher，所以这是当前推荐路径。

```text
./cli train online
```

在线训练，启用 temporal augmentation。每个 sample 会现场从视频取 clip、构造时间轨迹、跑 frozen teacher。主要用于调数据逻辑或没有 frozen cache 时验证，不是大规模训练首选。

```text
./cli train online-noaug
```

在线训练，关闭 temporal augmentation。名字里的 noaug 是重点，不代表一定快；它仍然要在线跑 frozen teacher。

```text
./cli cache build-camera
```

为固定 81 帧 clip 离线重建 camera cache。它只跑 reconstructor 的 camera 相关路径，输出很小。

```text
./cli cache build-frozen
```

为固定 clip 和多条 temporal trajectory 离线构建 frozen cache。它会读取 camera cache，生成训练真正需要的 target/source tokens、camera/time 和 teacher label。这个目录会很大。

```text
./cli cache inspect
```

统计 frozen cache 数量、大小，并按 hash 规则报告 train/eval 切分。

```text
./cli eval last CONFIG CHECKPOINT [dataset indices...]
```

加载某个 adapter checkpoint，跑 teacher/student generation 对比。不传 index 时默认跑 `0 17 33 81 159 240 319`。

```text
./cli debug temporal
./cli debug reconstruction
```

诊断 temporal trajectory 或 81 帧全量重建 vs 稀疏 context 重建。

```text
./cli overfit conv
./cli overfit cross
```

单视频 overfit。`conv` 强制用卷积 adapter，`cross` 用 cross-attention + RoPE adapter。

```text
./cli data prepare-spatialvid-hq
```

整理 SpatialVID-HQ 数据目录。

旧命令名不保留兼容别名。比如 `./cli train fast` 会报错，应使用 `./cli train online-noaug`。

## 3. 训练数据流

当前正确的数据链路是：

```text
原始视频
  -> 连续抽取 81 帧 clip
  -> 用这 81 帧重建 camera trajectory
  -> 从这 81 帧里采样 temporal trajectory, 例如 F/Z/R
  -> target rgb/camera/time 按 temporal trajectory 重排
  -> source rgb/camera/time 从原始连续 81 帧里抽 context
  -> student 输入 source tokens + source camera/time + target camera/time
  -> student 预测 teacher early condition embedding
```

关键点：

- target 可以是 forward/pause/backward/FZR 混合后的时间轨迹。
- source 必须来自最开始连续 81 帧 clip，而不是从已经重排过的 target 序列里再取。
- camera cache 也是针对原始连续 81 帧 clip 建的。
- frozen cache 负责把多条时间轨迹提前算好，训练时直接读。

## 4. 时序设置

`TEMPORAL_ORDER=trajectory` 表示 target 顺序使用采样出的 temporal trajectory。它允许前进、暂停、回退。

`TEMPORAL_TRAJECTORY_PROFILE` 常用值：

```text
forward_pause  偏自然：整体向前，中间可暂停
mixed_fzr      更激进：forward / zero-pause / reverse 混合
backward       整体反向轨迹
```

构建 frozen cache 时默认按权重采多种轨迹：

```bash
TEMPORAL_VARIANT_PROFILE_WEIGHTS=forward_pause:0.5,mixed_fzr:0.4,backward:0.1
```

每个 clip 采多少条轨迹由：

```bash
TRAJECTORIES_PER_CLIP=8
```

控制。更大覆盖更好，但 cache 体积和构建时间也线性增加。

## 5. Cache 目录

默认目录：

```text
outputs/NeoVerseControlLatentDistill/camera_cache
outputs/NeoVerseControlLatentDistill/frozen_cache
```

`camera_cache` 很小，记录固定 clip 的 camera poses/intrinsics/timestamps。

`frozen_cache` 很大，记录训练 forward cache。当前训练推荐直接从它采样：

```bash
./cli train cache
```

train/eval 切分不是离线写死的目录，而是训练时按文件名 hash 稳定切分：

```text
frozen_cache_split_mode: hash
frozen_cache_eval_ratio: 0.05
frozen_cache_split_seed: 0
```

检查切分：

```bash
./cli cache inspect
```

## 6. 常用环境变量

```bash
GPU_LIST=0,1
RUN_NAME=train_cache
MAX_STEPS=200000
DATA_ROOT=data/SpatialVID_full
FROZEN_CACHE_DIR=outputs/NeoVerseControlLatentDistill/frozen_cache
CAMERA_CACHE_DIR=outputs/NeoVerseControlLatentDistill/camera_cache
FIXED_CLIPS_PER_SCENE=16
TRAJECTORIES_PER_CLIP=8
FROZEN_CACHE_EVAL_RATIO=0.05
EVAL_FREQ=500
DRY_RUN=1
```

例子：

```bash
GPU_LIST=0 MAX_STEPS=1 NUM_WORKERS=0 ./cli train cache
```

只打印命令、不真正运行：

```bash
DRY_RUN=1 ./cli train cache
DRY_RUN=1 ./cli cache build-frozen
```

覆盖 OmegaConf 配置：

```bash
./cli train cache \
  learning_rate=5e-5 \
  adapter.num_blocks=2 \
  loss.stat_weight=0.02
```

cache builder 的 argparse 参数直接跟在命令后：

```bash
./cli cache build-frozen --limit 100 --overwrite
```

## 7. Adapter

默认 adapter 是：

```yaml
adapter:
  type: cross_attention_rope
```

切到卷积 adapter：

```bash
ADAPTER_TYPE=conv ./cli train cache
```

单视频 overfit：

```bash
./cli overfit conv
./cli overfit cross
```

## 8. 数据准备

默认训练数据根目录：

```text
data/SpatialVID
```

大规模数据可用：

```bash
DATA_ROOT=data/SpatialVID_full ./cli train cache
```

整理 SpatialVID-HQ：

```bash
SOURCE_ROOT=/root/tos/cmh/datasets/SpatialVID-HQ \
DEST_ROOT=/root/vepfs/diffsynth-dev/papers/neoverse/code/data/SpatialVID_full \
./cli data prepare-spatialvid-hq
```

如果某些压缩包坏了并且允许跳过：

```bash
SKIP_BAD_ARCHIVES=1 ./cli data prepare-spatialvid-hq
```

## 9. 评估和诊断

评估一个 checkpoint：

```bash
./cli eval last \
  configs/distill_control_latent.yaml \
  outputs/NeoVerseControlLatentDistill/YYYY-MM-DD/HH-MM-SS/adapter_last.pt
```

只跑某几个样本：

```bash
./cli eval last configs/distill_control_latent.yaml outputs/.../adapter_last.pt 0 17 33
```

输出会写到 checkpoint 目录下：

```text
eval_idx0/
eval_idx17/
...
```

重建诊断：

```bash
USE_CAMERA_ANNOTATIONS=true DATASET_INDEX=0 NUM_CONTEXT_VIEWS=20 \
./cli debug reconstruction
```

时序轨迹诊断：

```bash
./cli debug temporal
```

## 10. 代码目录约定

```text
cli                                      统一本地入口
configs/distill_control_latent.yaml      蒸馏训练配置
train_distill_control_latent.py          训练主程序

scripts/launch/                          训练和 cache 构建 launcher
scripts/overfit/                         单视频 overfit
scripts/data/                            数据整理

tools/cache/                             camera/frozen cache 工具
tools/eval/                              teacher/student 替换评估
tools/diagnostics/                       temporal 和 reconstruction 诊断
experiments/midterm/                     中期实验脚本
```

## 11. 输出目录清理规则

可以清理：

```text
outputs/NeoVerseControlLatentDistill/*/logs
outputs/NeoVerseControlLatentDistill/*/*smoke*
outputs/NeoVerseControlLatentDistill/frozen_cache_logs
outputs/NeoVerseControlLatentDistill/camera_cache_logs
```

谨慎清理：

```text
outputs/NeoVerseControlLatentDistill/YYYY-MM-DD/HH-MM-SS
```

这些目录可能有 `adapter_last.pt`，删之前先确认是否已经跑完或不再需要。

不要误删：

```text
outputs/NeoVerseControlLatentDistill/frozen_cache
outputs/NeoVerseControlLatentDistill/camera_cache
```

这两个是训练加速的核心缓存。

## 12. 当前主线建议

日常训练：

```bash
./cli train cache
```

调 temporal 逻辑：

```bash
DRY_RUN=1 ./cli train online
MAX_STEPS=1 NUM_WORKERS=0 ./cli train online
```

调 adapter 是否能学：

```bash
MAX_STEPS=2000 ./cli overfit cross
```

大规模训练前先确认：

```bash
./cli cache inspect
DRY_RUN=1 ./cli train cache
```
