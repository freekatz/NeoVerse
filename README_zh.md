# NeoVerse 精简仓库说明（训练 + 推理 + 可视化评估）

本仓库已裁剪为最小功能子集：**数据样例、模型占位、蒸馏训练（cache / online）、Legacy 全量微调入口、推理 CLI/Gradio、checkpoint 可视化对比评估**。`diffsynth/` 未改动。

统一入口：

```bash
cd /path/to/NeoVerse
./cli help
```

## 蒸馏训练（主路径：`train cache`）

已有 frozen cache 时直接训练：

```bash
./cli train cache
```

没有 cache 时先离线构建再训练：

```bash
./cli cache build-camera
./cli cache build-frozen
./cli train cache
```

在线训练（从视频在线抽 clip，用于调试或 sanity check）：

```bash
./cli train online
./cli train online-noaug
```

### 训练逻辑概要

- `train cache`：只读 `frozen_cache/*.pt`，不在线抽 clip、不在线跑完整 teacher 链路；训练 student adapter。
- 数据链路：`原始视频 → 连续 81 帧 clip → camera cache →（多条 temporal trajectory）→ frozen cache → adapter 训练`
- 约束：target 可按 forward/pause/backward/FZR 重排；source 须来自原始连续 81 帧 clip；camera normalization 在 build frozen cache 阶段完成。

### 评估 checkpoint（可视化）

```bash
./cli eval last configs/distill/control_latent.yaml outputs/NeoVerseControlLatentDistill/YYYY-MM-DD/HH-MM-SS/adapter_last.pt
```

## Legacy 训练（公开 README 路径）

在自有数据上微调 control branch（配置见 `training/configs/train.yaml`）：

```bash
./cli legacy train-base training/configs/train.yaml
```

## 推理与演示

详见根目录 [README.md](README.md)：

- `./cli infer neoverse ...`
- `./cli app neoverse`

## 数据准备

```bash
./cli dev data prepare-spatialvid-hq
```

## 关键目录

```text
configs/distill/control_latent.yaml       蒸馏主配置
training/config_utils.py                  共享配置解析 / SpatialVID 构建
training/engine.py                        Legacy 训练引擎
training/data/spatialvid.py               主数据集
training/control_latent/main.py           蒸馏训练入口
training/control_latent/module.py         ControlLatentDistillModule
training/control_latent/cache.py          frozen cache 读写 / 切分
training/control_latent/camera.py         相机 / 时间 / token 辅助
training/control_latent/loss.py           distill loss 与可视化
training/configs/train.yaml               Legacy 训练配置
tools/cache/                              cache 构建与检查
tools/inference/                          推理脚本
tools/apps/neoverse_app.py                Gradio 演示
tools/eval/render_student_comparison.py   可视化评估（eval last）
scripts/launch/                           训练与 cache 启动脚本
scripts/data/prepare_spatialvid_hq.sh     数据准备

outputs/NeoVerseControlLatentDistill/camera_cache
outputs/NeoVerseControlLatentDistill/frozen_cache
outputs/NeoVerseControlLatentDistill/YYYY-MM-DD/HH-MM-SS   # 训练输出 / checkpoint
```

## 常用环境变量

```bash
DATA_ROOT=data/SpatialVID
FROZEN_CACHE_DIR=outputs/NeoVerseControlLatentDistill/frozen_cache
CAMERA_CACHE_DIR=outputs/NeoVerseControlLatentDistill/camera_cache
MAX_STEPS=200000
FIXED_CLIPS_PER_SCENE=16
TRAJECTORIES_PER_CLIP=8
FROZEN_CACHE_EVAL_RATIO=0.05
EVAL_FREQ=500
SWANLAB_MODE=cloud        # 需要 SWANLAB_API_KEY；本地可改 offline
SWANLAB_PROJECT=NeoVerseControlLatentDistill
```

示例：

```bash
MAX_STEPS=1000 ./cli train cache
GPU_LIST=0 MAX_STEPS=1 NUM_WORKERS=0 ./cli train cache
./cli cache build-frozen --limit 100 --overwrite
```
