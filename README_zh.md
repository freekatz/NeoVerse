# NeoVerse QWM v2 训练

统一入口：

```bash
cd /root/vepfs/diffsynth-dev/papers/neoverse/code
./cli help
```

## 推荐流程

已有 v2 cache，直接训练：

```bash
./cli train cache
```

没有 cache，先离线构建：

```bash
./cli cache build-camera
./cli cache build-frozen
./cli train cache
```

评估 checkpoint：

```bash
./cli eval last configs/distill/control_latent_v2.yaml outputs/NeoVerseQueryableWorldModel/YYYY-MM-DD/HH-MM-SS/query_module_last.pt
```

## 训练逻辑

当前主线是 QWM v2 的 `train cache`：

- 训练只读 `frozen_cache/v2_*.pt`
- 不在线抽视频 clip
- 不在线跑 reconstructor/VGGT
- 不在线算 teacher condition
- 只训练 QueryModule

正确数据链路：

```text
原始视频 -> 连续 81 帧 clip -> camera cache
81 帧 clip -> 多条 temporal trajectory -> frozen cache
frozen cache -> QueryModule + Wan 联合前向训练
```

关键约束：

- target 可以按 forward/pause/backward/FZR 重排。
- source 必须从原始连续 81 帧 clip 里取，不能从重排后的 target 里取。
- camera normalization 在 build frozen cache 时完成；`train cache` 不会重新归一化旧 cache。

## 常用命令

```bash
# 主路径
./cli train cache                 # 主训练，从 v2 frozen cache 读

./cli cache build-camera          # 建 camera cache
./cli cache build-frozen          # 建 frozen cache
./cli cache inspect               # 看 cache 数量、大小、train/eval 切分

./cli eval last CONFIG CHECKPOINT # 评估 checkpoint

# 开发/诊断
./cli train online                # 在线时序增强训练，调试用
./cli train online-noaug          # 在线无时序增强，sanity check
./cli dev profile distill-step    # 单 step profile
./cli dev debug temporal          # 时序轨迹诊断
./cli dev debug reconstruction    # 重建诊断
./cli dev data prepare-spatialvid-hq

# Legacy
./cli legacy train-base training/configs/train.yaml
```

## 火山任务

先导入密钥到本机 MLP CLI 配置，密钥不会写入仓库：

```bash
./cli volc auth import ../AccessKey.txt
```

非密钥资源参数放这里：

```text
deploy/volc/env.local
```

火山上建 cache：

```bash
./cli volc submit build-camera --replicas 1 --name neoverse-build-camera --submit
./cli volc submit build-frozen --replicas 1 --name neoverse-build-frozen --submit
```

火山上训练：

```bash
./cli volc submit train-cache --replicas 1 --name neoverse-train-cache-1w --submit
```

只生成任务 YAML，不提交：

```bash
DRY_RUN=1 ./cli volc submit train-cache --replicas 1 --name neoverse-train-cache-1w
```

说明：

- `--replicas 1`：1 个 worker 实例，也就是 1 台 8 卡机器。
- `--name ...`：火山平台任务名。
- `--submit`：真正提交；不加只生成 YAML。
- `build-camera`：离线建 camera cache。
- `build-frozen`：读取 camera cache，离线建 v2 frozen cache。
- `train-cache`：读取 v2 frozen cache 训练 QueryModule，不会自动补建 cache。

## 关键目录

```text
configs/distill/control_latent_v2.yaml                   主配置
training/control_latent/                                 QWM / 蒸馏训练代码
tools/cache/                                             cache 构建与检查
tools/volc/                                              火山提交工具
scripts/launch/                                          launcher

data/camera_cache                                        camera cache，别误删
data/frozen_cache                                        v2 frozen cache，别误删
outputs/NeoVerseQueryableWorldModel/YYYY-MM-DD/HH-MM-SS 训练输出/checkpoint
```

## 常用变量

```bash
DATA_ROOT=data/SpatialVID_full
FROZEN_CACHE_DIR=data/frozen_cache
CAMERA_CACHE_DIR=data/camera_cache
MAX_STEPS=200000
FIXED_CLIPS_PER_SCENE=16
TRAJECTORIES_PER_CLIP=8
FROZEN_CACHE_EVAL_RATIO=0.05
EVAL_FREQ=500
SWANLAB_MODE=cloud        # 需要 SWANLAB_API_KEY；本地离线可改 offline
SWANLAB_PROJECT=NeoVerseQueryableWorldModel
```

例子：

```bash
MAX_STEPS=1000 ./cli train cache
GPU_LIST=0 MAX_STEPS=1 NUM_WORKERS=0 ./cli train cache
./cli cache build-frozen --limit 100 --overwrite
```
