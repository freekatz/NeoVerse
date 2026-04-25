# NeoVerse 脚本使用示例

本文档说明 `app.py`、`inference.py`、`train.py` 的常见用法。下面的命令默认使用已有 Python 环境：

```text
/root/vepfs/envs/neoverse/bin/python
```

并默认从 `code/` 目录执行：

```bash
cd /root/vepfs/diffsynth-dev/papers/neoverse/code
```

运行前请确认依赖已安装，模型权重已放在默认目录：

```text
models/NeoVerse/
├── reconstructor.ckpt
├── Wan2.1_VAE.pth
├── diffusion_pytorch_model-*.safetensors
├── models_t5_umt5-xxl-enc-bf16.pth
└── loras/
    └── Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors
```

如果显存不足，优先给推理和 Web 服务加 `--low_vram`。

## 1. app.py：启动交互式 Web Demo

`app.py` 用于启动 NeoVerse 的可视化 Web 页面。它会在启动时加载模型，支持上传视频或图片、重建 4D 场景、预览相机轨迹、生成最终视频。

### 基础启动

```bash
/root/vepfs/envs/neoverse/bin/python app.py
```

默认监听：

```text
http://0.0.0.0:7860/
```

页面入口：

- `/`：轻量交互式 Viewer，可以选择示例、重建场景、自由移动相机查看渲染。
- `/gradio`：完整 Gradio Demo，包含上传、重建、轨迹设计、预览和生成流程。

### 指定端口和地址

```bash
/root/vepfs/envs/neoverse/bin/python app.py \
  --server_name 0.0.0.0 \
  --server_port 7861
```

然后访问：

```text
http://服务器IP:7861/
http://服务器IP:7861/gradio
```

### 低显存启动

```bash
/root/vepfs/envs/neoverse/bin/python app.py --low_vram
```

低显存模式会在需要时把模型加载到 GPU，用完后再卸载到 CPU，显存压力更小，但速度会慢一些。

### 使用自定义重建器权重

```bash
/root/vepfs/envs/neoverse/bin/python app.py \
  --reconstructor_path models/NeoVerse/reconstructor.ckpt
```

也可以传入兼容的其他重建器权重，例如 Depth Anything 3 权重：

```bash
/root/vepfs/envs/neoverse/bin/python app.py \
  --reconstructor_path models/da3_giant_1.1.safetensors \
  --low_vram
```

### 常用输出文件

`app.py` 默认把中间结果和最终结果写到：

```text
outputs/gradio/
├── scene.glb
├── preview.mp4
├── mask.mp4
├── output.mp4
├── render_vs_generated.mp4
└── trajectory.json
```

### Web 使用流程

1. 上传输入视频或图片，选择 `General scene` 或 `Static scene`。
2. 点击重建按钮，生成 4D Gaussian Splat 场景。
3. 选择相机运动，或上传轨迹 JSON。
4. 先预览 RGB 和 mask，再输入 prompt 生成最终视频。

说明：

- 普通视频通常选 `General scene`。
- 单张图片或完全静止的场景选 `Static scene`。
- `--share` 在当前轻量 Viewer 中不会创建可用分享链接，建议直接使用本机或服务器地址访问。

## 2. inference.py：命令行推理生成视频

`inference.py` 用于直接从命令行生成新相机轨迹视频。必须指定输入和轨迹，轨迹有两种方式：

- `--trajectory`：使用内置轨迹。
- `--trajectory_file`：使用自定义 JSON 轨迹文件。

这两个参数互斥，二选一。

### 示例 1：内置轨迹，向上倾斜相机

```bash
/root/vepfs/envs/neoverse/bin/python inference.py \
  --input_path examples/videos/robot.mp4 \
  --trajectory tilt_up \
  --prompt "A two-arm robot assembles parts in front of a table." \
  --output_path outputs/robot_tilt_up.mp4
```

### 示例 2：向右平移相机

```bash
/root/vepfs/envs/neoverse/bin/python inference.py \
  --input_path examples/videos/tree_and_building.mp4 \
  --trajectory move_right \
  --distance 0.2 \
  --output_path outputs/tree_move_right.mp4
```

### 示例 3：静态轨迹加 2 倍变焦

```bash
/root/vepfs/envs/neoverse/bin/python inference.py \
  --input_path examples/videos/animal.mp4 \
  --trajectory static \
  --zoom_ratio 2.0 \
  --output_path outputs/animal_zoom_in.mp4
```

### 示例 4：使用自定义轨迹 JSON

```bash
/root/vepfs/envs/neoverse/bin/python inference.py \
  --input_path examples/videos/movie.mp4 \
  --trajectory_file examples/trajectories/orbit_left_pull_out.json \
  --alpha_threshold 0.95 \
  --output_path outputs/movie_orbit_left_pull_out.mp4
```

### 示例 5：单张图片或静态场景

```bash
/root/vepfs/envs/neoverse/bin/python inference.py \
  --input_path examples/videos/jungle.png \
  --static_scene \
  --trajectory_file examples/trajectories/custom2.json \
  --output_path outputs/jungle_static_scene.mp4
```

### 示例 6：保存中间渲染结果

```bash
/root/vepfs/envs/neoverse/bin/python inference.py \
  --input_path examples/videos/driving.mp4 \
  --trajectory_file examples/trajectories/custom.json \
  --vis_rendering \
  --output_path outputs/driving_custom.mp4
```

加 `--vis_rendering` 后，会在 `outputs/driving_custom/` 这类同名目录下保存目标视角渲染和 alpha 等中间结果，方便检查轨迹和遮罩。

### 示例 7：低显存推理

```bash
/root/vepfs/envs/neoverse/bin/python inference.py \
  --input_path examples/videos/robot.mp4 \
  --trajectory tilt_up \
  --output_path outputs/robot_low_vram.mp4 \
  --low_vram
```

### 示例 8：验证轨迹 JSON，不跑推理

```bash
/root/vepfs/envs/neoverse/bin/python inference.py \
  --trajectory_file examples/trajectories/custom.json \
  --validate_only
```

### 可选：使用仓库里的封装脚本

仓库根目录下有一个推理封装脚本：

```bash
cd /root/vepfs/diffsynth-dev/papers/neoverse
bash scripts/run_neoverse_inference.sh
```

也可以通过环境变量覆盖参数：

```bash
cd /root/vepfs/diffsynth-dev/papers/neoverse
INPUT_PATH=code/examples/videos/robot.mp4 \
OUTPUT_PATH=code/outputs/robot_tilt_up.mp4 \
TRAJECTORY=tilt_up \
LOW_VRAM=1 \
bash scripts/run_neoverse_inference.sh
```

### 内置轨迹列表

```text
pan_left / pan_right       水平旋转相机
tilt_up / tilt_down        垂直旋转相机
move_left / move_right     左右平移相机
push_in / pull_out         向前推进或向后拉远
boom_up / boom_down        上下平移相机
orbit_left / orbit_right   围绕场景中心环绕
static                     保持相机不动，可配合 zoom_ratio 做变焦
```

### 常用参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input_path` | 无 | 输入视频或图片路径。正常推理必须提供。 |
| `--output_path` | `outputs/inference.mp4` | 输出视频路径。建议带目录，例如 `outputs/xxx.mp4`。 |
| `--trajectory` | 无 | 使用内置轨迹。 |
| `--trajectory_file` | 无 | 使用自定义轨迹 JSON。 |
| `--prompt` | 默认补全场景 prompt | 文本提示词。 |
| `--negative_prompt` | 空 | 负向提示词。 |
| `--model_path` | `models` | 模型根目录。 |
| `--reconstructor_path` | `models/NeoVerse/reconstructor.ckpt` | 重建器权重路径。 |
| `--num_frames` | `81` | 输出帧数。 |
| `--height` / `--width` | `336` / `560` | 输入处理和输出分辨率。 |
| `--resize_mode` | `center_crop` | 视频缩放方式，可选 `center_crop` 或 `resize`。 |
| `--seed` | `42` | 随机种子。 |
| `--static_scene` | 关闭 | 单图或静态场景建议开启。 |
| `--low_vram` | 关闭 | 开启 CPU/GPU 换入换出，降低显存占用。 |
| `--disable_lora` | 关闭 | 不使用 4 步蒸馏 LoRA，改用更慢的 50 步推理。 |
| `--vis_rendering` | 关闭 | 保存中间渲染可视化结果。 |

几个容易混淆的参数：

- `--traj_mode relative`：默认模式，轨迹相对于原始相机运动叠加。
- `--traj_mode global`：直接使用世界坐标下的轨迹矩阵。
- `--alpha_threshold`：控制哪些区域保留重建渲染，哪些区域交给扩散模型重绘。值越高，重绘区域通常越大。
- `--zoom_ratio`：从 1.0 逐渐变到指定倍率，只改变焦距，不改变相机位置。

## 3. train.py：训练或微调

`train.py` 使用配置文件启动训练，入口格式是：

```bash
/root/vepfs/envs/neoverse/bin/python train.py training/configs/train.yaml
```

实际训练建议用 `/root/vepfs/envs/neoverse/bin/accelerate launch`，因为 `training/utils.py` 基于 Hugging Face Accelerate，支持 DeepSpeed ZeRO-2。

### 示例 1：单机快速启动

```bash
/root/vepfs/envs/neoverse/bin/accelerate launch train.py training/configs/train.yaml
```

这会读取：

```text
training/configs/train.yaml
```

默认输出到：

```text
models/train/NeoVerse/
```

### 示例 2：多机或多卡 ZeRO-2 训练

```bash
/root/vepfs/envs/neoverse/bin/accelerate launch \
  --use_deepspeed \
  --deepspeed_config_file training/configs/zero_stage2_config.json \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --num_machines $WORLD_SIZE \
  --num_processes $NUM_PROCESS \
  --machine_rank $NODE_RANK \
  --deepspeed_multinode_launcher standard \
  train.py training/configs/train.yaml
```

其中常见环境变量含义：

```text
MASTER_ADDR   主节点地址
MASTER_PORT   主节点通信端口
WORLD_SIZE    机器数量
NUM_PROCESS   总进程数，通常等于总 GPU 数
NODE_RANK     当前机器编号，从 0 开始
```

### 示例 3：调试模式

```bash
/root/vepfs/envs/neoverse/bin/python train.py training/configs/train.yaml --debug
```

调试模式会把 `num_workers` 设为 0，并监听 `debugpy` 的 `5678` 端口，等待调试器连接。

### 训练配置重点

常用配置在 `training/configs/train.yaml`：

```yaml
height: 336
width: 560
num_views: 81

train_dataset: SpatialVID(split=None, ROOT="data/SpatialVID", ...)

model_path: ./models
reconstructor_path: ./models/NeoVerse/reconstructor.ckpt

trainable_models: control_branch.*
learning_rate: 1e-5
num_epochs: 1
batch_size: 1
num_workers: 1
output_path: ./models/train/NeoVerse

use_gradient_checkpointing: true
gradient_accumulation_steps: 1
save_freq: 0.05
resume: null
```

### 修改数据路径

默认数据集是：

```yaml
train_dataset: SpatialVID(split=None, ROOT="data/SpatialVID", ...)
```

如果换成自己的数据，需要修改 `ROOT` 或实现新的 Dataset，并把 `train_dataset` 改成对应构造表达式。当前 `SpatialVID` 训练只依赖 RGB 视频和文本 prompt，不要求预先提供深度、相机内外参。

### 恢复训练

把配置里的 `resume` 改成已有 checkpoint 目录：

```yaml
resume: ./models/train/NeoVerse/checkpoint-epoch-1
```

然后重新启动：

```bash
/root/vepfs/envs/neoverse/bin/accelerate launch train.py training/configs/train.yaml
```

### 输出内容

训练时会在 `output_path` 下保存：

- 当前配置 `config.yaml`
- 当前代码快照
- TensorBoard 日志
- 训练 checkpoint 或导出的 safetensors 权重

可以用 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir models/train/NeoVerse
```

## 4. wan_4step_gradio.py：测试 4-step distilled Wan 文生视频

`wan_4step_gradio.py` 是一个独立 Gradio 页面，用来测试本地 Wan 2.1 T2V 权重加 LightX2V 4-step distilled LoRA。它不走 NeoVerse 重建器，只做普通文生视频。

启动：

```bash
cd /root/vepfs/diffsynth-dev/papers/neoverse/code
/root/vepfs/envs/neoverse/bin/python wan_4step_gradio.py \
  --server_name 0.0.0.0 \
  --server_port 7862 \
  --low_vram
```

然后访问：

```text
http://服务器IP:7862/
```

默认模型路径：

```text
models/NeoVerse/
models/NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors
```

推荐先用默认参数测试：

```text
width=560
height=336
num_frames=81
num_inference_steps=4
cfg_scale=1.0
fps=16
```

说明：

- 页面启动后不会立刻加载大模型，点击“加载模型”或“生成视频”才会加载。
- 当前仓库默认用 `推理步数=4`、`CFG scale=1.0` 跑这个 distilled LoRA。
- 输出视频默认保存在 `outputs/wan_4step_gradio/`。

如果不想低显存换入换出，可以去掉 `--low_vram`，但需要更高显存。

## 5. 推荐最小测试顺序

第一次跑通项目时，建议按这个顺序验证：

```bash
cd /root/vepfs/diffsynth-dev/papers/neoverse/code

# 1. 先验证轨迹文件格式
/root/vepfs/envs/neoverse/bin/python inference.py \
  --trajectory_file examples/trajectories/custom.json \
  --validate_only

# 2. 再跑一个低显存命令行推理
/root/vepfs/envs/neoverse/bin/python inference.py \
  --input_path examples/videos/robot.mp4 \
  --trajectory tilt_up \
  --output_path outputs/smoke_test.mp4 \
  --low_vram

# 3. 最后启动 Web Demo
/root/vepfs/envs/neoverse/bin/python app.py --low_vram --server_port 7860

# 4. 单独测试 4-step Wan 文生视频
/root/vepfs/envs/neoverse/bin/python wan_4step_gradio.py --low_vram --server_port 7862
```
