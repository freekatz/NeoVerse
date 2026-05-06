# Cross-Attention RoPE Adapter 流程图 Prompt

请画一张 CVPR / ICCV 论文风格的 method flowchart，用于说明 NeoVerse control-latent distillation 中的 **Cross-Attention RoPE Adapter** 路径。图应是白底、清晰、工程化的架构图，不要画成宣传海报，不要使用深色背景、渐变光斑或复杂装饰。使用柔和色块区分 frozen modules、trainable modules、tensor reshape/position encoding/loss。

图标题：

```text
Cross-Attention RoPE Adapter for Control-Latent Distillation
```

整体从左到右分成五个区域：

```text
Context Views -> SourceTokenEncoder -> Target Query Encoder -> Cross-Attention RoPE Blocks -> Student Control Latent
```

## 1. 左侧：Context Views and Frozen Token Extraction

画一个输入区域，显示从 81-frame clip 中采样出的 context/source views，标注：

```text
Context/source views
K = 10-20 views
```

从 context views 进入一个蓝灰色冻结模块：

```text
Frozen VGGT / WorldMirror Reconstructor
no gradient
```

输出 context tokens，标注张量形状：

```text
VGGT intermediate tokens
tokens: B x K x L x Hv x Wv x Ctok
default: B x K x 5 x 24 x 40 x 2048
L = 4 intermediate layers + 1 camera token
```

这里要清楚表达：token 来自 frozen reconstructor，训练时 adapter 只读这些 frozen/cache tokens。

## 2. 上方分支：SourceTokenEncoder builds K/V tokens

从 `VGGT intermediate tokens` 画箭头进入橙色可训练模块：

```text
SourceTokenEncoder
trainable
```

在模块内部按步骤画：

```text
spatial pool: 24 x 40 -> 16 x 16
Linear projection: 2048 -> 512
add token-group embedding
add source-view condition
```

显示 shape 变化：

```text
B x K x 5 x 24 x 40 x 2048
    -> pool
B x K x 5 x 16 x 16 x 2048
    -> flatten K,L,H,W and project
Source K/V tokens: B x S x 512
S = K x 5 x 16 x 16
default S = 1280K = 12800-25600
```

从 SourceTokenEncoder 旁边画一个位置编码小模块：

```text
Source 6D RoPE positions
B x S x 6
[time, camera_x, camera_y, camera_z, image_y, image_x]
```

如果空间允许，注明：

```text
time_position_mode = reroped
source times are sorted and mapped with interval = 4.0
```

## 3. 下方分支：Target output grid builds Query tokens

画一个 target grid 输入区域：

```text
Target output grid
F x H x W
default: 21 x 21 x 35
Q = F x H x W = 15435
```

这个 target grid 进入橙色可训练模块：

```text
ConditionGridEncoder
trainable
```

在模块内部列出输入条件：

```text
target time
target pose + intrinsics
target Plucker rays
3D grid position
diffusion timestep
```

显示 shape 变化：

```text
condition grid: B x 512 x F x H x W
    -> flatten spatial-temporal grid
Query tokens: B x Q x 512
```

旁边画 target 位置编码模块：

```text
Target 6D RoPE positions
B x Q x 6
[time, camera_x, camera_y, camera_z, grid_y, grid_x]
```

注明 target time 使用同一个 reroped time coordinate system，与 source RoPE time 对齐。

## 4. 可选但默认开启：LocalGridShortcut and TemporalFiLM

在 SourceTokenEncoder 到 Query tokens 之间画一条浅绿色 shortcut 分支，标注：

```text
LocalGridShortcut
default on
```

显示 shape：

```text
source tokens -> Linear 2048->512
mean over L token groups
B x 512 x K x 24 x 40
    -> resize spatial + interpolate time K -> F
B x 512 x F x H x W
    -> flatten
B x Q x 512
add to Query tokens
```

在 query 和 source token 旁边分别标注：

```text
TemporalFiLM on query
TemporalFiLM on source
```

不要把 LocalGridShortcut 画成主路径，它是加到 query 上的辅助几何先验。

## 5. 中右：Cross-Attention RoPE Blocks

画一个大的橙色模块：

```text
4 x CrossAttentionBlock
trainable
```

输入有四条：

```text
Query tokens: B x Q x 512
Source K/V tokens: B x S x 512
Target 6D positions: B x Q x 6
Source 6D positions: B x S x 6
```

在 block 内画 attention 的 shape：

```text
q: B x 8 x Q x 64
k: B x 8 x S x 64
v: B x 8 x S x 64
RoPE(q, target positions)
RoPE(k, source positions)
scaled dot-product attention
query attends to all source tokens
```

输出：

```text
Updated query tokens
B x Q x 512
```

在模块角落加小注释：

```text
query_chunk_size = 4096
used to reduce memory during Q x S attention
```

重点表达：每个 target latent-grid query 通过 6D RoPE cross-attention 从所有 source-view tokens 中聚合信息。

## 6. 右侧：3D Post Conv and Control-Latent Prediction

Cross-attention 输出进入 3D grid restore：

```text
B x Q x 512
    -> reshape
B x 512 x F x H x W
```

然后进入：

```text
3D post module
1x1 Conv3d + SiLU + 2 x CausalResBlock
```

shape 不变：

```text
B x 512 x F x H x W
```

最后画：

```text
condition_head
Conv3d 512 -> 5120
```

输出：

```text
Student condition latent
B x Q x 5120
default: B x 15435 x 5120
```

## 7. 最右：Distillation Loss

画一个蓝灰色冻结 teacher 分支，简洁表示：

```text
Frozen NeoVerse Control Encoder
Teacher condition latent
B x Q x 5120
```

让 student output 和 teacher output 进入 loss：

```text
Distillation loss
L1 + small statistic matching
train only Cross-Attention RoPE Adapter
teacher and reconstructor are frozen
```

## Visual Style Requirements

使用颜色建议：

```text
Frozen modules: blue-gray with lock icon
Trainable adapter modules: warm orange
Position / RoPE modules: purple
Shape transform / reshape / pool: light gray
Loss: red or coral
```

所有文字应清晰可读。不要堆太多代码，shape 用小标签贴在箭头旁边。主路径必须突出：

```text
Source K/V tokens + 6D RoPE positions
Target Query tokens + 6D RoPE positions
Cross-attention: query attends to source
Student condition latent matches frozen teacher latent
```

