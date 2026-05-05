# NeoVerse Control-Latent Distillation 架构图中文 Prompt

请画一张 CVPR 论文风格的 method figure，用于说明我们在 NeoVerse 之上新增的 **Control-Latent Distillation** 训练流程。注意：这不是原始 NeoVerse 论文中的主训练图，而是新增的 teacher-to-student distillation 分支。

整体风格参考学术方法图：白底、柔和 pastel 色块、圆角模块、黑色箭头、简洁、文字清晰可读。不要画成宣传海报，不要使用深色背景、渐变光斑或装饰性元素。

图标题：

```text
Control-Latent Distillation for NeoVerse
```

整体从左到右分成四个区域：

```text
Input clip -> Shared Frozen Reconstructor -> Trainable ConvAdapter -> Latent Distillation
```

## 1. 左侧：Input clip

画一个 RGB 视频帧序列，标注：

```text
RGB video clip
81 contiguous frames
```

下面画一条时间轴，从 0 到 80，共 81 个时间点。高亮其中均匀采样的 K 个 context frames，标注：

```text
K=10-20 context frames
uniformly sampled
```

在旁边加一个小注释，避免和原始 NeoVerse 论文混淆：

```text
Distillation default: K=10-20
Original NeoVerse generation training: 11-21 keyframes
```

其余 target 时间点用浅灰色表示，标注：

```text
81 target timestamps / cameras
```

从输入区域画两条箭头进入共享冻结重建器：

```text
full 81 views + context mask
K context RGB frames
```

含义要表达清楚：

```text
Use all 81 views to define / predict the target trajectory;
use context frames for splats and token extraction.
```

## 2. 中间：Shared Frozen Reconstructor

不要把 `Frozen 4DGS Reconstructor` 和 `Frozen VGGT Token Extractor` 画成两个完全独立模块。它们应统一成一个共享的冻结重建器模块，因为 token 和 teacher render 都来自同一个 frozen reconstructor/backbone。

大模块标题：

```text
Shared Frozen Reconstructor
VGGT / WorldMirror backbone
no gradient
```

模块使用蓝绿色，带 lock 或 snowflake 图标，表示冻结。

在这个共享重建器内部，简单画出结构：

```text
RGB view encoder
Multi-view transformer backbone
```

从 backbone 分出三个 heads：

```text
Intermediate tokens
Camera / depth heads
4D Gaussian head
```

不要画过多内部细节。

共享重建器有两条输出分支。

### 2.1 Token 分支

从 `Intermediate tokens` 输出：

```text
Context tokens
B x K x 5 x 24 x 40 x 2048
from intermediate backbone layers
```

这条分支后面进入 student adapter。

### 2.2 Teacher condition 分支

从 camera/depth/4D Gaussian heads 输出：

```text
K-context Gaussian splats
+ 81 target trajectory
```

然后进入：

```text
Render at 81 target views
```

renderer 输出：

```text
Rendered RGB / Depth / Mask
```

同时从 81 target trajectory 生成 camera condition，标注：

```text
Target Plucker maps
81 frames
```

`Rendered RGB / Depth / Mask` 和 `Target Plucker maps` 一起进入：

```text
Frozen NeoVerse Control Encoder
```

输出：

```text
Teacher condition latent
B x 15435 x 5120
15435 = 21 x 21 x 35
```

在共享重建器下方放一个很小的说明：

```text
Same frozen reconstructor is reused:
tokens come from intermediate layers;
teacher conditions come from rendered 4DGS outputs plus target Plucker maps.
```

## 3. 右中：Trainable ConvAdapter

画一个黄色/橙色的大模块：

```text
Trainable ConvAdapter
```

加绿色 tag：

```text
Trainable
```

不要画 adapter 内部细节，不要画 resize、conv layers、resblocks。

`Context tokens` 箭头进入 ConvAdapter。

另外画一个单独的 metadata 侧输入，只连接到 ConvAdapter，不要连接到 VGGT/backbone/reconstructor。标注：

```text
Metadata to adapter only
source: times + poses + intrinsics (K)
target: times + poses + intrinsics + Plucker rays (81)
```

可以补一句小字：

```text
poses / intrinsics are predicted by the frozen reconstructor
or taken from GT annotations in strict sparse-input experiments.
```

ConvAdapter 内部或下方写：

```text
Inputs: context tokens + adapter metadata
Output: student condition latent
```

输出：

```text
Student condition latent
B x 15435 x 5120
15435 = 21 x 21 x 35
```

## 4. 最右：Latent distillation

Teacher condition latent 和 Student condition latent 都指向 loss 模块。loss 模块用浅红色/粉色：

```text
Distillation loss
L = L1(Student, Teacher) + 0.01 Stat(mean/std)
```

从 loss 画一条回传箭头，只回到 `Trainable ConvAdapter`，标注：

```text
update adapter only
```

底部放一行小字：

```text
After training: replace teacher condition latent with student-predicted condition latent in the frozen Wan/NeoVerse generation pipeline.
```

## 必须遵守

- 不要出现 `Frozen cache`。
- 不要把 `Frozen 4DGS Reconstructor` 和 `Frozen VGGT Token Extractor` 画成两个独立模块。
- 必须统一成一个 `Shared Frozen Reconstructor`。
- metadata 箭头只能连接到 `Trainable ConvAdapter`。
- Teacher branch 的 frozen control encoder 需要接收 `Rendered RGB / Depth / Mask` 和 `Target Plucker maps`。
- VGGT/backbone/reconstructor、4DGS head、Control Encoder、Wan/NeoVerse pipeline 都是 frozen。
- 只有 `Trainable ConvAdapter` 是可训练模块。
- 不要展示 ConvAdapter 的内部 resize/conv/resblock 细节。
- 不要把这张图说成原始 NeoVerse 论文已有模块；它是我们新增的 distillation path。
- 图中必须包含这些英文短语：

```text
Shared Frozen Reconstructor
VGGT / WorldMirror backbone
Context tokens
Teacher condition latent
Student condition latent
Trainable ConvAdapter
Metadata to adapter only
Target Plucker maps
```
