# NeoVerse Student Adapter 蒸馏

这个实验用于训练一个 Student Adapter：直接从 NeoVerse / VGGT-like reconstruction backbone 的中间 tokens 预测 NeoVerse control branch 的早期 condition embedding `c`。这个 `c` 位于 `control_patch_embedding(...)` 之后、进入 frozen `control_blocks` 之前。训练时 NeoVerse reconstructor、4DGS degraded render teacher path、control branch 和 VAE 全部冻结，只更新 student adapter。早期 `c` 蒸馏不需要 Wan DiT 或 text encoder，默认训练配置不会加载它们；替换评估或真正 generation 时才需要加载 frozen DiT/text encoder，并把 student 预测的 `c` 送入 frozen `control_blocks` 得到 DiT hints。

## 文件

- `hooks/extract_vggt_tokens.py`：按 Gen3R 方式提取 `[4, 11, 17, 23]` 层 tokens，输出 `B x N x L x h_v x w_v x C`，如果存在 camera token 会 broadcast 后作为第 5 组 token 拼接。
- `diffsynth/models/student_adapters.py`：包含 `ConvAdapter` baseline 和 `CrossAttentionRoPEAdapter` ablation，默认输出早期 condition embedding。
- `train_distill_control_latent.py`：teacher-student latent distillation 训练入口。
- `configs/distill_control_latent.yaml`：默认训练配置。
- `eval_replace_teacher_with_student.py`：用 student `c` 替换 teacher `c`，再经过 frozen control blocks 生成 hints 并跑 generation 对比。

## 训练

在 `code/` 目录下运行：

```bash
accelerate launch train_distill_control_latent.py configs/distill_control_latent.yaml
```

`configs/distill_control_latent.yaml` 是唯一默认配置，默认 `max_steps: 20000`。正式长训请运行：

```bash
bash scripts/train_distill_control_latent.sh
```

脚本会把每次运行写到 `outputs/NeoVerseControlLatentDistill/YYYY-MM-DD/HH-MM-SS/`，日志写到该目录下的 `logs/`，并启用 `auto_resume: true`；如果显式设置 `OUTPUT_PATH` 指向已有 run 目录，且其中已有 `adapter_last.pt`，会自动恢复 adapter 和 optimizer。

临时 smoke test 不需要单独配置文件，直接用命令行覆盖：

```bash
accelerate launch train_distill_control_latent.py configs/distill_control_latent.yaml max_steps=1 num_workers=0 output_path=outputs/tmp/distill_smoke
```

默认配置使用卷积版 adapter。若更换 reconstructor 或 backbone preset，需要检查配置中的 `token.token_dim` 和 `token.token_groups`。当前仓库自带的 NeoVerse `reconstructor.ckpt` 会被识别为 WorldMirror，预期为：

- `token_dim: 2048`
- `token_groups: 5`

训练输出写入 `output_path`：

- `adapter_step_*.pt` / `adapter_last.pt`：adapter-only checkpoint，包含 optimizer state、config 和最近一次 shape 记录。
- `config.yaml`：解析后的运行配置。
- `visuals/*.png`：teacher/student condition embedding 和 absolute error heatmap。
- TensorBoard scalars：每层 L1 / L2 / cosine / latent statistics。

## 替换评估

在 `code/` 目录下运行：

```bash
python eval_replace_teacher_with_student.py \
  configs/distill_control_latent.yaml \
  outputs/NeoVerseControlLatentDistill/YYYY-MM-DD/HH-MM-SS/adapter_last.pt \
  --output_dir outputs/distill_eval \
  --dataset_index 0 \
  --modes teacher,student,combined
```

评估模式：

- `teacher`：原始 NeoVerse degraded-render condition path。
- `student`：绕过 degraded RGB / depth / mask 的 early condition embedding，改用 student 预测的 `c`，然后复用 frozen control blocks 得到最终 hints。
- `combined`：teacher `c` 和 student `c` 取平均，再经过 frozen control blocks，用于调试和过渡实验。

评估脚本默认跟官方 `inference.py` 对齐：如果存在 LightX2V 4-step LoRA，则使用 `num_inference_steps=4, cfg_scale=1.0`；如果传 `--disable_lora`，则使用 full inference 的 `num_inference_steps=50, cfg_scale=5.0`。也可以手动传 `--num_inference_steps` 和 `--cfg_scale` 覆盖。

## 关键设计

- 当前蒸馏目标是早期 condition embedding `c`，shape 与 DiT patch token sequence 对齐，通常为 `B x seq x dit_dim`。
- `c` 不依赖当前 denoising step；student 输入只需要 VGGT tokens 和目标 camera/time condition。
- 评估替换时，student `c` 还需要进入 frozen `control_blocks`，由它结合当前 noisy latent、timestep、text context 生成最终 per-block hints。
- source time / source camera 会显式注入到 reconstruction tokens。默认使用 NeoVerse/WorldMirror 自己估计的相机轨迹，按 timestamp gather 出 context/source 相机。
- SpatialVID annotation camera 是可选项。配置中 `use_camera_annotations: false` 为默认值；设为 `true` 时才读取 `annotations/*/poses.npy` 和 `intrinsics.npy`，teacher render 会优先使用这些 annotation camera/time，并让 reconstructor 和 student token extraction 只使用 context/source 图像。
- target camera 使用仓库已有 Plucker map 构造逻辑，同时也使用 per-frame pose/intrinsics MLP embedding。
- source/target time 使用同一个归一化尺度；target time 会插值到 Wan control latent temporal grid，source time 会标注每个 context/source token 的原始时间。
- 不修改 diffusion backbone，不训练 reconstructor，不训练 teacher control branch。
- Student 主输入是 backbone intermediate tokens，不使用最终 DPT depth / pointmap / 4DGS attributes 作为主输入。

## 常见改动

切换到 cross-attention + RoPE adapter 时，在配置中把：

```yaml
adapter:
  type: cross_attention_rope
```

并按显存情况调整：

- `source_pool_hw`
- `max_source_tokens`
- `query_chunk_size`
- `hidden_dim`
- `num_blocks`
