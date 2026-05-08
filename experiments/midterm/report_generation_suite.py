import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def metric_row(df, run_id=None, group=None, method=None, config=None):
    rows = df
    if run_id is not None:
        rows = rows[rows["run_id"] == run_id]
    if group is not None:
        rows = rows[rows["group"] == group]
    if method is not None:
        rows = rows[rows["method"] == method]
    if config is not None:
        rows = rows[rows["config"] == config]
    if len(rows) != 1:
        raise ValueError(f"Expected one row, got {len(rows)} for run_id={run_id}, group={group}, method={method}, config={config}")
    return rows.iloc[0]


def summary_row(summary, group, method):
    rows = summary[(summary["group"] == group) & (summary["method"] == method)]
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row for {group}/{method}, got {len(rows)}")
    return rows.iloc[0]


def save_bar(path, labels, values, title, ylabel="PSNR (dB)", colors=None):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_plots(df, summary, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)

    main_methods = ["render", "teacher", "student"]
    main_values = [summary_row(summary, "main_teacher_student", method)["psnr_mean"] for method in main_methods]
    save_bar(
        plots_dir / "main_psnr_teacher_student.png",
        ["Render", "Teacher", "Student"],
        main_values,
        "Main Generation: GT PSNR",
        colors=["#8a8f98", "#2b6cb0", "#c05621"],
    )

    ctx = df[(df["group"] == "context_count") & (df["method"] == "student")].copy()
    ctx["context"] = ctx["config"].str.extract(r"(\d+)").astype(int)
    ctx = ctx.sort_values("context")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ctx["context"], ctx["psnr"], marker="o", linewidth=2, label="Student PSNR")
    ax.plot(ctx["context"], ctx["ssim"] * 30, marker="s", linewidth=2, label="Student SSIM x30")
    ax.set_title("Context Robustness")
    ax.set_xlabel("Number of context views")
    ax.set_ylabel("Score")
    ax.set_xticks(ctx["context"].tolist())
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "context_student_curve.png", dpi=180)
    plt.close(fig)

    zero = metric_row(df, run_id="E07_control0_case0", method="zero_control")
    half = metric_row(df, run_id="E08_control05_case0", method="teacher_scale_0_5")
    full = metric_row(df, run_id="E01_main_case0", method="teacher")
    save_bar(
        plots_dir / "control_scale_psnr.png",
        ["0.0", "0.5", "1.0"],
        [zero["psnr"], half["psnr"], full["psnr"]],
        "Control Scale Ablation",
        colors=["#9b2c2c", "#b7791f", "#2f855a"],
    )

    temporal = df[(df["group"] == "temporal_trajectory") & (df["method"].isin(["teacher", "student"]))].copy()
    pivot = temporal.pivot(index="config", columns="method", values="psnr").loc[["temporal_backward", "temporal_mixed"]]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = range(len(pivot.index))
    width = 0.34
    ax.bar([i - width / 2 for i in x], pivot["teacher"], width=width, label="Teacher", color="#2b6cb0")
    ax.bar([i + width / 2 for i in x], pivot["student"], width=width, label="Student", color="#c05621")
    ax.set_title("Temporal Trajectory Stress Test")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Backward", "Mixed"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for i, value in enumerate(pivot["teacher"]):
        ax.text(i - width / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    for i, value in enumerate(pivot["student"]):
        ax.text(i + width / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "temporal_teacher_student.png", dpi=180)
    plt.close(fig)


def markdown_table(rows, headers):
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        output.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(output)


def write_report(output_root):
    df = pd.read_csv(output_root / "metrics_by_video.csv")
    summary = pd.read_csv(output_root / "metrics_summary.csv")
    plots_dir = output_root / "plots"
    generate_plots(df, summary, plots_dir)
    mp4_count = len(list((output_root / "runs").glob("*/*.mp4")))

    main_render = summary_row(summary, "main_teacher_student", "render")
    main_teacher = summary_row(summary, "main_teacher_student", "teacher")
    main_student = summary_row(summary, "main_teacher_student", "student")
    context_teacher = summary_row(summary, "context_count", "teacher")
    context_student = summary_row(summary, "context_count", "student")
    temporal_teacher = summary_row(summary, "temporal_trajectory", "teacher")
    temporal_student = summary_row(summary, "temporal_trajectory", "student")

    zero = metric_row(df, run_id="E07_control0_case0", method="zero_control")
    half = metric_row(df, run_id="E08_control05_case0", method="teacher_scale_0_5")
    full = metric_row(df, run_id="E01_main_case0", method="teacher")
    mixed = metric_row(df, run_id="E10_temporal_mixed_case0", method="student")
    mixed_render = metric_row(df, run_id="E10_temporal_mixed_case0", method="render")

    main_student_delta = main_student["psnr_mean"] - main_teacher["psnr_mean"]
    context_gap = context_teacher["psnr_mean"] - context_student["psnr_mean"]
    control_gain = full["psnr"] - zero["psnr"]
    temporal_student_delta = temporal_student["psnr_mean"] - temporal_teacher["psnr_mean"]

    main_rows = []
    for method in ["render", "teacher", "student"]:
        row = summary_row(summary, "main_teacher_student", method)
        main_rows.append([method, int(row["n"]), fmt(row["psnr_mean"]), fmt(row["ssim_mean"], 4), fmt(row["lpips_mean"]), fmt(row["motion_psnr_mean"])])

    ctx_rows = []
    ctx_student_psnr = []
    for run_id in ["E04_context4_case0", "E05_context8_case0", "E06_context16_case0"]:
        student = metric_row(df, run_id=run_id, method="student")
        teacher = metric_row(df, run_id=run_id, method="teacher")
        ctx_num = student["config"].replace("context", "")
        ctx_student_psnr.append(fmt(student["psnr"]))
        ctx_rows.append([ctx_num, fmt(teacher["psnr"]), fmt(student["psnr"]), fmt(student["ssim"], 4), fmt(student["lpips"]), fmt(student["motion_psnr"])])

    control_rows = [
        ["0.0", "zero_control", fmt(zero["psnr"]), fmt(zero["ssim"], 4), fmt(zero["lpips"]), fmt(zero["motion_psnr"])],
        ["0.5", "teacher_scale_0_5", fmt(half["psnr"]), fmt(half["ssim"], 4), fmt(half["lpips"]), fmt(half["motion_psnr"])],
        ["1.0", "teacher (E01)", fmt(full["psnr"]), fmt(full["ssim"], 4), fmt(full["lpips"]), fmt(full["motion_psnr"])],
    ]

    temporal_rows = []
    for run_id, name in [("E09_temporal_backward_case0", "backward"), ("E10_temporal_mixed_case0", "mixed")]:
        teacher = metric_row(df, run_id=run_id, method="teacher")
        student = metric_row(df, run_id=run_id, method="student")
        render = metric_row(df, run_id=run_id, method="render")
        temporal_rows.append([
            name,
            fmt(render["psnr"]),
            fmt(teacher["psnr"]),
            fmt(student["psnr"]),
            fmt(student["ssim"], 4),
            fmt(student["motion_pixels_ratio"], 6),
        ])

    report = f"""# 中期答辩生成实验结论

## 实验口径修正

你指出“1 和 2 不是一个实验吗”是对的。这里不再把跨 scene 的 teacher/student 主对比和 context 数量变化硬拆成两个独立主实验。更科学的组织方式是：

1. Adapter 替换生成主实验：包含 3 个 scene 的 teacher/student 对比，以及 context=4/8/16 作为鲁棒性子维度。
2. Control scale 独立实验：控制强度从 0.0 到 0.5 到 1.0，验证时空控制分支是否真的影响生成。
3. Temporal trajectory 独立实验：backward 和 mixed 轨迹，验证时序/相机路径变化下的稳定性。

之前 latent-cache L1 只能说明第一阶段 adapter 是否接近 teacher control latent，不能证明视频生成可控。这里的 v3 camera-oracle/context-splats 套件直接评估生成视频相对 GT 的质量，所以才是中期答辩可用的主结果。

这次重跑采用你要求的方案 2：全 81 帧只用于估计 target camera/timestamp；实际 splats 和 rendered RGB 只由 context views 重建。因此这里的 render 不再是“81 帧重建基础上渲染”的 oracle 内容。剩余限制是 camera/timestamp 仍然是 oracle，还不是完全由 sparse context 自己估计出来的相机轨迹。

## 关键结论

- Render reference 不是 GT，也不是全 81 帧重建内容：主实验 render PSNR={fmt(main_render["psnr_mean"])}，不是无穷大；说明评估不再把 GT 当 degraded render。
- 主实验不是 teacher 全胜：三 scene 平均 teacher PSNR={fmt(main_teacher["psnr_mean"])}，student PSNR={fmt(main_student["psnr_mean"])}，student 高 {fmt(main_student_delta)} dB；但 teacher 的 LPIPS={fmt(main_teacher["lpips_mean"])} 明显好于 student 的 {fmt(main_student["lpips_mean"])}，所以不能把它解读成 student 已经优于 teacher，只能说像素指标上 student 更接近 GT。
- Context 增多能改善 student，但不能解决根本问题：context=4/8/16 的 student PSNR 从 {" 到 ".join(ctx_student_psnr)}，上升明显，但均低于对应 teacher。
- Control 是必要条件：no-control PSNR={fmt(zero["psnr"])}，full-control teacher PSNR={fmt(full["psnr"])}，提升 {fmt(control_gain)} dB；这证明实验不是无意义的随机生成对比，而是在验证控制链路是否工作。
- Temporal 轨迹主要暴露条件退化边界：teacher 平均 PSNR={fmt(temporal_teacher["psnr_mean"])}，student 平均 PSNR={fmt(temporal_student["psnr_mean"])}，两者差 {fmt(temporal_student_delta)} dB；mixed 中 render/GT 条件视频本身接近空序列，student PSNR 只有 {fmt(mixed["psnr"])}，不应作为正常生成质量证据。

## 主实验：Teacher vs Student

{markdown_table(main_rows, ["method", "n", "PSNR", "SSIM", "LPIPS", "motion PSNR"])}

解释：在这 3 个 base scene 上，student 的 PSNR/SSIM 高于 teacher，但 LPIPS 更差。这是一个有用信号：adapter 替换后并非直接崩溃，且有可能更贴近像素 GT；但它没有证明 student 的时空控制策略优于 teacher，因为感知指标、context 子维度和 temporal stress 都没有同步变好。更合理的结论是：第一阶段 adapter 已经能进入生成链路，但还需要 diffusion/generation 闭环训练来稳定控制能力。

图：`plots/main_psnr_teacher_student.png`

## Context 鲁棒性

{markdown_table(ctx_rows, ["context views", "teacher PSNR", "student PSNR", "student SSIM", "student LPIPS", "student motion PSNR"])}

解释：这个表只包含 E04/E05/E06 的固定 context-count sweep，不包含 E01。E01 是 base 配置，实际使用 19 个 context views 和 `mixed/random_first` 采样；E04/E05/E06 是强制 `uniform_first` 的 4/8/16 views。因此 E01 的 `comparison_grid_gt.mp4` 不能直接拿来和这个表逐行对比。严格的 context ablation 应该固定同一个 clip/start 和同一套嵌套 context indices 后再比较。

在 E04/E05/E06 这个受控子集内部，context 数量增加时 student 指标确实变好，说明 adapter 使用了输入上下文，不是完全无条件生成。但 context=16 仍比 teacher 低约 {fmt(metric_row(df, run_id="E06_context16_case0", method="teacher")["psnr"] - metric_row(df, run_id="E06_context16_case0", method="student")["psnr"])} dB，问题不只是上下文数量不足，而是 teacher control 到 student adapter 的训练目标不够贴近最终生成。

图：`plots/context_student_curve.png`

## Control Scale 实验

{markdown_table(control_rows, ["control_scale", "method", "PSNR", "SSIM", "LPIPS", "motion PSNR"])}

解释：`control_scale=0.0` 直接崩到 PSNR={fmt(zero["psnr"])}，说明 NeoVerse 的生成确实依赖控制条件；`control_scale=0.5` 能恢复到 {fmt(half["psnr"])}，但仍低于 full-control。这个实验回答的是“控制通路是否有效”，是独立于 adapter 替换的控制变量实验。

图：`plots/control_scale_psnr.png`

## Temporal Trajectory 实验

{markdown_table(temporal_rows, ["trajectory", "render PSNR", "teacher PSNR", "student PSNR", "student SSIM", "motion ratio"])}

解释：backward 轨迹下 teacher 为 {fmt(metric_row(df, run_id="E09_temporal_backward_case0", method="teacher")["psnr"])} PSNR，student 为 {fmt(metric_row(df, run_id="E09_temporal_backward_case0", method="student")["psnr"])}；mixed 轨迹下 teacher 为 {fmt(metric_row(df, run_id="E10_temporal_mixed_case0", method="teacher")["psnr"])}，student 为 {fmt(mixed["psnr"])}。mixed 的 motion ratio 只有 {fmt(mixed_render["motion_pixels_ratio"], 6)}，并且条件视频文件很小，说明该配置接近退化/稀疏目标序列；它适合作为压力测试，而不应该作为“正常生成质量”的唯一证据。

图：`plots/temporal_teacher_student.png`

## 对研究路线的含义

这组实验没有证明当前 adapter 已经解决时空可控视频生成；它证明的是更窄的一点：方案 2 修正后，adapter 可以接入真实生成链路并在 base scene 的像素指标上工作，但在 context 减少、感知质量和退化 temporal 轨迹上仍不稳定。第一阶段蒸馏的上限仍然存在：adapter 学到的是 teacher control latent 的近似，而不是在 diffusion 采样中稳定产生正确视频的控制策略。

下一阶段应该做的是把 adapter 放回 denoising/generation 闭环里训练。可行方向包括：对 teacher control hints 做 step-wise distillation；加入 teacher/student 生成 latent 的 denoising loss；增加 temporal consistency 和 motion-region loss；避免通过 RGB 投影造成的信息损失；并把 context/trajectory/control scale 这些变量纳入训练分布，而不是只做离线 latent 对齐。

## 文件清单

- Manifest: `manifest.json`
- Per-video metrics: `metrics_by_video.csv`
- Group summary: `metrics_summary.csv`
- Auto report: `REPORT.md`
- Chinese findings: `FINDINGS_CN.md`
- Plots: `plots/*.png`
- Videos: `runs/*/*.mp4`，共 {mp4_count} 个 mp4。teacher/student run 含 GT、render、teacher、student 和 comparison grids；teacher-only control run 不含 student。
"""

    (output_root / "FINDINGS_CN.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    write_report(args.output_root)


if __name__ == "__main__":
    main()
