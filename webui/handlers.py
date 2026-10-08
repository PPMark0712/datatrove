"""Presentation helpers shared by the dataset and task views."""

from html import escape
from pathlib import Path

import gradio as gr

from task_manager import Task, list_tasks, list_workspace_datasets, load_scores, plot_score_distribution


STATUS_NAMES = {
    "created": "待配置", "running": "运行中", "stopping": "停止中",
    "completed": "已完成", "error": "失败", "stopped": "已停止", "unknown": "未知",
}
STEP_NAMES = {
    "split": "输入分片", "data_clean": "数据清洗", "calc_fcd": "FCD 打分",
    "calc_cdf_gc": "CDF-GC 筛选", "sample": "采样", "merge_output": "整理结果",
}


def dataset_choices():
    return [(f'{d["name"]} · {d["n_files"]} 个文件', d["name"]) for d in list_workspace_datasets()]


def task_choices():
    return [(f'{t["display_name"]} · {STATUS_NAMES.get(t["status"], t["status"])}', t["task_id"])
            for t in list_tasks()]


def next_dataset_name():
    names = {d["name"] for d in list_workspace_datasets()}
    number = 1
    while f"数据集{number}" in names:
        number += 1
    return f"数据集{number}"


def row_count(paths):
    total = 0
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            total += sum(bool(line.strip()) for line in source)
    return total


def status_html(task):
    status = task.get_status()
    state = status.get("status", "unknown")
    step = STEP_NAMES.get(status.get("step"), status.get("step", ""))
    return (
        f'<div class="status-line"><span class="badge badge-{escape(state)}">'
        f'{STATUS_NAMES.get(state, escape(state))}</span>'
        f'<strong>{escape(task.get_display_name())}</strong>'
        f'<span>{escape(step)}</span></div>'
        f'<div class="muted">{escape(status.get("detail", ""))}</div>'
    )


def result_view(task):
    state = task.get_status().get("status")
    if state != "completed":
        messages = {
            "created": "尚未运行。请在「配置与运行」中设置处理步骤并开始处理。",
            "running": "任务正在运行，完成后将自动显示结果。可在「运行日志」中查看进度。",
            "stopping": "正在停止任务，结果尚未生成。",
            "stopped": "任务已停止，尚无完整结果。可继续运行，或修改配置后重新运行。",
            "error": "任务未完成，请查看「运行日志」中的错误原因，修正后继续或重新运行。",
        }
        return messages.get(state, "任务完成后，将在这里展示输出、过滤统计和下载入口。"), [], 0
    total = task.count_input_rows()
    output = row_count(Path(task.output_dir).glob("*.jsonl"))
    config = task.load_config()
    stats = [["输入", total]]
    if "data_clean" in config.get("steps", []):
        clean = Path(task.work_dir) / "data_clean"
        stats.extend([
            ["语言过滤后", row_count((clean / "1_language_filter" / "output").rglob("*.jsonl"))],
            ["质量过滤后", row_count((clean / "2_quality_filter").glob("*/output/*.jsonl"))],
            ["去重后", row_count((clean / "3_minhash_deduplication").glob("*/4_result/output/*.jsonl"))],
        ])
    if "fcd" in config.get("steps", []):
        stats.append(["FCD 打分后（采样前）", row_count((Path(task.work_dir) / "fcd" / "annotated").glob("*.jsonl"))])
    stats.append(["最终输出", output])
    removed = max(total - output, 0)
    summary = f"**输入 {total:,} 条 → 输出 {output:,} 条** · 减少 {removed:,} 条"
    if total:
        summary += f" · 保留率 {output / total:.1%}"
    if not output:
        summary += "\n\n所有记录均被过滤或未被采样选中；任务已正常结束。"
    if config.get("sample"):
        sample = config["sample"]
        summary += f'\n\n采样：**{sample["method"]}** · 保留比例 **{sample["sample_rate"]:.0%}**（相对于采样前数据）'
    return summary, stats, output


def config_values(task=None):
    config = task.load_config() if task else {}
    steps = config.get("steps", ["data_clean"])
    fcd, gc, sample = (config.get(key, {}) for key in ("fcd_params", "gc_params", "sample"))
    return [
        "data_clean" in steps, config.get("languages", ["zh", "en"]),
        "fcd" in steps, fcd.get("freq_scaling_factor", 0.7), fcd.get("w_f", 0.5), fcd.get("noun_weight", 0.7),
        "cdf_gc" in steps, gc.get("sample_rate", 0.2), gc.get("tokenizer_path", ""),
        gc.get("ltp_model_path", ""), gc.get("cuda_visible_devices", "0"), gc.get("workers_per_gpu", 4),
        bool(sample), sample.get("method", "cdf"), sample.get("sample_rate", 0.2), "doc",
        config.get("workers", 4),
    ]


def config_controls(values, busy=False):
    """Enable only parameters belonging to an enabled processing step."""
    clean, _, fcd, _, _, _, gc, _, _, _, _, _, sample, *_ = values
    enabled = [
        True, clean, True, fcd, fcd, fcd, True, gc, gc, gc, gc, gc,
        fcd, sample and fcd, sample and fcd, False, True,
    ]
    return [gr.update(interactive=bool(flag) and not busy) for flag in enabled]


def config_hint(values):
    clean, languages, fcd, _, _, _, gc, _, _, _, _, _, sample, *_ = values
    if not any((clean, fcd, gc)):
        return "请选择至少一个处理步骤。"
    if clean and not languages:
        return "请选择清洗后保留的语言。"
    if fcd and gc:
        return "FCD 和 CDF-GC 分别面向英文和中文，请分成两个任务运行。"
    if fcd and clean and languages != ["en"]:
        return "FCD 面向英文：请将保留语言设为仅 en。"
    if gc and clean and languages != ["zh"]:
        return "CDF-GC 面向中文：请将保留语言设为仅 zh。"
    if sample and not fcd:
        return "采样依赖 FCD 打分，请重新启用 FCD 后调整采样。"
    return "配置修改会暂存在当前会话，切换任务或刷新状态不会丢失；刷新整个网页后恢复上次运行配置。"


def make_config(clean, languages, fcd, freq, wf, noun, gc, gc_rate, tokenizer, ltp,
                cuda, gpu_workers, sample, method, sample_rate, unit, workers):
    steps = [name for enabled, name in ((clean, "data_clean"), (fcd, "fcd"), (gc, "cdf_gc")) if enabled]
    if not steps:
        raise gr.Error("请至少启用一个处理步骤")
    if clean and not languages:
        raise gr.Error("请选择至少一种目标语言")
    if sample and not fcd:
        raise gr.Error("采样需要先启用 FCD 打分")
    if sample and unit != "doc":
        raise gr.Error("当前平台支持按文档采样；按 token 采样需额外提供逐文档 token 计数")
    if fcd and gc:
        raise gr.Error("FCD 面向英文、CDF-GC 面向中文，请分别创建任务")
    if fcd and clean and languages != ["en"]:
        raise gr.Error("FCD 面向英文，请将清洗语言设为仅 en")
    if gc:
        if clean and languages != ["zh"]:
            raise gr.Error("CDF-GC 面向中文，请将清洗语言设为仅 zh")
        if not tokenizer or not Path(tokenizer).is_file():
            raise gr.Error("请填写存在的 Tokenizer 文件路径")
        if not ltp or not Path(ltp).is_dir():
            raise gr.Error("请填写存在的 LTP 模型目录")
        try:
            import torch
        except ImportError as exc:
            raise gr.Error("当前环境未安装 PyTorch，请先配置 CUDA 和 CDF-GC 依赖") from exc
        if not torch.cuda.is_available():
            raise gr.Error("当前服务器没有可用的 CUDA GPU，请在 GPU 服务器上运行 CDF-GC")
    return {
        "steps": steps, "languages": languages, "workers": int(workers), "rerun": True,
        "fcd_params": {"freq_scaling_factor": freq, "w_f": wf, "noun_weight": noun},
        "gc_params": {"sample_rate": gc_rate, "language": "zh", "tokenizer_path": tokenizer,
                      "ltp_model_path": ltp, "cuda_visible_devices": cuda, "workers_per_gpu": int(gpu_workers)},
        **({"sample": {"method": method, "sample_rate": sample_rate, "unit": unit, "score_source": "fcd"}} if sample else {}),
    }


def plot_scores(task_id, score_type):
    if not task_id:
        raise gr.Error("请先选择任务")
    values = load_scores(Task(task_id), score_type)
    if not values:
        raise gr.Error("该任务没有此类分数，请先运行相应打分步骤")
    fig = plot_score_distribution(values, f"{score_type.upper()} Score")
    # Gradio serializes the figure object; pyplot need not retain it globally.
    import matplotlib.pyplot as plt
    plt.close(fig)
    return fig
