import os
import json
import glob as glob_module

import gradio as gr

from task_manager import (
    Task, receive_files, receive_directory,
    run_pipeline_async, list_tasks,
    load_scores, plot_score_distribution,
)

WEBUI_DIR = os.path.dirname(os.path.abspath(__file__))
AUTH_USERS = os.environ.get("WEBUI_AUTH", "admin:datatrove2024")

CUSTOM_CSS = """
.main-title {
    text-align: center;
    margin-bottom: 0.5em;
}
.main-title h1 {
    font-size: 2em;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.main-title p {
    color: #6b7280;
    font-size: 0.95em;
    margin-top: 0.2em;
}
.stat-card {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
}
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.85em;
    font-weight: 600;
}
footer { display: none !important; }
.gradio-container { max-width: 1200px !important; }
"""


def parse_auth(auth_str: str):
    pairs = []
    for pair in auth_str.split(","):
        if ":" in pair:
            u, p = pair.split(":", 1)
            pairs.append((u.strip(), p.strip()))
    return pairs


# ============================================================
# Tab 1: Upload & Preview
# ============================================================

def handle_upload(files, dir_path, task_id_input):
    task_id = task_id_input.strip() if task_id_input and task_id_input.strip() else None
    if files:
        task = receive_files(files, task_id)
    elif dir_path and dir_path.strip():
        task = receive_directory(dir_path.strip(), task_id)
    else:
        return "请上传文件或填写目录路径", "", "等待上传...", gr.update()

    n_files = task.count_input_files()
    if n_files == 0:
        return (
            f"任务 `{task.task_id}` 已创建，但未找到 .jsonl 文件",
            task.task_id,
            "无数据",
            gr.update(value=task.task_id),
        )

    n_rows = task.count_input_rows()
    preview = load_preview(task.input_dir)
    info = f"任务 `{task.task_id}` 已创建 — {n_files} 个文件，共 {n_rows:,} 行"
    return info, task.task_id, preview, gr.update(value=task.task_id)


def load_preview(input_dir: str, max_rows: int = 10) -> str:
    rows = []
    for f in sorted(glob_module.glob(os.path.join(input_dir, "*.jsonl")))[:3]:
        with open(f) as fh:
            for line in fh:
                if len(rows) >= max_rows:
                    break
                try:
                    item = json.loads(line)
                    text = item.get("text", "")
                    if len(text) > 150:
                        text = text[:150] + "..."
                    rows.append({"id": item.get("id", ""), "text": text})
                except json.JSONDecodeError:
                    continue
    if not rows:
        return "无数据可预览"
    lines = ["| # | id | text |", "|:--|:--|:--|"]
    for i, r in enumerate(rows):
        lines.append(f"| {i+1} | `{r['id']}` | {r['text']} |")
    return "\n".join(lines)


# ============================================================
# Tab 2: Pipeline Configuration & Execution
# ============================================================

def run_task(
    task_id,
    enable_clean, clean_languages,
    enable_fcd, fcd_freq_scaling, fcd_wf, fcd_noun_weight,
    enable_gc, gc_sample_rate, gc_language, gc_tokenizer_path,
    gc_ltp_model_path, gc_cuda_devices, gc_workers_per_gpu,
    enable_sample, sample_method, sample_rate, sample_unit, sample_score_source,
    max_workers,
):
    if not task_id or not task_id.strip():
        return "请先在「数据上传」页创建任务", ""

    task = Task(task_id.strip())
    if not os.path.exists(task.input_dir):
        return f"任务 `{task_id}` 不存在，请先上传数据", ""

    status = task.get_status()
    if status.get("status") == "running":
        return f"任务 `{task_id}` 正在运行，请等待完成后再操作", task.get_log()

    steps = []
    config = {}

    if enable_clean:
        steps.append("data_clean")
        langs = [l.strip() for l in clean_languages.split(",") if l.strip()]
        config["languages"] = langs if langs else ["zh"]

    if enable_fcd:
        steps.append("fcd")
        config["fcd_params"] = {
            "freq_scaling_factor": fcd_freq_scaling,
            "w_f": fcd_wf,
            "noun_weight": fcd_noun_weight,
        }

    if enable_gc:
        steps.append("cdf_gc")
        config["gc_params"] = {
            "sample_rate": gc_sample_rate,
            "language": gc_language,
            "tokenizer_path": gc_tokenizer_path,
            "ltp_model_path": gc_ltp_model_path,
            "cuda_visible_devices": gc_cuda_devices,
            "workers_per_gpu": int(gc_workers_per_gpu),
        }

    if enable_sample and enable_fcd:
        config["sample"] = {
            "method": sample_method,
            "sample_rate": sample_rate,
            "unit": sample_unit,
            "score_source": sample_score_source,
        }

    if not steps:
        return "请至少选择一个处理步骤", ""

    config["steps"] = steps
    run_pipeline_async(task, config, workers=int(max_workers))
    step_names = {"data_clean": "数据清洗", "fcd": "FCD 打分", "cdf_gc": "CDF-GC"}
    display_steps = " → ".join(step_names.get(s, s) for s in steps)
    if enable_sample and enable_fcd:
        display_steps += f" → {sample_method.upper()} 采样"
    return f"任务 `{task_id}` 已启动：{display_steps}", ""


def poll_status(task_id):
    if not task_id or not task_id.strip():
        return "请输入任务 ID", "", None
    task = Task(task_id.strip())
    status = task.get_status()
    s = status.get("status", "unknown")
    step = status.get("step", "")
    detail = status.get("detail", "")

    status_emoji = {
        "created": "🔵", "running": "🟡", "completed": "🟢", "error": "🔴"
    }.get(s, "⚪")
    status_text = f"{status_emoji} **{s.upper()}**"
    if step:
        status_text += f"  ·  当前步骤: `{step}`"
    if detail:
        status_text += f"\n\n> {detail}"

    log_tail = task.get_log()
    if log_tail and len(log_tail) > 8000:
        log_tail = "...(truncated)...\n" + log_tail[-8000:]

    fig = try_plot_scores(task)
    return status_text, log_tail, fig


def try_plot_scores(task: Task):
    for score_type, title in [("fcd", "FCD Score"), ("gc", "GC Score")]:
        scores = load_scores(task, score_type)
        if scores:
            return plot_score_distribution(scores, title)
    return None


def plot_scores_manual(task_id, score_type):
    if not task_id or not task_id.strip():
        return None
    task = Task(task_id.strip())
    title_map = {"fcd": "FCD Score", "gc": "GC Score"}
    scores = load_scores(task, score_type)
    return plot_score_distribution(scores, title_map.get(score_type, "Score"))


def download_result(task_id):
    if not task_id or not task_id.strip():
        return None
    task = Task(task_id.strip())
    status = task.get_status()
    if status.get("status") != "completed":
        return None
    archive = task.get_output_archive()
    if os.path.exists(archive):
        return archive
    return None


# ============================================================
# Tab 3: Task Management
# ============================================================

def refresh_task_list():
    tasks = list_tasks()
    if not tasks:
        return "暂无任务记录"
    lines = ["| 任务 ID | 状态 | 步骤 | 详情 |", "|:--|:--|:--|:--|"]
    emoji = {"created": "🔵", "running": "🟡", "completed": "🟢", "error": "🔴"}
    for t in tasks:
        e = emoji.get(t["status"], "⚪")
        lines.append(
            f"| `{t['task_id']}` | {e} {t['status']} | {t['step']} | {t['detail'][:60] if t['detail'] else '—'} |"
        )
    return "\n".join(lines)


def resume_task(task_id, max_workers):
    if not task_id or not task_id.strip():
        return "请输入任务 ID"
    task = Task(task_id.strip())
    if not os.path.exists(task.config_file):
        return f"任务 `{task_id}` 无配置文件，无法续传"
    config = task.load_config()
    config["rerun"] = False
    run_pipeline_async(task, config, workers=int(max_workers))
    return f"任务 `{task_id}` 已恢复运行（skip_completed 模式）"


# ============================================================
# Build Gradio App
# ============================================================

def build_app():
    with gr.Blocks(
        title="Datatrove 数据筛选平台",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
            radius_size=gr.themes.sizes.radius_lg,
        ),
        css=CUSTOM_CSS,
    ) as app:

        gr.HTML(
            '<div class="main-title">'
            "<h1>Datatrove 数据筛选平台</h1>"
            "<p>上传 JSONL 数据 · 配置处理流水线 · 查看分布 · 下载筛选结果</p>"
            "</div>"
        )

        current_task_id = gr.State("")

        # ====== Tab 1: Upload ======
        with gr.Tab("数据上传", id="tab-upload"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=320):
                    gr.Markdown("#### 上传方式")
                    upload_files = gr.File(
                        label="拖放或选择 JSONL 文件",
                        file_count="multiple",
                        file_types=[".jsonl"],
                        height=140,
                    )
                    dir_input = gr.Textbox(
                        label="或输入服务器目录路径",
                        placeholder="/data/corpus/jsonl_folder",
                        lines=1,
                    )
                    with gr.Row():
                        task_id_input = gr.Textbox(
                            label="任务 ID",
                            placeholder="留空自动生成，填写已有 ID 可续传",
                            scale=3,
                        )
                        upload_btn = gr.Button("创建任务", variant="primary", scale=1)

                with gr.Column(scale=3, min_width=400):
                    upload_info = gr.Markdown("等待上传...")
                    task_id_display = gr.Textbox(
                        label="当前任务 ID（复制到「流水线配置」使用）",
                        interactive=False,
                        show_copy_button=True,
                    )
                    preview_md = gr.Markdown("数据预览将在上传后显示")

            upload_btn.click(
                fn=handle_upload,
                inputs=[upload_files, dir_input, task_id_input],
                outputs=[upload_info, task_id_display, preview_md, current_task_id],
            )

        # ====== Tab 2: Pipeline ======
        with gr.Tab("流水线配置", id="tab-pipeline"):
            pipeline_task_id = gr.Textbox(
                label="任务 ID",
                placeholder="粘贴上一步获得的任务 ID",
                interactive=True,
                show_copy_button=True,
            )

            gr.Markdown("#### 选择处理步骤")

            with gr.Accordion("数据清洗（语言过滤 + 质量过滤 + MinHash 去重）", open=False):
                enable_clean = gr.Checkbox(label="启用", value=False)
                clean_languages = gr.Textbox(
                    label="目标语言",
                    value="zh",
                    placeholder="zh,en（逗号分隔）",
                    info="仅保留指定语言的文档",
                )

            with gr.Accordion("FCD 词汇难度打分（英文，纯 CPU）", open=False):
                enable_fcd = gr.Checkbox(label="启用", value=False)
                with gr.Row():
                    fcd_freq_scaling = gr.Slider(
                        0.1, 2.0, value=0.7, step=0.05,
                        label="freq_scaling_factor",
                        info="词频 sigmoid 缩放因子",
                    )
                    fcd_wf = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.05,
                        label="w_f",
                        info="词频权重（vs 概念距离）",
                    )
                    fcd_noun_weight = gr.Slider(
                        0.0, 1.0, value=0.7, step=0.05,
                        label="noun_weight",
                        info="名词难度占比",
                    )

            with gr.Accordion("CDF-GC 生成复杂度打分 + 采样（中文，需 GPU）", open=False):
                enable_gc = gr.Checkbox(label="启用", value=False)
                with gr.Row():
                    gc_language = gr.Dropdown(["zh"], value="zh", label="语言")
                    gc_sample_rate = gr.Slider(0.01, 1.0, value=0.2, step=0.01, label="采样率")
                with gr.Row():
                    gc_tokenizer_path = gr.Textbox(label="Tokenizer 路径", placeholder="/path/to/tokenizer.json")
                    gc_ltp_model_path = gr.Textbox(label="LTP 模型路径", placeholder="/path/to/LTP/small")
                with gr.Row():
                    gc_cuda_devices = gr.Textbox(label="CUDA_VISIBLE_DEVICES", value="0", placeholder="0,1,2,3")
                    gc_workers_per_gpu = gr.Number(label="Workers / GPU", value=4, precision=0)

            with gr.Accordion("采样（基于 FCD 分数，需先启用 FCD）", open=False):
                enable_sample = gr.Checkbox(label="启用", value=False)
                with gr.Row():
                    sample_method = gr.Dropdown(
                        ["cdf", "hard", "random"], value="cdf", label="采样策略",
                        info="CDF=概率采样 | Hard=取Top | Random=随机",
                    )
                    sample_rate = gr.Slider(0.01, 1.0, value=0.2, step=0.01, label="采样率")
                with gr.Row():
                    sample_unit = gr.Dropdown(["doc", "token"], value="doc", label="采样单位")
                    sample_score_source = gr.Dropdown(["fcd"], value="fcd", label="分数来源")

            with gr.Row():
                max_workers = gr.Slider(1, 64, value=8, step=1, label="并行 Workers", scale=2)
                run_btn = gr.Button("开始处理", variant="primary", size="lg", scale=1)

            run_info = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("#### 任务状态")
            with gr.Row():
                poll_btn = gr.Button("刷新状态", size="sm")
                download_btn = gr.Button("下载结果", size="sm", variant="secondary")

            status_md = gr.Markdown("点击「刷新状态」查看进度")
            log_box = gr.Textbox(label="运行日志", lines=12, max_lines=25, interactive=False)

            gr.Markdown("#### 分数分布可视化")
            with gr.Row():
                plot_score_type = gr.Dropdown(
                    ["fcd", "gc"], value="fcd", label="分数类型",
                    info="刷新状态时自动绘制，也可手动触发",
                    scale=2,
                )
                plot_btn = gr.Button("绘制分布图", size="sm", scale=1)

            score_plot = gr.Plot(label="概率密度 (PDF) & 累积分布 (CDF)")
            download_file = gr.File(label="结果压缩包", interactive=False, visible=True)

            run_btn.click(
                fn=run_task,
                inputs=[
                    pipeline_task_id,
                    enable_clean, clean_languages,
                    enable_fcd, fcd_freq_scaling, fcd_wf, fcd_noun_weight,
                    enable_gc, gc_sample_rate, gc_language, gc_tokenizer_path,
                    gc_ltp_model_path, gc_cuda_devices, gc_workers_per_gpu,
                    enable_sample, sample_method, sample_rate, sample_unit, sample_score_source,
                    max_workers,
                ],
                outputs=[run_info, log_box],
            )
            poll_btn.click(
                fn=poll_status,
                inputs=[pipeline_task_id],
                outputs=[status_md, log_box, score_plot],
            )
            plot_btn.click(
                fn=plot_scores_manual,
                inputs=[pipeline_task_id, plot_score_type],
                outputs=[score_plot],
            )
            download_btn.click(
                fn=download_result,
                inputs=[pipeline_task_id],
                outputs=[download_file],
            )

        # ====== Tab 3: Task Management ======
        with gr.Tab("任务管理", id="tab-tasks"):
            gr.Markdown("#### 历史任务")
            refresh_btn = gr.Button("刷新列表", size="sm")
            task_list_md = gr.Markdown("点击刷新查看")
            refresh_btn.click(fn=refresh_task_list, outputs=[task_list_md])

            gr.Markdown("---")
            gr.Markdown("#### 断点续传")
            gr.Markdown(
                "填入已有任务 ID 即可从上次中断处继续。"
                "框架会自动跳过已完成的步骤（`skip_completed`）。"
            )
            with gr.Row():
                resume_task_id = gr.Textbox(label="任务 ID", placeholder="粘贴任务 ID", scale=3)
                resume_workers = gr.Slider(1, 64, value=8, step=1, label="Workers", scale=2)
                resume_btn = gr.Button("恢复运行", variant="primary", scale=1)
            resume_info = gr.Markdown("")
            resume_btn.click(
                fn=resume_task,
                inputs=[resume_task_id, resume_workers],
                outputs=[resume_info],
            )

    return app


if __name__ == "__main__":
    auth_pairs = parse_auth(AUTH_USERS)
    app = build_app()
    port = int(os.environ.get("WEBUI_PORT", "7860"))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        auth=auth_pairs if auth_pairs else None,
        share=False,
    )
