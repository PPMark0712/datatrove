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
/* ── Global ── */
.gradio-container {
    max-width: 100% !important;
    padding: 0 40px !important;
}
footer { display: none !important; }

/* ── Header ── */
.hero-banner {
    text-align: center;
    padding: 32px 40px 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    margin-bottom: 20px;
}
.hero-banner h1 {
    font-size: 1.7em;
    font-weight: 700;
    color: #fff;
    margin: 0 0 6px;
    letter-spacing: -0.02em;
}
.hero-banner p {
    color: rgba(255,255,255,.8);
    font-size: 0.88em;
    margin: 0;
}

/* ── Section titles ── */
.section-title {
    font-size: 0.95em !important;
    font-weight: 600 !important;
    color: #374151 !important;
    padding-bottom: 6px;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 10px !important;
    margin-top: 4px !important;
}

/* ── Cards ── */
.card-group {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: 16px !important;
    background: #fff !important;
}

/* ── Status chips ── */
.status-card {
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
}
.status-chip {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.78em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .04em;
}
.chip-running { background: #fef3c7; color: #92400e; }
.chip-completed { background: #d1fae5; color: #065f46; }
.chip-error { background: #fee2e2; color: #991b1b; }
.chip-created { background: #dbeafe; color: #1e40af; }

/* ── Pipeline step accordion ── */
.step-accordion {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
}

/* ── Log area ── */
.log-area textarea {
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace !important;
    font-size: 0.8em !important;
    line-height: 1.5 !important;
    background: #1a1b26 !important;
    color: #a9b1d6 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

/* ── Info banner ── */
.info-banner {
    background: #f0f4ff;
    border-left: 3px solid #6366f1;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 0.88em;
    color: #1e1b4b;
    line-height: 1.5;
}

/* ── Tab bar ── */
.tab-nav {
    border-bottom: 2px solid #e5e7eb !important;
    gap: 0 !important;
}
.tab-nav button {
    font-weight: 500 !important;
    font-size: 0.9em !important;
    padding: 10px 24px !important;
    border-radius: 6px 6px 0 0 !important;
}
.tab-nav button.selected {
    font-weight: 600 !important;
    border-bottom: 2px solid #4f46e5 !important;
}

/* ── Two-column pipeline layout ── */
.pipeline-left {
    border-right: 1px solid #e5e7eb;
    padding-right: 20px !important;
}
.pipeline-right {
    padding-left: 20px !important;
}

/* ── Buttons ── */
.primary-btn {
    min-height: 40px !important;
}

/* ── Plot ── */
.plot-area {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background: #fff;
    padding: 4px;
}

/* ── Task table ── */
.task-table table {
    width: 100% !important;
    border-collapse: collapse;
}
.task-table th, .task-table td {
    padding: 8px 12px !important;
    text-align: left;
}
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
        return (
            '<div class="info-banner">&#9888;&#65039; 请上传文件或填写目录路径</div>',
            "", "等待上传...",
        )

    n_files = task.count_input_files()
    if n_files == 0:
        return (
            f'<div class="info-banner">任务 <code>{task.task_id}</code> 已创建，但未找到 .jsonl 文件</div>',
            task.task_id,
            "无数据",
        )

    n_rows = task.count_input_rows()
    preview = load_preview(task.input_dir)
    info = (
        f'<div class="info-banner">'
        f'&#9989; 任务 <b>{task.task_id}</b> 创建成功 &nbsp;·&nbsp; '
        f'{n_files} 个文件 &nbsp;·&nbsp; {n_rows:,} 行'
        f'</div>'
    )
    return info, task.task_id, preview


def load_preview(input_dir: str, max_rows: int = 10) -> str:
    rows = []
    for f in sorted(glob_module.glob(os.path.join(input_dir, "*.jsonl")))[:3]:
        with open(f, encoding="utf-8", errors="replace") as fh:
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
        return "*无数据可预览*"
    lines = ["| # | ID | Text |", "|:--|:--|:--|"]
    for i, r in enumerate(rows):
        lines.append(f"| {i+1} | `{r['id']}` | {r['text']} |")
    lines.append(f"\n*显示前 {len(rows)} 条*")
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
        return '<div class="info-banner">&#9888;&#65039; 请先在「数据上传」页创建任务</div>', ""

    task = Task(task_id.strip())
    if not os.path.exists(task.input_dir):
        return f'<div class="info-banner">&#9888;&#65039; 任务 <code>{task_id}</code> 不存在</div>', ""

    status = task.get_status()
    if status.get("status") == "running":
        return f'<div class="info-banner">&#9888;&#65039; 任务正在运行中，请等待完成</div>', task.get_log()

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

    if enable_sample and not enable_fcd:
        return '<div class="info-banner">&#9888;&#65039; 采样需要先启用 FCD 打分</div>', ""

    if enable_sample and enable_fcd:
        config["sample"] = {
            "method": sample_method,
            "sample_rate": sample_rate,
            "unit": sample_unit,
            "score_source": sample_score_source,
        }

    if not steps:
        return '<div class="info-banner">&#9888;&#65039; 请至少选择一个处理步骤</div>', ""

    config["steps"] = steps
    run_pipeline_async(task, config, workers=int(max_workers))
    step_names = {"data_clean": "清洗", "fcd": "FCD", "cdf_gc": "GC"}
    display_steps = " → ".join(step_names.get(s, s) for s in steps)
    if enable_sample and enable_fcd:
        display_steps += f" → {sample_method.upper()} 采样"
    return (
        f'<div class="info-banner">'
        f'&#128640; 任务 <b>{task_id}</b> 已启动 &nbsp;·&nbsp; {display_steps}'
        f'</div>'
    ), ""


def poll_status(task_id):
    if not task_id or not task_id.strip():
        return '<div class="info-banner" style="opacity:.5">请输入任务 ID</div>', "", None
    task = Task(task_id.strip())
    status = task.get_status()
    s = status.get("status", "unknown")
    step = status.get("step", "")
    detail = status.get("detail", "")

    chip_class = {
        "created": "chip-created", "running": "chip-running",
        "completed": "chip-completed", "error": "chip-error",
    }.get(s, "")

    html = f'<div class="status-card"><span class="status-chip {chip_class}">{s}</span>'
    if step:
        html += f'&nbsp;&nbsp; 当前步骤: <b>{step}</b>'
    if detail:
        html += f'<br><span style="color:#6b7280;font-size:0.85em;margin-top:4px;display:inline-block">{detail}</span>'
    html += '</div>'

    log_tail = task.get_log()
    if log_tail and len(log_tail) > 8000:
        log_tail = "...(truncated)...\n" + log_tail[-8000:]

    fig = try_plot_scores(task)
    return html, log_tail or "", fig


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
    if not scores:
        return None
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
        return '*暂无任务记录*'
    lines = ["| 状态 | 任务 ID | 当前步骤 | 详情 |", "|:--|:--|:--|:--|"]
    for t in tasks:
        s = t["status"]
        chip = {"created": "🔵", "running": "🟡", "completed": "🟢", "error": "🔴"}.get(s, "⚪")
        detail = (t["detail"][:60] + "…") if t["detail"] and len(t["detail"]) > 60 else (t["detail"] or "—")
        lines.append(f"| {chip} {s} | `{t['task_id']}` | {t['step'] or '—'} | {detail} |")
    return "\n".join(lines)


def resume_task(task_id, max_workers):
    if not task_id or not task_id.strip():
        return '<div class="info-banner">&#9888;&#65039; 请输入任务 ID</div>'
    task = Task(task_id.strip())
    if not os.path.exists(task.config_file):
        return f'<div class="info-banner">&#9888;&#65039; 任务 <code>{task_id}</code> 无配置文件</div>'
    config = task.load_config()
    config["rerun"] = False
    run_pipeline_async(task, config, workers=int(max_workers))
    return f'<div class="info-banner">&#128260; 任务 <b>{task_id}</b> 已恢复运行（skip_completed 模式）</div>'


# ============================================================
# Build Gradio App
# ============================================================

def build_app():
    with gr.Blocks(title="Datatrove 数据筛选平台") as app:

        # ── Header ──
        gr.HTML(
            '<div class="hero-banner">'
            "<h1>Datatrove 数据筛选平台</h1>"
            "<p>上传数据 · 配置流水线 · 查看分布 · 下载结果</p>"
            "</div>"
        )

        # ====== Tab 1: Upload ======
        with gr.Tab("📂  数据上传", id="tab-upload"):
            with gr.Row(equal_height=True):
                # Left: upload controls
                with gr.Column(scale=2, min_width=360):
                    gr.Markdown('<p class="section-title">上传方式</p>')
                    with gr.Group(elem_classes="card-group"):
                        upload_files = gr.File(
                            label="拖放或选择 JSONL 文件",
                            file_count="multiple",
                            file_types=[".jsonl"],
                            height=140,
                        )
                        gr.Markdown("&nbsp;", visible=True)
                        dir_input = gr.Textbox(
                            label="或填写服务器本地路径",
                            placeholder="/data/corpus/my_dataset/",
                            lines=1,
                        )
                    gr.Markdown('<p class="section-title">任务设置</p>')
                    with gr.Group(elem_classes="card-group"):
                        task_id_input = gr.Textbox(
                            label="任务 ID（留空自动生成）",
                            placeholder="填写已有 ID 可恢复任务",
                        )
                        upload_btn = gr.Button("创建任务", variant="primary", size="lg", elem_classes="primary-btn")

                # Right: result & preview
                with gr.Column(scale=3, min_width=480):
                    gr.Markdown('<p class="section-title">任务信息</p>')
                    upload_info = gr.HTML(
                        '<div class="info-banner" style="opacity:.4">等待上传数据…</div>'
                    )
                    task_id_display = gr.Textbox(label="任务 ID", interactive=False)

                    gr.Markdown('<p class="section-title">数据预览</p>')
                    preview_md = gr.Markdown("*上传后展示前 10 条*", elem_classes="task-table")

            upload_btn.click(
                fn=handle_upload,
                inputs=[upload_files, dir_input, task_id_input],
                outputs=[upload_info, task_id_display, preview_md],
            )

        # ====== Tab 2: Pipeline ======
        with gr.Tab("⚙️  流水线配置", id="tab-pipeline"):
            with gr.Row(equal_height=False):
                # ── Left column: config ──
                with gr.Column(scale=1, min_width=480, elem_classes="pipeline-left"):
                    with gr.Group(elem_classes="card-group"):
                        pipeline_task_id = gr.Textbox(
                            label="任务 ID",
                            placeholder="粘贴「数据上传」中获得的任务 ID",
                        )

                    gr.Markdown('<p class="section-title">处理步骤</p>')

                    with gr.Accordion("🧹 数据清洗", open=False, elem_classes="step-accordion"):
                        enable_clean = gr.Checkbox(label="启用", value=False)
                        clean_languages = gr.Textbox(
                            label="目标语言", value="zh",
                            placeholder="zh,en（逗号分隔）",
                            info="语言过滤 + 质量过滤 + MinHash 去重",
                        )

                    with gr.Accordion("📊 FCD 词汇难度打分", open=False, elem_classes="step-accordion"):
                        enable_fcd = gr.Checkbox(label="启用", value=False)
                        with gr.Row():
                            fcd_freq_scaling = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="freq_scaling_factor")
                            fcd_wf = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="w_f")
                            fcd_noun_weight = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="noun_weight")

                    with gr.Accordion("🧠 CDF-GC 生成复杂度", open=False, elem_classes="step-accordion"):
                        enable_gc = gr.Checkbox(label="启用（需 GPU）", value=False)
                        with gr.Row():
                            gc_language = gr.Dropdown(["zh"], value="zh", label="语言")
                            gc_sample_rate = gr.Slider(0.01, 1.0, value=0.2, step=0.01, label="采样率")
                        with gr.Row():
                            gc_tokenizer_path = gr.Textbox(label="Tokenizer 路径", placeholder="/path/to/tokenizer.json")
                            gc_ltp_model_path = gr.Textbox(label="LTP 模型路径", placeholder="/path/to/LTP/small")
                        with gr.Row():
                            gc_cuda_devices = gr.Textbox(label="CUDA_VISIBLE_DEVICES", value="0")
                            gc_workers_per_gpu = gr.Number(label="Workers / GPU", value=4, precision=0)

                    with gr.Accordion("🎲 采样", open=False, elem_classes="step-accordion"):
                        enable_sample = gr.Checkbox(label="启用（需先启用 FCD）", value=False)
                        with gr.Row():
                            sample_method = gr.Dropdown(
                                ["cdf", "hard", "random"], value="cdf", label="策略",
                                info="CDF: 概率递增 · Hard: Top-K · Random: 均匀",
                            )
                            sample_rate = gr.Slider(0.01, 1.0, value=0.2, step=0.01, label="采样率")
                        with gr.Row():
                            sample_unit = gr.Dropdown(["doc", "token"], value="doc", label="单位")
                            sample_score_source = gr.Dropdown(["fcd"], value="fcd", label="分数来源")

                    with gr.Row():
                        max_workers = gr.Slider(1, 64, value=8, step=1, label="并行 Workers", scale=3)
                        run_btn = gr.Button("▶  开始处理", variant="primary", size="lg", scale=2, elem_classes="primary-btn")

                    run_info = gr.HTML("")

                # ── Right column: status & results ──
                with gr.Column(scale=1, min_width=480, elem_classes="pipeline-right"):
                    gr.Markdown('<p class="section-title">任务状态</p>')
                    with gr.Row():
                        poll_btn = gr.Button("刷新状态", variant="secondary", size="sm", scale=1)
                        download_btn = gr.Button("下载结果", variant="secondary", size="sm", scale=1)

                    status_md = gr.HTML(
                        '<div class="info-banner" style="opacity:.4">点击「刷新状态」查看进度</div>'
                    )
                    log_box = gr.Textbox(
                        label="运行日志", lines=12, max_lines=24,
                        interactive=False, elem_classes="log-area",
                    )

                    gr.Markdown('<p class="section-title">分数分布</p>')
                    with gr.Row():
                        plot_score_type = gr.Dropdown(["fcd", "gc"], value="fcd", label="类型", scale=2)
                        plot_btn = gr.Button("绘制", size="sm", scale=1)

                    score_plot = gr.Plot(label="PDF & CDF", elem_classes="plot-area")
                    download_file = gr.File(label="结果压缩包", interactive=False)

            # ── Events ──
            run_btn.click(
                fn=run_task,
                inputs=[
                    pipeline_task_id,
                    enable_clean, clean_languages,
                    enable_fcd, fcd_freq_scaling, fcd_wf, fcd_noun_weight,
                    enable_gc, gc_sample_rate, gc_language, gc_tokenizer_path,
                    gc_ltp_model_path, gc_cuda_devices, gc_workers_per_gpu,
                    enable_sample, sample_method, sample_rate, sample_unit,
                    sample_score_source, max_workers,
                ],
                outputs=[run_info, log_box],
            )
            poll_btn.click(fn=poll_status, inputs=[pipeline_task_id], outputs=[status_md, log_box, score_plot])
            plot_btn.click(fn=plot_scores_manual, inputs=[pipeline_task_id, plot_score_type], outputs=[score_plot])
            download_btn.click(fn=download_result, inputs=[pipeline_task_id], outputs=[download_file])

        # ====== Tab 3: Task Management ======
        with gr.Tab("📋  任务管理", id="tab-tasks"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=500):
                    gr.Markdown('<p class="section-title">历史任务</p>')
                    refresh_btn = gr.Button("刷新列表", variant="secondary", size="sm")
                    task_list_md = gr.Markdown("*点击刷新查看所有任务*", elem_classes="task-table")
                    refresh_btn.click(fn=refresh_task_list, outputs=[task_list_md])

                with gr.Column(scale=2, min_width=400):
                    gr.Markdown('<p class="section-title">断点续传</p>')
                    gr.HTML(
                        '<div class="info-banner" style="margin-bottom:10px">'
                        '填入已有任务 ID，自动跳过已完成步骤继续执行。'
                        '</div>'
                    )
                    with gr.Group(elem_classes="card-group"):
                        resume_task_id = gr.Textbox(label="任务 ID", placeholder="粘贴任务 ID")
                        resume_workers = gr.Slider(1, 64, value=8, step=1, label="Workers")
                        resume_btn = gr.Button("恢复运行", variant="primary", elem_classes="primary-btn")
                    resume_info = gr.HTML("")
                    resume_btn.click(fn=resume_task, inputs=[resume_task_id, resume_workers], outputs=[resume_info])

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
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
            radius_size=gr.themes.sizes.radius_md,
        ),
        css=CUSTOM_CSS,
    )
