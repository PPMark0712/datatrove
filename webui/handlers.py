"""Gradio event handlers for the Datatrove WebUI."""

import os
import json
import glob as glob_module
from html import escape

import gradio as gr

from task_manager import (
    Task,
    stage_upload,
    confirm_upload,
    cancel_upload,
    rename_dataset,
    list_workspace_datasets,
    create_task_from_workspace,
    run_pipeline_async,
    list_tasks,
    load_scores,
    plot_score_distribution,
    save_output_as_dataset,
    delete_task,
)


# ── Helpers ──

def _next_dataset_name():
    datasets = list_workspace_datasets()
    nums = []
    for d in datasets:
        name = d["name"]
        prefix = "数据集"
        if name.startswith(prefix):
            try:
                nums.append(int(name[len(prefix):]))
            except ValueError:
                pass
    return f"数据集{max(nums, default=0) + 1}"


def _display_name(task_id: str) -> str:
    return Task(task_id).get_display_name()


def _task_header_html(task: Task) -> str:
    n_files = task.count_input_files()
    n_rows = task.count_input_rows() if n_files > 0 else 0
    status = task.get_status()
    s = status.get("status", "unknown")
    badge = f"badge-{s}" if s in ("created", "running", "completed", "error", "stopped") else ""
    return (
        f'<div class="task-info-card">'
        f'<span class="badge {badge}">{s}</span> &nbsp; '
        f'<b>{_display_name(task.task_id)}</b> &nbsp;·&nbsp; '
        f'{n_files} files &nbsp;·&nbsp; {n_rows:,} rows'
        f'</div>'
    )


def _status_html(task: Task) -> str:
    status = task.get_status()
    s = status.get("status", "unknown")
    step = status.get("step", "")
    detail = status.get("detail", "")
    badge = f"badge-{s}" if s in ("created", "running", "completed", "error", "stopped") else ""
    html = f'<div class="task-info-card"><span class="badge {badge}">{s}</span>'
    if step:
        html += f" &nbsp; step: <b>{step}</b>"
    if detail:
        html += f'<br><span style="color:#64748b;font-size:0.85em">{detail}</span>'
    html += "</div>"
    return html


def _preview_md(input_dir: str, max_rows: int = 8) -> str:
    rows = []
    for f in sorted(glob_module.glob(os.path.join(input_dir, "*.jsonl")))[:3]:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if len(rows) >= max_rows:
                    break
                try:
                    item = json.loads(line)
                    text = item.get("text", "")
                    if len(text) > 120:
                        text = text[:120] + "..."
                    rows.append({"id": item.get("id", ""), "text": text})
                except json.JSONDecodeError:
                    continue
    if not rows:
        return ""
    def _esc(s):
        return s.replace("|", "\\|").replace("\n", " ").replace("\r", "")
    lines = ["| # | ID | Text |", "|:--|:--|:--|"]
    for i, r in enumerate(rows):
        lines.append(f"| {i+1} | `{_esc(r['id'])}` | {_esc(r['text'])} |")
    return "\n".join(lines)


def _try_plot(task: Task):
    for stype, title in [("fcd", "FCD Score"), ("gc", "GC Score")]:
        scores = load_scores(task, stype)
        if scores:
            return plot_score_distribution(scores, title)
    return None


def _find_running_task() -> str:
    for t in list_tasks():
        if t["status"] == "running":
            return t["task_id"]
    return ""


# ═══════════════════════════════════════════
# Card rendering
# ═══════════════════════════════════════════

def render_dataset_cards(selected=None):
    datasets = list_workspace_datasets()
    html = '<div class="card-list">'
    if not datasets:
        html += '<div class="empty-hint">点击下方按钮添加数据集</div>'
    for d in datasets:
        name = escape(d["name"])
        sel = " selected" if d["name"] == selected else ""
        html += (
            f'<div class="card dataset-card{sel}" data-name="{name}">'
            f'<span class="card-name">{name}</span>'
            f'<span class="card-meta">{d["n_files"]} files</span>'
            f'</div>'
        )
    html += '</div>'
    return html


def render_task_cards(selected=None):
    tasks = list_tasks()
    icons = {"created": "⚪", "running": "🟡", "completed": "🟢", "error": "🔴", "stopped": "⚫"}
    html = '<div class="card-list">'
    if not tasks:
        html += '<div class="empty-hint">选择数据集后点击下方按钮新建任务</div>'
    for t in tasks:
        icon = icons.get(t["status"], "⚪")
        sel = " selected" if t["task_id"] == selected else ""
        display = escape(t.get("display_name", _display_name(t["task_id"])))
        html += (
            f'<div class="card task-card{sel}" data-id="{escape(t["task_id"])}">'
            f'<span class="card-icon">{icon}</span>'
            f'<span class="card-name">{display}</span>'
            f'<span class="card-delete" title="删除任务">🗑</span>'
            f'</div>'
        )
    html += '</div>'
    return html


# ═══════════════════════════════════════════
# Dataset handlers
# ═══════════════════════════════════════════

def handle_dataset_select(dataset_name):
    return dataset_name


def handle_dataset_preview(ds_name):
    if not ds_name:
        return "", "", ""
    datasets = list_workspace_datasets()
    ds = next((d for d in datasets if d["name"] == ds_name), None)
    if not ds:
        return ds_name, "", ""
    header = (
        f'<div class="task-info-card">'
        f'📂 <b>{escape(ds_name)}</b> &nbsp;·&nbsp; {ds["n_files"]} files'
        f'</div>'
    )
    preview = _preview_md(ds["path"])
    return ds_name, header, preview


def handle_dataset_rename(rename_str, current_dataset):
    if not rename_str or "|||" not in rename_str:
        return gr.update(), current_dataset
    old_name, new_name = rename_str.split("|||", 1)
    new_name = new_name.strip()
    if not new_name or old_name == new_name:
        return render_dataset_cards(selected=current_dataset), current_dataset
    rename_dataset(old_name, new_name)
    new_selected = new_name if current_dataset == old_name else current_dataset
    return render_dataset_cards(selected=new_selected), new_selected


def handle_show_add_dialog(_trigger=None):
    return gr.Group(visible=True), _next_dataset_name(), None, "", ""


def handle_file_upload(files):
    if not files:
        return None, ""
    valid_files = []
    names = []
    for f in files:
        fname = f.name if hasattr(f, "name") else str(f)
        basename = os.path.basename(fname)
        if basename.lower().endswith(('.jsonl', '.zip')):
            valid_files.append(f)
            names.append(basename)
        else:
            gr.Warning(f"不支持的文件格式: {basename}，仅支持 .jsonl 和 .zip")
    if not valid_files:
        return None, '<div style="color:#dc2626;font-size:0.82em">未找到有效文件</div>'
    tags = " ".join(
        f'<span style="background:#eef2ff;padding:2px 8px;border-radius:4px;'
        f'margin:2px;display:inline-block;font-size:0.82em;color:#475569">'
        f"📄 {escape(n)}</span>"
        for n in names
    )
    return valid_files, f"<div>{tags}</div>"


def handle_confirm_add(files, dir_path, name):
    no_change = (gr.update(),) * 9
    if not name or not name.strip():
        gr.Warning("请输入数据集名称")
        return no_change
    if files:
        staging = stage_upload(files=files)
    elif dir_path and dir_path.strip():
        staging = stage_upload(dir_path=dir_path.strip())
    else:
        gr.Warning("请上传文件或输入服务器路径")
        return no_change
    if not glob_module.glob(os.path.join(staging, "*.jsonl")):
        cancel_upload(staging)
        gr.Warning("未找到有效的 .jsonl 文件，请确保上传 .jsonl 或包含 .jsonl 文件的 .zip")
        return no_change
    final_name = confirm_upload(staging, name.strip())
    _, header, preview = handle_dataset_preview(final_name)
    return (
        gr.Group(visible=False),
        render_dataset_cards(selected=final_name),
        final_name,
        None,
        "",
        "",
        "",
        header,
        preview,
    )


def handle_cancel_add():
    return gr.Group(visible=False), None, "", "", ""


# ═══════════════════════════════════════════
# Task handlers
# ═══════════════════════════════════════════

def handle_create_task(current_dataset, current_task_id):
    if not current_dataset:
        gr.Warning("请先在左侧选择数据集")
        return (
            current_dataset,
            current_task_id,
            render_task_cards(selected=current_task_id),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )
    running = _find_running_task()
    if running:
        gr.Warning(f"{_display_name(running)} 正在运行，请等待完成或终止后再新建")
        return (
            current_dataset,
            current_task_id,
            render_task_cards(selected=current_task_id),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )
    task = create_task_from_workspace(current_dataset)
    tid = task.task_id
    return (
        current_dataset,
        tid,
        render_task_cards(selected=tid),
        _task_header_html(task),
        _status_html(task),
        "",
        _preview_md(task.input_dir),
        None,
    )


def handle_switch_task(task_choice, current_task_id):
    if not task_choice or task_choice == current_task_id:
        return (
            current_task_id,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.Group(visible=False), None, gr.update(),
        )

    if current_task_id:
        current = Task(current_task_id)
        if current.get_status().get("status") == "running":
            return (
                current_task_id,
                _task_header_html(current),
                _status_html(current),
                current.get_log() or "",
                _preview_md(current.input_dir),
                _try_plot(current),
                gr.Group(visible=True),
                task_choice,
                render_task_cards(selected=current_task_id),
            )

    task = Task(task_choice)
    log = task.get_log()
    if log and len(log) > 8000:
        log = "...(truncated)...\n" + log[-8000:]
    return (
        task_choice,
        _task_header_html(task),
        _status_html(task),
        log or "",
        _preview_md(task.input_dir),
        _try_plot(task),
        gr.Group(visible=False),
        None,
        render_task_cards(selected=task_choice),
    )


def handle_confirm_switch(pending_task, current_task_id):
    if current_task_id:
        Task(current_task_id).stop()

    if not pending_task:
        return (
            current_task_id,
            gr.update(), gr.update(), "", "", None,
            gr.Group(visible=False), None,
            render_task_cards(),
        )

    task = Task(pending_task)
    log = task.get_log()
    if log and len(log) > 8000:
        log = "...(truncated)...\n" + log[-8000:]
    return (
        pending_task,
        _task_header_html(task),
        _status_html(task),
        log or "",
        _preview_md(task.input_dir),
        _try_plot(task),
        gr.Group(visible=False),
        None,
        render_task_cards(selected=pending_task),
    )


def handle_cancel_switch(current_task_id):
    return (
        gr.Group(visible=False),
        None,
        render_task_cards(selected=current_task_id),
    )


def handle_task_delete(delete_str, current_task_id):
    if not delete_str:
        return (
            current_task_id,
            render_task_cards(selected=current_task_id),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )
    task_id = delete_str.split("|||")[0].strip()
    task = Task(task_id)
    if task.get_status().get("status") == "running":
        gr.Warning("无法删除正在运行的任务")
        return (
            current_task_id,
            render_task_cards(selected=current_task_id),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )
    delete_task(task_id)
    if current_task_id == task_id:
        return (
            "",
            render_task_cards(),
            '<div class="info-banner" style="opacity:.5">'
            "← 在左侧选择数据集，然后新建任务开始</div>",
            '<div style="color:#94a3b8;font-size:0.85em">等待任务启动...</div>',
            "",
            "",
            None,
        )
    return (
        current_task_id,
        render_task_cards(selected=current_task_id),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
    )


def handle_task_rename(rename_str, current_task_id):
    if not rename_str or "|||" not in rename_str:
        return render_task_cards(selected=current_task_id), gr.update()
    task_id, new_name = rename_str.split("|||", 1)
    new_name = new_name.strip()
    if not new_name:
        return render_task_cards(selected=current_task_id), gr.update()
    task = Task(task_id)
    task.set_display_name(new_name)
    header_update = gr.update()
    if task_id == current_task_id:
        header_update = _task_header_html(task)
    return render_task_cards(selected=current_task_id), header_update


# ═══════════════════════════════════════════
# Pipeline handlers
# ═══════════════════════════════════════════

def handle_start(
    current_task_id,
    enable_clean, clean_languages,
    enable_fcd, fcd_freq_scaling, fcd_wf, fcd_noun_weight,
    enable_gc, gc_sample_rate, gc_language, gc_tokenizer_path,
    gc_ltp_model_path, gc_cuda_devices, gc_workers_per_gpu,
    enable_sample, sample_method, sample_rate, sample_unit, sample_score_source,
    max_workers,
):
    if not current_task_id:
        return '<div class="info-banner">请先创建或选择任务</div>', "", gr.update()
    task = Task(current_task_id)
    if not os.path.exists(task.input_dir):
        return '<div class="info-banner">任务不存在</div>', "", gr.update()
    status = task.get_status()
    if status.get("status") == "running":
        return '<div class="info-banner">任务正在运行中</div>', task.get_log(), gr.update()

    running = _find_running_task()
    if running and running != current_task_id:
        return (
            f'<div class="info-banner">⚠️ {_display_name(running)} 正在运行，不允许同时运行多个任务</div>',
            "",
            gr.update(),
        )

    steps = []
    config = {}

    if enable_clean:
        steps.append("data_clean")
        config["languages"] = clean_languages if clean_languages else ["zh"]

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
        return '<div class="info-banner">采样需要先启用 FCD 打分</div>', "", gr.update()

    if enable_sample and enable_fcd:
        config["sample"] = {
            "method": sample_method,
            "sample_rate": sample_rate,
            "unit": sample_unit,
            "score_source": sample_score_source,
        }

    if not steps:
        return '<div class="info-banner">请至少选择一个处理步骤</div>', "", gr.update()

    config["steps"] = steps
    run_pipeline_async(task, config, workers=int(max_workers))
    step_names = {"data_clean": "清洗", "fcd": "FCD", "cdf_gc": "GC"}
    display = " → ".join(step_names.get(s, s) for s in steps)
    if enable_sample and enable_fcd:
        display += f" → {sample_method.upper()} 采样"
    return (
        f'<div class="info-banner">🚀 已启动: {display}</div>',
        "",
        render_task_cards(selected=current_task_id),
    )


def handle_resume(current_task_id, max_workers):
    if not current_task_id:
        return '<div class="info-banner">请先选择任务</div>', gr.update()
    task = Task(current_task_id)
    if not os.path.exists(task.config_file):
        return '<div class="info-banner">该任务无配置文件，无法续传</div>', gr.update()
    running = _find_running_task()
    if running and running != current_task_id:
        return (
            f'<div class="info-banner">⚠️ {_display_name(running)} 正在运行，不允许同时运行多个任务</div>',
            gr.update(),
        )
    config = task.load_config()
    config["rerun"] = False
    run_pipeline_async(task, config, workers=int(max_workers))
    return (
        f'<div class="info-banner">↻ {_display_name(current_task_id)} 已恢复运行</div>',
        render_task_cards(selected=current_task_id),
    )


def handle_poll(current_task_id):
    if not current_task_id:
        return '<div style="color:#94a3b8;font-size:0.85em">请选择任务</div>', "", None, gr.update()
    task = Task(current_task_id)
    log = task.get_log()
    if log and len(log) > 8000:
        log = "...(truncated)...\n" + log[-8000:]
    return _status_html(task), log or "", _try_plot(task), render_task_cards(selected=current_task_id)


def handle_plot(current_task_id, score_type):
    if not current_task_id:
        return None
    task = Task(current_task_id)
    titles = {"fcd": "FCD Score", "gc": "GC Score"}
    scores = load_scores(task, score_type)
    if not scores:
        return None
    return plot_score_distribution(scores, titles.get(score_type, "Score"))


def handle_download(current_task_id):
    if not current_task_id:
        return None
    task = Task(current_task_id)
    if task.get_status().get("status") != "completed":
        return None
    archive = task.get_output_archive()
    return archive if archive and os.path.exists(archive) else None


# ═══════════════════════════════════════════
# Save result as dataset
# ═══════════════════════════════════════════

def handle_show_save_dialog(current_task_id):
    if not current_task_id:
        return gr.Group(visible=False), "", '<div class="info-banner">请先选择任务</div>'
    task = Task(current_task_id)
    if task.get_status().get("status") != "completed":
        return gr.Group(visible=False), "", '<div class="info-banner">任务尚未完成，无法保存为数据集</div>'
    default_name = f"{_display_name(current_task_id)}_结果".replace(" ", "_")
    return gr.Group(visible=True), default_name, ""


def handle_confirm_save(current_task_id, dataset_name):
    if not current_task_id or not dataset_name:
        return (
            gr.Group(visible=False),
            gr.update(),
            gr.update(),
            '<div class="info-banner">保存失败</div>',
        )
    task = Task(current_task_id)
    final_name = save_output_as_dataset(task, dataset_name.strip())
    if not final_name:
        return (
            gr.Group(visible=False),
            gr.update(),
            gr.update(),
            '<div class="info-banner">没有可保存的输出文件</div>',
        )
    return (
        gr.Group(visible=False),
        render_dataset_cards(selected=final_name),
        final_name,
        f'<div class="info-banner">✅ 已保存为数据集: {final_name}</div>',
    )


def handle_cancel_save():
    return gr.Group(visible=False)
