"""Datatrove workbench. Component events own selection and panel state."""

import os
import zipfile
from html import escape
from pathlib import Path

import gradio as gr

from styles import CUSTOM_CSS
from preview import Preview, SCROLL_JS
from handlers import (
    config_values, dataset_choices, task_choices, next_dataset_name,
    status_html, result_view, make_config, plot_scores,
    config_controls, config_hint,
)
from task_manager import (
    Task, WORKSPACE_DIR, stage_upload, confirm_upload, cancel_upload,
    create_task_from_workspace, run_pipeline_async, rename_dataset,
    save_output_as_dataset, delete_task, list_tasks, validate_dataset_name,
)


def parse_auth(value):
    return [tuple(pair.strip().split(":", 1)) for pair in value.split(",") if ":" in pair]


def build_app():
    with gr.Blocks(title="Datatrove · 数据筛选平台") as app:
        current_dataset = gr.State("")
        current_task = gr.State("")
        current_mode = gr.State("idle")
        last_status = gr.State(None)
        drafts = gr.State({})
        gr.HTML('<header class="app-header"><strong>Datatrove</strong>'
                '<span>数据筛选与质量评估平台</span><small>导入 → 配置 → 处理 → 结果</small></header>', elem_id="brand")
        with gr.Row(elem_id="workspace", equal_height=False):
            with gr.Column(scale=0, min_width=270, elem_id="sidebar"):
                with gr.Column(elem_classes="sidebar-section"):
                    gr.HTML('<div class="sidebar-label">数据集</div>', elem_classes="section-heading")
                    with gr.Column(elem_classes="sidebar-scroll"):
                        datasets = gr.Radio(choices=dataset_choices(), label="数据集", show_label=False,
                                            elem_classes="item-list", interactive=True)
                    add_dataset = gr.Button("+ 添加数据集", elem_classes="add-button", elem_id="add-dataset")
                with gr.Column(elem_classes="sidebar-section"):
                    gr.HTML('<div class="sidebar-label">任务</div>', elem_classes="section-heading")
                    with gr.Column(elem_classes="sidebar-scroll"):
                        tasks = gr.Radio(choices=task_choices(), label="任务", show_label=False,
                                         elem_classes="item-list", interactive=True)
                    create_task = gr.Button("+ 新建任务", elem_classes="add-button", interactive=False)
                refresh_lists = gr.Button("刷新列表", size="sm")

            with gr.Column(scale=1, elem_id="main-panel"):
                with gr.Column(visible=True, elem_classes="idle-panel") as idle_panel:
                    gr.HTML('<div class="empty-state"><div class="eyebrow">DATA WORKSPACE</div>'
                            '<h2>从一个数据集开始</h2><p>选择左侧数据集查看内容，或导入 JSONL / ZIP 文件。</p>'
                            '<p>创建任务后，可进行质量过滤、去重、难度打分和采样。</p></div>')
                with gr.Column(visible=False) as dataset_panel:
                    dataset_header = gr.HTML()
                    with gr.Row(elem_classes="name-actions"):
                        dataset_new_name = gr.Textbox(label="数据集名称", scale=5, max_lines=1)
                        rename_ds = gr.Button("重命名", scale=1)
                    dataset_preview = Preview("dataset")
                    gr.Markdown("点击左侧 **新建任务**，使用此数据集的副本进行处理。")
                with gr.Column(visible=False) as task_panel:
                    task_status = gr.HTML()
                    with gr.Row(elem_classes="name-actions"):
                        task_new_name = gr.Textbox(label="任务名称", scale=5, max_lines=1)
                        rename_job = gr.Button("重命名", scale=1)
                        delete_job = gr.Button("删除任务", variant="stop", scale=1)
                    with gr.Group(visible=False) as delete_dialog:
                        gr.Markdown("删除后将移除此任务的输入副本、日志和结果。原始数据集保留。")
                        with gr.Row():
                            confirm_delete = gr.Button("确认删除", variant="stop")
                            cancel_delete = gr.Button("取消")
                    with gr.Tabs(elem_classes="task-tabs") as task_tabs:
                        with gr.Tab("配置与运行", id="config", render_children=True):
                            task_input_info = gr.Markdown()
                            with gr.Accordion("数据清洗", open=True):
                                clean = gr.Checkbox(label="启用数据清洗", value=True)
                                languages = gr.CheckboxGroup(["zh", "en"], value=["zh", "en"], label="保留语言")
                                gr.Markdown("语言识别 → Gopher / FineWeb 质量过滤 → MinHash 去重 → 隐私信息格式化")
                            with gr.Accordion("FCD · 英文词汇难度", open=False):
                                fcd = gr.Checkbox(label="启用 FCD 打分")
                                with gr.Row():
                                    freq = gr.Slider(0.1, 2, value=0.7, step=0.05, label="词频缩放")
                                    wf = gr.Slider(0, 1, value=0.5, step=0.05, label="词频权重")
                                    noun = gr.Slider(0, 1, value=0.7, step=0.05, label="名词权重")
                            with gr.Accordion("CDF-GC · 中文生成复杂度（需要 CUDA GPU）", open=False):
                                gc = gr.Checkbox(label="启用 CDF-GC 筛选")
                                gc_rate = gr.Slider(0.01, 1, value=0.2, step=0.01, label="采样率")
                                tokenizer = gr.Textbox(label="Tokenizer 文件路径")
                                ltp = gr.Textbox(label="LTP 模型目录")
                                with gr.Row():
                                    cuda = gr.Textbox(label="CUDA 设备", value="0")
                                    gpu_workers = gr.Number(label="每张 GPU 的进程数", value=4, minimum=1, precision=0)
                            with gr.Accordion("采样 · 基于 FCD 打分", open=False):
                                sample = gr.Checkbox(label="启用采样")
                                with gr.Row():
                                    method = gr.Dropdown(["cdf", "hard", "random"], value="cdf", label="采样策略")
                                    sample_rate = gr.Slider(0.01, 1, value=0.2, step=0.01, label="采样率")
                                    unit = gr.Dropdown(["doc"], value="doc", label="采样单位（文档）")
                            workers = gr.Slider(1, 32, value=4, step=1, label="最大工作进程数")
                            config_feedback = gr.Markdown(config_hint(config_values()), elem_classes="config-feedback")
                            gr.Markdown("重新运行会替换旧结果；继续运行沿用上次配置和已完成阶段。")
                            with gr.Row(elem_classes="run-actions"):
                                start = gr.Button("开始处理", variant="primary")
                                resume = gr.Button("继续运行")
                                stop = gr.Button("停止", variant="stop", interactive=False)
                        with gr.Tab("结果与分析", id="results", render_children=True):
                            result_summary = gr.Markdown()
                            with gr.Row():
                                download = gr.Button("下载结果 ZIP")
                                save = gr.Button("保存为数据集")
                            download_file = gr.File(label="下载文件", interactive=False, visible=False)
                            with gr.Group(visible=False) as save_dialog:
                                save_name = gr.Textbox(label="新数据集名称")
                                with gr.Row():
                                    confirm_save = gr.Button("确认保存", variant="primary")
                                    cancel_save = gr.Button("取消")
                            result_stats = gr.Dataframe(headers=["阶段", "记录数"], datatype=["str", "number"],
                                                       interactive=False, label="处理统计", visible=False,
                                                       elem_classes="stats-table")
                            result_preview = Preview("result", visible=False)
                            with gr.Accordion("分数分布", open=False, visible=False) as score_section:
                                score_type = gr.Dropdown(
                                    [("FCD", "fcd"), ("GC · 词性熵", "gc:pos_ent"),
                                     ("GC · 成分熵", "gc:con_ent"), ("GC · 依存熵", "gc:dep_ent"),
                                     ("GC · 依存树高度", "gc:avg_dep_height"), ("GC · 依存距离", "gc:avg_dep_dis")],
                                    value="fcd", label="分数类型（GC 展示归一化分量）")
                                plot = gr.Button("绘制分布")
                                score_plot = gr.Plot()
                        with gr.Tab("运行日志", id="logs", render_children=True):
                            gr.Markdown("每 3 秒自动更新运行状态与最近日志。切换任务不会终止正在运行的任务。")
                            logs = gr.Textbox(label="运行日志", lines=20, max_lines=24, interactive=False, elem_classes="log-area")
                    refresh_task = gr.Button("刷新状态与结果", size="sm")
        # Modal is a sibling of the workspace, never a child of a clipped sidebar.
        with gr.Column(visible=False, elem_id="dataset-modal") as add_dialog:
            with gr.Column(elem_id="dataset-dialog-card", min_width=0):
                gr.Markdown("### 添加数据集")
                upload = gr.File(label="上传 JSONL 或 ZIP（支持多个文件）", file_count="multiple",
                                 file_types=[".jsonl", ".zip"], type="filepath", height=120)
                directory = gr.Textbox(label="或使用服务器目录", placeholder="/data/corpus")
                new_name = gr.Textbox(label="数据集名称", elem_id="new-dataset-name")
                gr.Markdown("每行一个 JSON 对象，必须包含非空的 `text` 字符串。")
                with gr.Row():
                    confirm_add = gr.Button("确认导入", variant="primary")
                    cancel_add = gr.Button("取消", elem_id="cancel-dataset")
        timer = gr.Timer(3)
        all_score_choices = score_type.choices

        def score_options(task):
            steps = task.load_config().get("steps", []) if task else []
            choices = [choice for choice in all_score_choices
                       if ("fcd" in steps and choice[1] == "fcd")
                       or ("cdf_gc" in steps and choice[1].startswith("gc:"))]
            return gr.update(choices=choices, value=choices[0][1] if choices else None)

        config_inputs = [clean, languages, fcd, freq, wf, noun, gc, gc_rate, tokenizer, ltp,
                         cuda, gpu_workers, sample, method, sample_rate, unit, workers]
        selection = [current_dataset, current_task, current_mode]
        view_outputs = [
            datasets, tasks, create_task, idle_panel, dataset_panel, task_panel,
            dataset_header, dataset_new_name,
            task_status, task_new_name, task_input_info, logs, result_summary, result_stats,
            download_file, score_plot, delete_dialog, save_dialog, task_tabs,
            start, resume, stop, download, save, last_status, config_feedback, score_section, delete_job, score_type,
        ] + config_inputs
        view_outputs += [dataset_preview.section] + dataset_preview.outputs
        view_outputs += [result_preview.section] + result_preview.outputs

        def render(ds, tid, mode, task_drafts):
            task = Task(tid) if tid else None
            state = task.get_status() if task else {}
            busy = state.get("status") in ("running", "stopping")
            done = state.get("status") == "completed"
            dataset_path = Path(WORKSPACE_DIR) / ds if ds else None
            header = ""
            if ds and mode == "dataset" and dataset_path.is_dir():
                count = len(list(dataset_path.glob("*.jsonl")))
                header = f'<h2>{escape(ds)}</h2><p class="muted">{count} 个文件 · JSONL 数据集</p>'
            summary, stats, output = result_view(task) if task else ("", [], 0)
            values = task_drafts.get(tid, config_values(task))
            controls = config_controls(values, busy)
            scored = done and bool(set(task.load_config().get("steps", [])) & {"fcd", "cdf_gc"})
            return [
                gr.update(choices=dataset_choices(), value=ds if mode == "dataset" else None),
                gr.update(choices=task_choices(), value=tid or None),
                gr.update(interactive=bool(ds)),
                gr.update(visible=mode == "idle"), gr.update(visible=mode == "dataset"),
                gr.update(visible=mode == "task"), header, ds,
                status_html(task) if task else "", task.get_display_name() if task else "",
                f"输入副本：**{task.count_input_rows():,} 条** · {task.count_input_files()} 个文件" if task else "",
                task.get_log() if task else "", summary,
                gr.update(value=stats, visible=done),
                gr.update(value=None, visible=False), None, gr.update(visible=False), gr.update(visible=False),
                gr.skip(),
                gr.update(value="开始处理" if state.get("status") == "created" else "重新运行",
                          interactive=bool(task) and not busy),
                gr.update(interactive=bool(task) and state.get("status") in ("error", "stopped")),
                gr.update(interactive=state.get("status") == "running"),
                gr.update(interactive=done), gr.update(interactive=done and bool(output)), state,
                config_hint(values), gr.update(visible=scored), gr.update(interactive=not busy), score_options(task),
            ] + [dict(control, value=value) for value, control in zip(values, controls)] + dataset_preview.reset() + result_preview.reset(bool(output))

        def show_view(event):
            event = event.success(render, selection + [drafts], view_outputs,
                                  show_progress="hidden", concurrency_id="view-state")
            # Mount the task panel before selecting a tab; switching a tab while
            # its parent is hidden leaves Gradio's lazy tab body unmounted.
            return event.success(
                lambda tid: gr.update(selected="results" if tid and Task(tid).get_status().get("status") == "completed" else "config"),
                current_task, task_tabs, show_progress="hidden",
            ).success(
                fn=None, js="() => document.getElementById('main-panel')?.scrollTo({top: 0})",
            )

        show_view(datasets.input(lambda ds: (ds or "", "", "dataset" if ds else "idle"),
                                 datasets, selection, show_progress="hidden", concurrency_id="view-state"))
        show_view(tasks.input(lambda tid, ds: (ds, tid or "", "task" if tid else "idle"),
                              [tasks, current_dataset], selection, show_progress="hidden", concurrency_id="view-state"))

        def remember_config(tid, task_drafts, *values):
            busy = bool(tid) and Task(tid).get_status().get("status") in ("running", "stopping")
            if tid and not busy:
                task_drafts = {**task_drafts, tid: list(values)}
            return [task_drafts, config_hint(values)] + config_controls(values, busy)

        for control in config_inputs:
            control.input(remember_config, [current_task, drafts] + config_inputs,
                          [drafts, config_feedback] + config_inputs,
                          show_progress="hidden", concurrency_id="view-state")

        def new_task(ds):
            try:
                task = create_task_from_workspace(ds)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            return ds, task.task_id, "task"

        show_view(create_task.click(new_task, current_dataset, selection))
        refresh_lists.click(lambda: (gr.update(choices=dataset_choices()), gr.update(choices=task_choices())),
                            outputs=[datasets, tasks], show_progress="hidden")
        add_dataset.click(lambda: (gr.update(visible=True), None, "", next_dataset_name()),
                          outputs=[add_dialog, upload, directory, new_name], show_progress="hidden")
        cancel_add.click(lambda: (gr.update(visible=False), None, ""),
                         outputs=[add_dialog, upload, directory], show_progress="hidden")

        def import_dataset(files, path, name):
            staging = None
            try:
                validate_dataset_name(name)
                if files and path and path.strip():
                    raise ValueError("请只选择一种导入方式：上传文件，或填写服务器目录")
                staging = stage_upload(files, path)
                name = confirm_upload(staging, name)
            except (ValueError, OSError, UnicodeError, zipfile.BadZipFile) as exc:
                cancel_upload(staging)
                raise gr.Error(str(exc)) from exc
            gr.Info(f"已导入数据集：{name}")
            return name, "", "dataset", gr.update(visible=False), None, ""

        show_view(confirm_add.click(import_dataset, [upload, directory, new_name],
                                    selection + [add_dialog, upload, directory]))

        def rename_data(ds, name, tid, mode):
            try:
                if not rename_dataset(ds, name) and ds != name.strip():
                    raise ValueError("重命名失败：名称已存在或原数据集不存在")
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            return name.strip(), tid, mode

        show_view(rename_ds.click(rename_data, [current_dataset, dataset_new_name, current_task, current_mode], selection))

        def rename_task(tid, name):
            if not name.strip():
                raise gr.Error("任务名称不能为空")
            Task(tid).set_display_name(name)
            gr.Info("任务已重命名")
            return status_html(Task(tid)), gr.update(choices=task_choices(), value=tid), name.strip()

        rename_job.click(rename_task, [current_task, task_new_name], [task_status, tasks, task_new_name])
        delete_job.click(lambda: gr.update(visible=True), outputs=delete_dialog)
        cancel_delete.click(lambda: gr.update(visible=False), outputs=delete_dialog)

        def remove_task(tid, ds):
            if Task(tid).get_status().get("status") in ("running", "stopping"):
                raise gr.Error("请先停止任务，再删除")
            delete_task(tid)
            return ds, "", "dataset" if ds else "idle"

        show_view(confirm_delete.click(remove_task, [current_task, current_dataset], selection))

        def start_task(tid, *values):
            if not tid:
                raise gr.Error("请先选择任务")
            config = make_config(*values)
            try:
                run_pipeline_async(Task(tid), config, config["workers"])
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            gr.Info("任务已启动，可在运行日志中查看进度")

        show_view(start.click(start_task, [current_task] + config_inputs))

        def resume_task(tid):
            task = Task(tid)
            config = task.load_config()
            if not config or task.get_status().get("status") not in ("error", "stopped"):
                raise gr.Error("只有失败或停止的任务可以继续运行")
            config["rerun"] = False
            try:
                run_pipeline_async(task, config, config.get("workers", 4))
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

        show_view(resume.click(resume_task, current_task))
        show_view(stop.click(lambda tid: Task(tid).stop(), current_task))
        def poll(tid, previous, task_drafts, force=False):
            unchanged = (gr.skip(),) * (18 + len(config_inputs))
            # Refresh the task list even when browsing a dataset, so jobs don't
            # appear stuck in "running" until they are selected again.
            choices = gr.update(choices=task_choices())
            if not tid:
                return unchanged[:2] + (choices,) + unchanged[3:]
            task = Task(tid)
            state = task.get_status()
            busy = state.get("status") in ("running", "stopping")
            if state == previous and not busy and not force:
                return unchanged[:2] + (choices,) + unchanged[3:]
            done = state.get("status") == "completed"
            summary, stats, output = result_view(task)
            values = task_drafts.get(tid, config_values(task))
            scored = done and bool(set(task.load_config().get("steps", [])) & {"fcd", "cdf_gc"})
            return (
                status_html(task), task.get_log(), choices,
                summary, gr.update(value=stats, visible=done), gr.update(visible=bool(output)),
                gr.update(value="开始处理" if state.get("status") == "created" else "重新运行",
                          interactive=not busy),
                gr.update(interactive=state.get("status") in ("error", "stopped")),
                gr.update(interactive=state.get("status") == "running"),
                gr.update(interactive=done), gr.update(interactive=done and bool(output)), state,
                gr.update(selected="results") if done and previous and previous.get("status") in ("running", "stopping") else gr.skip(),
                gr.update(choices=dataset_choices()), config_hint(values), gr.update(visible=scored),
                gr.update(interactive=not busy), score_options(task) if state != previous else gr.skip(),
            ) + tuple(config_controls(values, busy))

        poll_outputs = [task_status, logs, tasks, result_summary, result_stats, result_preview.section,
                        start, resume, stop, download, save, last_status, task_tabs, datasets,
                        config_feedback, score_section, delete_job, score_type] + config_inputs
        timer.tick(poll, [current_task, last_status, drafts], poll_outputs,
                   show_progress="hidden", concurrency_id="view-state")
        refresh_task.click(lambda tid, previous, saved: poll(tid, previous, saved, force=True),
                           [current_task, last_status, drafts], poll_outputs,
                           show_progress="hidden", concurrency_id="view-state")

        dataset_preview.bind(
            lambda ds, tid, mode: Path(WORKSPACE_DIR) / ds if ds and mode == "dataset" else None, selection)
        result_preview.bind(
            lambda ds, tid, mode: Task(tid).output_dir
            if tid and mode == "task" and Task(tid).get_status().get("status") == "completed" else None, selection)
        app.load(fn=None, js=SCROLL_JS)

        def download_result(tid):
            task = Task(tid)
            if task.get_status().get("status") != "completed":
                raise gr.Error("任务尚未完成")
            archive = task.get_output_archive()
            if not archive:
                raise gr.Error("没有可下载的结果")
            return gr.update(value=archive, visible=True)

        download.click(download_result, current_task, download_file)
        save.click(lambda tid: (gr.update(visible=True), Task(tid).get_display_name() + "_结果"),
                   current_task, [save_dialog, save_name])
        cancel_save.click(lambda: gr.update(visible=False), outputs=save_dialog)

        def save_result(tid, name):
            if Task(tid).get_status().get("status") != "completed":
                raise gr.Error("任务尚未完成")
            try:
                result = save_output_as_dataset(Task(tid), name)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            if not result:
                raise gr.Error("输出为 0 条，没有可保存的数据")
            gr.Info(f"已保存为数据集：{result}")
            return result, "", "dataset"

        show_view(confirm_save.click(save_result, [current_task, save_name], selection))
        plot.click(plot_scores, [current_task, score_type], score_plot)
        app.load(fn=None, js="""() => {
            if (window.__datasetDialogKeys) return;
            window.__datasetDialogKeys = true;
            let opened = false;
            const syncDialog = () => {
                const modal = document.getElementById('dataset-modal');
                const visible = !!modal?.getClientRects().length;
                if (visible === opened) return;
                opened = visible;
                const workspace = document.getElementById('workspace');
                if (workspace) workspace.inert = visible;
                if (visible) {
                    modal.setAttribute('role', 'dialog');
                    modal.setAttribute('aria-modal', 'true');
                    modal.setAttribute('aria-label', '添加数据集');
                    requestAnimationFrame(() => {
                        document.querySelector('#new-dataset-name textarea, #new-dataset-name input')?.focus();
                    });
                } else {
                    document.getElementById('add-dataset')?.focus();
                }
            };
            new MutationObserver(syncDialog).observe(document.body, {
                childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'class']
            });
            syncDialog();
            document.addEventListener('keydown', (event) => {
                const modal = document.getElementById('dataset-modal');
                if (!modal || !modal.getClientRects().length) return;
                if (event.key === 'Escape') {
                    event.preventDefault();
                    document.getElementById('cancel-dataset')?.click();
                }
                if (event.key === 'Tab') {
                    const items = [...modal.querySelectorAll('button, input, textarea, [tabindex="0"]')]
                        .filter(el => !el.disabled && el.getClientRects().length);
                    const first = items[0], last = items[items.length - 1];
                    if (!modal.contains(document.activeElement) ||
                        (!event.shiftKey && document.activeElement === last)) {
                        event.preventDefault(); first?.focus();
                    } else if (event.shiftKey && document.activeElement === first) {
                        event.preventDefault(); last?.focus();
                    }
                }
            });
        }""")
    return app


if __name__ == "__main__":
    # The worker belongs to this server process; stale running states cannot resume
    # by themselves after a server restart.
    for task in list_tasks():
        if task["status"] in ("running", "stopping"):
            Task(task["task_id"]).set_status("stopped", detail="服务已重启，可继续运行任务")
    build_app().launch(
        server_name=os.environ.get("WEBUI_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("WEBUI_PORT", "7860")),
        auth=parse_auth(os.environ.get("WEBUI_AUTH", "admin:datatrove2024")) or None,
        auth_message="Datatrove 数据筛选平台 · 请登录",
        share=False, css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate",
                             font=["system-ui", "sans-serif"]),
    )
