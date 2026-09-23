"""Datatrove WebUI — card-based sidebar with inline rename."""

import os
import gradio as gr

from styles import CUSTOM_CSS
from handlers import (
    render_dataset_cards,
    render_task_cards,
    handle_dataset_rename,
    handle_dataset_preview,
    handle_show_add_dialog,
    handle_file_upload,
    handle_confirm_add,
    handle_cancel_add,
    handle_create_task,
    handle_switch_task,
    handle_confirm_switch,
    handle_cancel_switch,
    handle_task_delete,
    handle_task_rename,
    handle_start,
    handle_resume,
    handle_poll,
    handle_plot,
    handle_download,
    handle_show_save_dialog,
    handle_confirm_save,
    handle_cancel_save,
)

AUTH_USERS = os.environ.get("WEBUI_AUTH", "admin:datatrove2024")


def parse_auth(auth_str: str):
    pairs = {}
    for pair in auth_str.split(","):
        if ":" in pair:
            u, p = pair.split(":", 1)
            pairs[u.strip()] = p.strip()
    return pairs


def _set_bridge(elem_id):
    """JS IIFE factory: returns (value)=>void that sets a hidden Textbox to trigger .input()."""
    return (
        f"(function(v){{ "
        f"var c = document.getElementById('{elem_id}'); "
        f"if (!c) {{ console.warn('[bridge] #{elem_id} not found'); return; }} "
        f"var inp = c.querySelector('textarea') || c.querySelector('input'); "
        f"if (!inp) {{ console.warn('[bridge] no input in #{elem_id}'); return; }} "
        f"var proto = inp.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype; "
        f"var desc = Object.getOwnPropertyDescriptor(proto, 'value'); "
        f"if (desc && desc.set) {{ desc.set.call(inp, v); }} else {{ inp.value = v; }} "
        f"inp.dispatchEvent(new Event('input', {{bubbles: true}})); "
        f"inp.dispatchEvent(new Event('change', {{bubbles: true}})); "
        f"console.log('[bridge] #{elem_id} =', v); "
        f"}})"
    )


GLOBAL_JS = """
() => {
    console.log('[WebUI] Global JS v2 loaded');

    var setBridge = function(elemId, v) {
        var c = document.getElementById(elemId);
        if (!c) { console.warn('[bridge] #' + elemId + ' not found'); return; }
        var inp = c.querySelector('textarea') || c.querySelector('input');
        if (!inp) { console.warn('[bridge] no input in #' + elemId); return; }
        var proto = inp.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        var desc = Object.getOwnPropertyDescriptor(proto, 'value');
        if (desc && desc.set) { desc.set.call(inp, v); } else { inp.value = v; }
        inp.dispatchEvent(new Event('input', {bubbles: true}));
        inp.dispatchEvent(new Event('change', {bubbles: true}));
        console.log('[bridge] #' + elemId + ' =', v);
    };

    // ── Panel switching ──
    window.switchPanel = function(mode) {
        ['idle', 'dataset', 'task'].forEach(function(m) {
            document.querySelectorAll('.panel-' + m).forEach(function(el) {
                if (m === mode) el.classList.remove('panel-hidden');
                else el.classList.add('panel-hidden');
            });
        });
        console.log('[WebUI] switchPanel:', mode);
    };

    // ── Dataset card clicks ──
    document.addEventListener('click', function(e) {
        var card = e.target.closest('.dataset-card');
        if (!card) return;
        var wasSelected = card.classList.contains('selected');
        var isNameClick = e.target.closest('.card-name') !== null;
        card.closest('.card-list').querySelectorAll('.dataset-card').forEach(function(c) { c.classList.remove('selected'); });
        card.classList.add('selected');

        if (wasSelected && isNameClick) {
            var nameEl = card.querySelector('.card-name');
            if (nameEl && !nameEl.isContentEditable) {
                nameEl.contentEditable = 'true';
                nameEl.dataset.original = nameEl.textContent;
                nameEl.focus();
                var r = document.createRange(); r.selectNodeContents(nameEl);
                window.getSelection().removeAllRanges(); window.getSelection().addRange(r);
            }
        } else {
            window.__selectedDataset = card.dataset.name;
            setBridge('ds-bridge-box', card.dataset.name);
            window.switchPanel('dataset');
        }
    });

    // ── Dataset rename (keydown + focusout on card-name inside dataset-card) ──
    document.addEventListener('keydown', function(e) {
        if (!e.target.closest('.dataset-card') || !e.target.classList.contains('card-name') || !e.target.isContentEditable) return;
        if (e.key === 'Enter') {
            e.preventDefault(); e.target.contentEditable = 'false';
            var o = e.target.dataset.original, n = e.target.textContent.trim();
            if (n && n !== o) { setBridge('ds-rename-bridge', o + '|||' + n); }
            else { e.target.textContent = o; }
        } else if (e.key === 'Escape') {
            e.target.textContent = e.target.dataset.original; e.target.contentEditable = 'false';
        }
    });
    document.addEventListener('focusout', function(e) {
        if (!e.target.closest('.dataset-card') || !e.target.classList.contains('card-name') || !e.target.isContentEditable) return;
        var o = e.target.dataset.original, n = e.target.textContent.trim();
        e.target.contentEditable = 'false';
        if (n && n !== o) { setBridge('ds-rename-bridge', o + '|||' + n); }
        else { e.target.textContent = o; }
    });

    // ── Task card clicks ──
    document.addEventListener('click', function(e) {
        var deleteBtn = e.target.closest('.card-delete');
        if (deleteBtn) {
            var delCard = deleteBtn.closest('.task-card');
            if (delCard) { setBridge('task-delete-bridge', delCard.dataset.id + '|||' + Date.now()); }
            return;
        }
        var card = e.target.closest('.task-card');
        if (!card) return;
        var wasSelected = card.classList.contains('selected');
        var isNameClick = e.target.closest('.card-name') !== null;
        card.closest('.card-list').querySelectorAll('.task-card').forEach(function(c) { c.classList.remove('selected'); });
        card.classList.add('selected');

        if (wasSelected && isNameClick) {
            var nameEl = card.querySelector('.card-name');
            if (nameEl && !nameEl.isContentEditable) {
                nameEl.contentEditable = 'true';
                nameEl.dataset.original = nameEl.textContent;
                nameEl.focus();
                var r = document.createRange(); r.selectNodeContents(nameEl);
                window.getSelection().removeAllRanges(); window.getSelection().addRange(r);
            }
        } else {
            setBridge('task-bridge-box', card.dataset.id);
            window.switchPanel('task');
        }
    });

    // ── Task rename (keydown + focusout) ──
    document.addEventListener('keydown', function(e) {
        if (!e.target.closest('.task-card') || !e.target.classList.contains('card-name') || !e.target.isContentEditable) return;
        if (e.key === 'Enter') {
            e.preventDefault(); e.target.contentEditable = 'false';
            var card = e.target.closest('.task-card');
            var o = e.target.dataset.original, n = e.target.textContent.trim();
            if (n && n !== o && card) { setBridge('task-rename-bridge', card.dataset.id + '|||' + n); }
            else { e.target.textContent = o; }
        } else if (e.key === 'Escape') {
            e.target.textContent = e.target.dataset.original; e.target.contentEditable = 'false';
        }
    });
    document.addEventListener('focusout', function(e) {
        if (!e.target.closest('.task-card') || !e.target.classList.contains('card-name') || !e.target.isContentEditable) return;
        var card = e.target.closest('.task-card');
        var o = e.target.dataset.original, n = e.target.textContent.trim();
        e.target.contentEditable = 'false';
        if (n && n !== o && card) { setBridge('task-rename-bridge', card.dataset.id + '|||' + n); }
        else { e.target.textContent = o; }
    });

    console.log('[WebUI] All event handlers ready');
}
"""


def build_app():
    auth_dict = parse_auth(AUTH_USERS)
    need_auth = bool(auth_dict)

    with gr.Blocks(title="Datatrove 数据筛选平台") as app:

        current_dataset = gr.State("")
        current_task_id = gr.State("")
        pending_switch_task = gr.State(None)

        # ── Login overlay ──
        with gr.Column(visible=need_auth, elem_classes="login-overlay") as login_screen:
            gr.HTML(
                '<div style="text-align:center;margin:80px 0 20px">'
                '<div style="font-size:2em;margin-bottom:8px">🔒</div>'
                '<div style="font-size:1.1em;font-weight:600;color:#4338ca">'
                "Datatrove 数据筛选平台</div>"
                '<div style="font-size:0.85em;color:#64748b;margin-top:4px">'
                "请登录以继续</div></div>"
            )
            login_user = gr.Textbox(label="用户名", placeholder="admin")
            login_pass = gr.Textbox(label="密码", type="password", placeholder="••••••••")
            login_btn = gr.Button("登录", variant="primary", size="sm")
            login_msg = gr.HTML("")

        # ── Header ──
        gr.HTML(
            '<div class="app-header">'
            "<h1>Datatrove</h1>"
            "<span>数据筛选与质量评估平台</span>"
            "</div>"
        )

        with gr.Row(equal_height=False):

            # ── SIDEBAR ──
            with gr.Column(scale=0, min_width=300, elem_classes="sidebar"):

                # ── Dataset section (top half) ──
                with gr.Column(elem_classes="sidebar-section"):
                    gr.HTML('<div class="sidebar-label">📂 数据集</div>')
                    dataset_list = gr.HTML(value=render_dataset_cards(), elem_classes="card-list-wrap")
                    add_ds_btn = gr.Button("+ 添加数据集", elem_classes="card-add-btn", size="sm")

                    ds_bridge_box = gr.Textbox(value="", elem_id="ds-bridge-box", elem_classes="hidden-bridge", lines=1)
                    ds_rename_bridge_box = gr.Textbox(value="", elem_id="ds-rename-bridge", elem_classes="hidden-bridge", lines=1)

                    with gr.Group(visible=False, elem_classes="add-dialog") as add_dialog:
                        upload_btn = gr.UploadButton(
                            "📂 上传本地文件 (.jsonl / .zip)",
                            file_count="multiple",
                            size="sm",
                            elem_classes="upload-btn",
                        )
                        upload_files = gr.State(None)
                        upload_info = gr.HTML("")
                        gr.HTML('<div class="add-dialog-divider">— 或 —</div>')
                        dir_input = gr.Textbox(
                            label="服务器路径", lines=1,
                            placeholder="/data/corpus/...",
                        )
                        add_name_input = gr.Textbox(label="数据集名称", lines=1)
                        with gr.Row():
                            confirm_add_btn = gr.Button("确认", variant="primary", size="sm")
                            cancel_add_btn = gr.Button("取消", size="sm")

                # ── Task section (bottom half) ──
                with gr.Column(elem_classes="sidebar-section"):
                    gr.HTML('<div class="sidebar-label">📋 任务</div>')
                    task_list = gr.HTML(value=render_task_cards(), elem_classes="card-list-wrap")
                    add_task_btn = gr.Button("+ 新建任务", elem_classes="card-add-btn", size="sm")
                    task_bridge_box = gr.Textbox(value="", elem_id="task-bridge-box", elem_classes="hidden-bridge", lines=1)
                    task_delete_bridge = gr.Textbox(value="", elem_id="task-delete-bridge", elem_classes="hidden-bridge", lines=1)
                    task_rename_bridge = gr.Textbox(value="", elem_id="task-rename-bridge", elem_classes="hidden-bridge", lines=1)

                    with gr.Group(visible=False, elem_classes="confirm-overlay") as switch_dialog:
                        gr.HTML('<div class="confirm-msg">⚠️ 当前任务正在运行，是否终止并切换？</div>')
                        with gr.Row():
                            confirm_switch_btn = gr.Button("终止并切换", variant="stop", size="sm")
                            cancel_switch_btn = gr.Button("取消", size="sm")

            # ── MAIN PANEL ──
            with gr.Column(scale=1, elem_classes="main-panel"):

                # ── Panel: Idle (default) ──
                with gr.Group(elem_classes="panel-idle"):
                    gr.HTML(
                        '<div class="idle-message">'
                        '<div style="font-size:2.5em;margin-bottom:16px">📂</div>'
                        '<div style="font-size:1.1em;font-weight:600;color:#1e293b">选择数据集开始</div>'
                        '<div style="font-size:0.85em;color:#64748b;margin-top:8px">'
                        '在左侧选择一个数据集，或点击"+ 添加数据集"上传新数据</div>'
                        '</div>'
                    )

                # ── Panel: Dataset preview ──
                with gr.Group(elem_classes="panel-dataset panel-hidden"):
                    ds_info_header = gr.HTML("")
                    ds_preview_md = gr.Markdown("", elem_classes="preview-table")
                    gr.HTML(
                        '<div class="info-banner" style="margin-top:16px">'
                        '💡 点击左侧"+ 新建任务"创建处理任务'
                        '</div>'
                    )

                # ── Panel: Task config/run ──
                with gr.Group(elem_classes="panel-task panel-hidden"):

                    task_header = gr.HTML("")
                    preview_md = gr.Markdown("", elem_classes="preview-table")

                    gr.HTML('<div class="section-heading">流水线配置</div>')

                    with gr.Accordion(
                        "🧹 数据清洗 — 语言过滤 · 质量过滤 · 去重",
                        open=False, elem_classes="step-accordion",
                    ):
                        enable_clean = gr.Checkbox(label="启用", value=False)
                        clean_languages = gr.CheckboxGroup(
                            choices=["zh", "en"],
                            value=["zh"],
                            label="目标语言（可多选）",
                        )

                    with gr.Accordion(
                        "📊 FCD 词汇难度打分 — 英文，纯 CPU",
                        open=False, elem_classes="step-accordion",
                    ):
                        enable_fcd = gr.Checkbox(label="启用", value=False)
                        with gr.Row():
                            fcd_freq = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="freq_scaling_factor")
                            fcd_wf = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="w_f")
                            fcd_noun = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="noun_weight")

                    with gr.Accordion(
                        "🧠 CDF-GC 生成复杂度 — 中文，需 GPU",
                        open=False, elem_classes="step-accordion",
                    ):
                        enable_gc = gr.Checkbox(label="启用", value=False)
                        with gr.Row():
                            gc_lang = gr.Dropdown(["zh"], value="zh", label="语言")
                            gc_rate = gr.Slider(0.01, 1.0, value=0.2, step=0.01, label="采样率")
                        with gr.Row():
                            gc_tok = gr.Textbox(label="Tokenizer 路径", placeholder="/path/to/tokenizer.json")
                            gc_ltp = gr.Textbox(label="LTP 模型路径", placeholder="/path/to/LTP/small")
                        with gr.Row():
                            gc_cuda = gr.Textbox(label="CUDA_VISIBLE_DEVICES", value="0")
                            gc_wpg = gr.Number(label="Workers / GPU", value=4, precision=0)

                    with gr.Accordion(
                        "🎲 采样 — 需先启用 FCD",
                        open=False, elem_classes="step-accordion",
                    ):
                        enable_sample = gr.Checkbox(label="启用", value=False)
                        with gr.Row():
                            samp_method = gr.Dropdown(["cdf", "hard", "random"], value="cdf", label="策略")
                            samp_rate = gr.Slider(0.01, 1.0, value=0.2, step=0.01, label="采样率")
                        with gr.Row():
                            samp_unit = gr.Dropdown(["doc", "token"], value="doc", label="单位")
                            samp_src = gr.Dropdown(["fcd"], value="fcd", label="分数来源")

                    with gr.Row():
                        max_workers = gr.Slider(1, 64, value=8, step=1, label="Workers", scale=3)
                        start_btn = gr.Button("▶ 开始处理", variant="primary", size="lg", scale=2)
                        resume_btn = gr.Button("↻ 续传", variant="secondary", size="lg", scale=1)

                    run_info = gr.HTML("")

                    gr.HTML('<div class="section-heading">执行状态</div>')
                    poll_btn = gr.Button("刷新状态", size="sm")
                    status_html = gr.HTML(
                        '<div style="color:#94a3b8;font-size:0.85em">等待任务启动...</div>'
                    )
                    log_box = gr.Textbox(
                        label="运行日志", lines=10, max_lines=20,
                        interactive=False, elem_classes="log-area",
                    )

                    gr.HTML('<div class="section-heading">结果与分析</div>')
                    with gr.Row():
                        plot_type = gr.Dropdown(["fcd", "gc"], value="fcd", label="分数类型", scale=2)
                        plot_btn = gr.Button("绘制分布", size="sm", scale=1)
                        download_btn = gr.Button("📦 下载结果", size="sm", scale=1)
                        save_dataset_btn = gr.Button("💾 保存为数据集", size="sm", scale=1)
                    score_plot = gr.Plot(elem_classes="plot-area")
                    download_file = gr.File(label="结果压缩包", interactive=False)

                    with gr.Group(visible=False) as save_dialog:
                        save_name_input = gr.Textbox(label="保存为数据集名称", lines=1)
                        with gr.Row():
                            confirm_save_btn = gr.Button("确认保存", variant="primary", size="sm")
                            cancel_save_btn = gr.Button("取消", size="sm")

        # ═══════════════════════════════
        #  LOGIN HANDLER
        # ═══════════════════════════════
        def handle_login(username, password):
            if username in auth_dict and auth_dict[username] == password:
                return gr.Column(visible=False), ""
            return (
                gr.update(),
                '<div style="color:#dc2626;text-align:center;font-size:0.85em;margin-top:8px">'
                "用户名或密码错误</div>",
            )

        login_btn.click(
            fn=handle_login,
            inputs=[login_user, login_pass],
            outputs=[login_screen, login_msg],
        )
        login_pass.submit(
            fn=handle_login,
            inputs=[login_user, login_pass],
            outputs=[login_screen, login_msg],
        )

        # ═══════════════════════════════
        #  EVENT WIRING
        #  Pattern: JS sets hidden Textbox value → .input() fires → Python handler runs.
        # ═══════════════════════════════

        # ── Dataset select: bridge textbox → update state + preview ──
        ds_bridge_box.input(
            fn=handle_dataset_preview,
            inputs=[ds_bridge_box],
            outputs=[current_dataset, ds_info_header, ds_preview_md],
        )

        # ── Dataset rename: bridge textbox → rename handler ──
        ds_rename_bridge_box.input(
            fn=handle_dataset_rename,
            inputs=[ds_rename_bridge_box, current_dataset],
            outputs=[dataset_list, current_dataset],
        )

        # ── Task select: bridge textbox → switch task ──
        task_bridge_box.input(
            fn=lambda new_tid, current_tid: handle_switch_task(new_tid, current_tid),
            inputs=[task_bridge_box, current_task_id],
            outputs=[
                current_task_id,
                task_header, status_html, log_box, preview_md, score_plot,
                switch_dialog,
                pending_switch_task,
                task_list,
            ],
        )

        add_ds_btn.click(
            fn=handle_show_add_dialog,
            outputs=[add_dialog, add_name_input, upload_files, upload_info, dir_input],
        )
        upload_btn.upload(
            fn=handle_file_upload,
            inputs=[upload_btn],
            outputs=[upload_files, upload_info],
        )
        confirm_add_btn.click(
            fn=handle_confirm_add,
            inputs=[upload_files, dir_input, add_name_input],
            outputs=[add_dialog, dataset_list, current_dataset, upload_files, upload_info, dir_input, add_name_input],
        )
        cancel_add_btn.click(
            fn=handle_cancel_add,
            outputs=[add_dialog, upload_files, upload_info, dir_input, add_name_input],
        )

        # ── Create task ──
        add_task_btn.click(
            fn=handle_create_task,
            inputs=[current_dataset, current_task_id],
            outputs=[
                current_dataset, current_task_id, task_list,
                task_header, status_html, log_box, preview_md, score_plot,
            ],
            js="(ds, tid) => { if(window.switchPanel) window.switchPanel('task'); return [window.__selectedDataset || document.querySelector('.dataset-card.selected')?.dataset?.name || ds || '', tid]; }",
        )

        confirm_switch_btn.click(
            fn=handle_confirm_switch,
            inputs=[pending_switch_task, current_task_id],
            outputs=[
                current_task_id,
                task_header, status_html, log_box, preview_md, score_plot,
                switch_dialog,
                pending_switch_task,
                task_list,
            ],
        )
        cancel_switch_btn.click(
            fn=handle_cancel_switch,
            inputs=[current_task_id],
            outputs=[switch_dialog, pending_switch_task, task_list],
        )

        # ── Task delete / rename ──
        task_delete_bridge.change(
            fn=handle_task_delete,
            inputs=[task_delete_bridge, current_task_id],
            outputs=[
                current_task_id, task_list,
                task_header, status_html, log_box, preview_md, score_plot,
            ],
        )
        task_rename_bridge.input(
            fn=handle_task_rename,
            inputs=[task_rename_bridge, current_task_id],
            outputs=[task_list, task_header],
        )

        start_btn.click(
            fn=handle_start,
            inputs=[
                current_task_id,
                enable_clean, clean_languages,
                enable_fcd, fcd_freq, fcd_wf, fcd_noun,
                enable_gc, gc_rate, gc_lang, gc_tok, gc_ltp, gc_cuda, gc_wpg,
                enable_sample, samp_method, samp_rate, samp_unit, samp_src,
                max_workers,
            ],
            outputs=[run_info, log_box, task_list],
        )
        resume_btn.click(
            fn=handle_resume,
            inputs=[current_task_id, max_workers],
            outputs=[run_info, task_list],
        )

        poll_btn.click(
            fn=handle_poll,
            inputs=[current_task_id],
            outputs=[status_html, log_box, score_plot, task_list],
        )

        plot_btn.click(
            fn=handle_plot,
            inputs=[current_task_id, plot_type],
            outputs=[score_plot],
        )
        download_btn.click(
            fn=handle_download,
            inputs=[current_task_id],
            outputs=[download_file],
        )

        save_dataset_btn.click(
            fn=handle_show_save_dialog,
            inputs=[current_task_id],
            outputs=[save_dialog, save_name_input, run_info],
        )
        confirm_save_btn.click(
            fn=handle_confirm_save,
            inputs=[current_task_id, save_name_input],
            outputs=[save_dialog, dataset_list, current_dataset, run_info],
        )
        cancel_save_btn.click(
            fn=handle_cancel_save,
            outputs=[save_dialog],
        )

        # ── Global JS setup: document-level event delegation ──
        app.load(fn=None, js=GLOBAL_JS)

    return app


if __name__ == "__main__":
    app = build_app()
    port = int(os.environ.get("WEBUI_PORT", "7860"))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
            radius_size=gr.themes.sizes.radius_md,
        ),
    )
