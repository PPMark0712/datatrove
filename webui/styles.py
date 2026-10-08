"""Layout styles: native controls remain responsible for interaction."""

CUSTOM_CSS = """
:root { color-scheme: light; }
body { margin: 0; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }
.gradio-container > .main { padding: 0 !important; }
.gradio-container > .main > .wrap { gap: 0 !important; }
.gradio-container main > .column { gap: 0 !important; }
#brand { padding: 0 !important; margin: 0 !important; border: 0 !important; }
#brand .html-container { padding: 0 !important; }
footer { display: none !important; }
.app-header {
    box-sizing: border-box; height: 60px; padding: 0 24px;
    display: flex; align-items: center; gap: 16px;
    border-bottom: 1px solid #e2e8f0; background: #fff; color: #0f172a;
}
.app-header strong { color: #4338ca; font-size: 20px; letter-spacing: -.5px; }
.app-header span { color: #475569; font-size: 14px; }
.app-header small { margin-left: auto; color: #64748b; }
#workspace {
    height: calc(100dvh - 60px); min-height: 0 !important;
    flex-wrap: nowrap !important; gap: 0 !important;
}
#sidebar {
    box-sizing: border-box; height: 100%; min-height: 0 !important;
    flex: 0 0 280px !important; gap: 0 !important; padding: 0 12px 12px;
    border-right: 1px solid #e2e8f0; background: #f8fafc;
    overflow: hidden; display: flex; flex-direction: column;
}
.sidebar-section {
    flex: 1 1 0 !important; min-height: 0 !important;
    overflow: hidden; gap: 0 !important; padding-bottom: 10px;
    display: flex !important; flex-direction: column !important;
}
.sidebar-section + .sidebar-section { border-top: 1px solid #e2e8f0; }
.section-heading { flex: 0 0 auto !important; }
.section-heading .html-container { padding: 0 !important; }
.sidebar-label { font-weight: 600; font-size: 13px; color: #475569; padding: 14px 4px 10px; }
.sidebar-scroll {
    display: block !important; min-height: 0 !important;
    overflow-y: auto !important; overflow-x: hidden !important;
    flex: 0 1 auto !important; scrollbar-gutter: stable;
}
.item-list { border: 0 !important; padding: 0 !important; background: transparent !important; }
.item-list .wrap { display: flex !important; flex-direction: column; gap: 6px !important; }
.item-list label {
    width: 100%; box-sizing: border-box; margin: 0 !important;
    padding: 8px 10px !important; border-radius: 6px !important;
    border: 1px solid #e2e8f0 !important; box-shadow: none !important;
    font-size: 13px !important; line-height: 1.5 !important;
    color: #334155; overflow-wrap: anywhere; cursor: pointer;
}
.item-list label:has(input:checked) { background: #eef2ff !important; border-color: #818cf8 !important; }
.item-list input { flex-shrink: 0; }
.item-list label:focus-within { outline: 2px solid #6366f1; outline-offset: -2px; }
.add-button {
    flex: 0 0 auto !important; margin-top: 8px !important; width: 100%;
    min-height: 38px !important; border: 1px dashed #94a3b8 !important;
    background: transparent !important; color: #475569 !important;
}
.add-button:disabled { opacity: .5; }
#main-panel {
    box-sizing: border-box; height: 100%; min-width: 0 !important;
    padding: 18px 28px 28px; display: block !important;
    overflow-y: auto; overflow-x: hidden;
}
#main-panel h2 { margin: 0 0 8px; font-size: 23px; letter-spacing: -.4px; }
.name-actions { align-items: end !important; }
.name-actions > button { flex-grow: 0 !important; min-width: 88px !important; margin-bottom: 1px; }
.config-feedback { border-left: 3px solid #a5b4fc; padding: 6px 12px !important; background: #eef2ff; }
.config-feedback p { color: #475569; font-size: 13px; margin: 0 !important; }
.task-tabs > .tab-nav { background: var(--background-fill-primary, #fff); }
.run-actions { padding: 8px 0; }
.empty-state { max-width: 620px; margin: 12vh auto 0; }
.empty-state p { color: #64748b; line-height: 1.8; }
.eyebrow { color: #6366f1; font-size: 12px; letter-spacing: 2px; margin-bottom: 16px; }
.muted { color: #64748b; font-size: 13px; line-height: 1.8; }
.status-line { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; }
.status-line strong { font-size: 20px; overflow-wrap: anywhere; }
.status-line > span:last-child { font-size: 13px; color: #64748b; }
.badge { padding: 4px 9px; border-radius: 5px; font-size: 12px; background: #f1f5f9; color: #475569; }
.badge-running, .badge-stopping { background: #fef3c7; color: #92400e; }
.badge-completed { background: #d1fae5; color: #065f46; }
.badge-error { background: #fee2e2; color: #991b1b; }
.badge-created { background: #e0e7ff; color: #3730a3; }
.preview-table {
    max-width: 100%; border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important; overflow: hidden !important;
}
.preview-table table { border-collapse: separate !important; border-spacing: 0 !important; }
.preview-table thead { position: sticky; top: 0; z-index: 2; }
.preview-table th {
    background: #f1f5f9 !important; color: #334155 !important;
    font-size: 12px !important; font-weight: 600 !important;
}
.preview-table th,
.preview-table td {
    padding: 7px 9px !important; border-right: 1px solid #dbe3ec !important;
    border-bottom: 1px solid #dbe3ec !important; vertical-align: top !important;
}
.preview-table th:last-child,
.preview-table td:last-child { border-right: 0 !important; }
.preview-table tbody tr:last-child td { border-bottom: 0 !important; }
.preview-table tbody tr:nth-child(even) td { background: #f8fafc !important; }
.preview-table tbody tr:hover td { background: #eef2ff !important; }
.preview-table td { max-width: 540px; font-size: 12px !important; line-height: 1.4 !important; }
.preview-table td .cell-wrap {
    display: -webkit-box !important; max-height: 34px !important; overflow: hidden !important;
    -webkit-box-orient: vertical; -webkit-line-clamp: 2;
}
/* Gradio 6 virtualizes body rows as divs, separately from its header table. */
.preview-table .body-cell,
.stats-table .body-cell {
    border-right: 1px solid #dbe3ec !important;
    border-bottom: 1px solid #dbe3ec !important;
}
.preview-table .body-cell:last-child,
.stats-table .body-cell:last-child { border-right: 0 !important; }
.preview-table .virtual-row:nth-child(even) .body-cell { background: #f8fafc; }
.preview-table .virtual-row:hover .body-cell { background: #eef2ff !important; }
.preview-table .body-cell .cell-wrap { padding: 8px 10px !important; }
.preview-table .body-cell .text {
    font: 13px/20px system-ui, sans-serif !important;
    white-space: normal !important; overflow-wrap: anywhere;
    display: -webkit-box !important; -webkit-box-orient: vertical;
    -webkit-line-clamp: 2; max-height: 40px; overflow: hidden;
}
.preview-table .body-cell:focus-within { outline: 2px solid #6366f1; outline-offset: -2px; }
.log-area textarea {
    font-family: ui-monospace, Menlo, monospace !important;
    font-size: 12px !important; line-height: 1.6;
}
#dataset-modal {
    position: fixed !important; inset: 0; z-index: 100;
    width: 100vw; height: 100dvh; background: #0f172a66;
    box-sizing: border-box; padding: 16px; align-items: center; justify-content: center;
}
#dataset-dialog-card {
    width: min(520px, calc(100vw - 32px)) !important;
    max-height: calc(100dvh - 40px); overflow-y: auto;
    box-sizing: border-box; padding: 24px; border-radius: 12px;
    background: white; border: 1px solid #cbd5e1;
    box-shadow: 0 20px 50px #0f172a33;
    flex: 0 1 auto !important; overscroll-behavior: contain;
}
@media (max-width: 800px) {
    #sidebar { flex-basis: 220px !important; min-width: 220px !important; }
    #main-panel { padding: 14px; }
    .app-header small { display: none; }
}
@media (max-width: 560px) {
    .app-header { padding: 0 12px; gap: 8px; }
    .app-header span { font-size: 11px; }
    #workspace { height: auto; flex-wrap: wrap !important; }
    #sidebar { flex: 1 1 100% !important; height: 300px; border-right: 0; }
    #main-panel { height: auto; overflow: visible; }
    .empty-state { margin-top: 24px; }
}
"""
