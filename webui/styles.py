"""CSS styles for the Datatrove WebUI."""

CUSTOM_CSS = """
/* ── Reset & full-height layout ── */
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    height: 100vh !important;
    overflow: hidden !important;
}
.gradio-container > .main > .wrap {
    overflow: hidden !important;
}
footer { display: none !important; }

/* Hidden bridge buttons — rendered in DOM but invisible */
.hidden-bridge {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    clip: rect(0,0,0,0) !important;
    border: 0 !important;
    padding: 0 !important;
    margin: -1px !important;
    pointer-events: none !important;
}

/* ── App header — sticky top bar ── */
.app-header {
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
    padding: 14px 28px;
    display: flex;
    align-items: baseline;
    gap: 12px;
    flex-shrink: 0;
}
.app-header h1 {
    color: #fff;
    font-size: 1.15em;
    font-weight: 700;
    margin: 0;
}
.app-header span {
    color: rgba(255,255,255,0.65);
    font-size: 0.82em;
}

/* ── Sidebar — fixed, independently scrollable ── */
.sidebar {
    background: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
    padding: 12px !important;
    overflow-y: auto !important;
    flex-shrink: 0 !important;
    height: calc(100vh - 60px) !important;
}

/* ── Main panel — scrollable content area ── */
.main-panel {
    padding: 0 32px 24px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    height: calc(100vh - 60px) !important;
}
.main-panel > * {
    max-width: 100% !important;
    box-sizing: border-box !important;
}

/* ── Sidebar labels ── */
.sidebar-label {
    font-size: 0.72em;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 8px 0 6px 2px;
    flex-shrink: 0;
}

/* ── Card list ── */
.card-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.card {
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 12px;
    cursor: pointer;
    transition: all 0.15s ease;
    background: #fff;
    display: flex;
    align-items: center;
    gap: 8px;
    min-height: 36px;
    user-select: none;
}
.card:hover {
    border-color: #a5b4fc;
}
.card.selected {
    border-color: #6366f1;
    background: #eef2ff;
}
.card-name {
    flex: 1;
    font-size: 0.85em;
    color: #1e293b;
    outline: none;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.card-name[contenteditable="true"] {
    background: #fff;
    border-radius: 3px;
    padding: 1px 4px;
    box-shadow: 0 0 0 2px #6366f1;
    white-space: pre;
    user-select: text;
    cursor: text;
}
.card-meta {
    font-size: 0.72em;
    color: #94a3b8;
    flex-shrink: 0;
}
.card-icon {
    flex-shrink: 0;
    font-size: 0.8em;
}
.card-delete {
    display: none;
    cursor: pointer;
    font-size: 0.8em;
    opacity: 0.35;
    flex-shrink: 0;
    transition: opacity 0.15s;
    padding: 0 2px;
}
.card:hover .card-delete {
    display: inline-block;
}
.card-delete:hover {
    opacity: 1;
}

/* ── Add card (dashed +) ── */
.card-add {
    border: 2px dashed #cbd5e1;
    background: transparent;
    justify-content: center;
    color: #94a3b8;
    font-size: 1.1em;
    font-weight: 500;
}
.card-add:hover {
    border-color: #6366f1;
    color: #6366f1;
}

/* ── Add card button (native Gradio button styled as dashed card) ── */
.card-add-btn {
    border: 2px dashed #cbd5e1 !important;
    background: transparent !important;
    color: #94a3b8 !important;
    font-size: 0.9em !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    margin-top: 4px !important;
    box-shadow: none !important;
}
.card-add-btn:hover {
    border-color: #6366f1 !important;
    color: #6366f1 !important;
    background: transparent !important;
}

/* ── Empty state hint ── */
.empty-hint {
    text-align: center;
    color: #94a3b8;
    font-size: 0.78em;
    padding: 6px 0;
}

/* ── Section headings in main panel ── */
.section-heading {
    font-size: 0.92em;
    font-weight: 600;
    color: #1e293b;
    padding: 10px 0 8px;
    border-bottom: 1px solid #e2e8f0;
    margin: 24px 0 12px;
}

/* ── Step accordion ── */
.step-accordion {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
}

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.75em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.badge-created  { background: #dbeafe; color: #1e40af; }
.badge-running  { background: #fef3c7; color: #92400e; }
.badge-completed { background: #d1fae5; color: #065f46; }
.badge-error    { background: #fee2e2; color: #991b1b; }
.badge-stopped  { background: #f1f5f9; color: #475569; }

/* ── Task info card ── */
.task-info-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    line-height: 1.6;
}

/* ── Log area ── */
.log-area textarea {
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace !important;
    font-size: 0.78em !important;
    line-height: 1.5 !important;
    background: #1e293b !important;
    color: #cbd5e1 !important;
    border-radius: 6px !important;
}

/* ── Info banner ── */
.info-banner {
    background: #f0f4ff;
    border-left: 3px solid #6366f1;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 0.85em;
    color: #1e1b4b;
    line-height: 1.5;
    margin-bottom: 8px;
}

/* ── Plot area ── */
.plot-area {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background: #fff;
}

/* ── Preview table ── */
.preview-table table {
    font-size: 0.85em;
}
.preview-table th {
    background: #f1f5f9 !important;
}

/* ── Confirmation dialog ── */
.confirm-overlay {
    background: #fefce8 !important;
    border: 1px solid #fbbf24 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin: 8px 0 !important;
}
.confirm-msg {
    font-size: 0.85em;
    color: #92400e;
    margin-bottom: 8px;
    line-height: 1.5;
}

/* ── Add dataset dialog ── */
.add-dialog {
    background: #f0f4ff !important;
    border: 1px solid #c7d2fe !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin: 8px 0 !important;
    position: relative;
    z-index: 10;
}
/* ── Screen-reader only (visually hidden but clickable by JS) ── */
.sr-only {
    position: fixed !important;
    top: -9999px !important;
    left: -9999px !important;
    opacity: 0 !important;
    width: 1px !important;
    height: 1px !important;
}

.add-dialog-divider {
    text-align: center;
    color: #94a3b8;
    font-size: 0.8em;
    margin: 6px 0;
}

/* ── Upload button inside add-dialog ── */
.upload-btn {
    width: 100% !important;
    margin-bottom: 8px !important;
}
.upload-btn button {
    width: 100% !important;
    border: 1px dashed #a5b4fc !important;
    background: #fff !important;
    color: #4338ca !important;
    font-size: 0.85em !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
}
.upload-btn button:hover {
    border-color: #6366f1 !important;
    background: #eef2ff !important;
}

/* ── Login overlay — full-screen cover over main app ── */
.login-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 9999;
    background: #fff;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0 20px;
    max-width: none !important;
    overflow-y: auto;
}
.login-overlay > * {
    max-width: 360px !important;
    width: 100% !important;
}

/* ── Main app wrapper (no extra chrome) ── */
.main-app-wrap {
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}
"""
