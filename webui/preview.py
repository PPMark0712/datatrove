"""Bounded JSONL preview: retain summaries and offsets, never open file handles."""

import json
from pathlib import Path

import gradio as gr


PAGE_SIZE = 20
MAX_ROWS = 100
HINT = "展开后加载前 20 行，下滑逐页加载；至多预览 100 行（5 页）。点击行查看完整记录，收起后释放预览内容。"


def new_preview(folder):
    return {"files": [str(p) for p in sorted(Path(folder).glob("*.jsonl"))],
            "file": 0, "offset": 0, "rows": [], "locations": [], "more": True}


def next_record(state):
    """Return one nonblank line and its location; advance across file boundaries."""
    while state["file"] < len(state["files"]):
        path = state["files"][state["file"]]
        with open(path, "rb") as source:
            source.seek(state["offset"])
            while True:
                offset = source.tell()
                line = source.readline()
                state["offset"] = source.tell()
                if not line:
                    break
                if line.strip():
                    return line, (path, offset)
        state["file"] += 1
        state["offset"] = 0
    return None


def load_page(state):
    if not state or not state["more"] or len(state["rows"]) >= MAX_ROWS:
        return state
    # Work on a copy so a failed parse does not leave a partially advanced cursor.
    state = {**state, "rows": list(state["rows"]), "locations": list(state["locations"])}
    for _ in range(min(PAGE_SIZE, MAX_ROWS - len(state["rows"]))):
        record = next_record(state)
        if record is None:
            state["more"] = False
            break
        line, location = record
        item = json.loads(line)
        values = [str(item.get("id", len(state["rows"]) + 1)), str(item.get("text", "")),
                  json.dumps(item.get("metadata", {}), ensure_ascii=False)]
        state["rows"].append([v[:160] + "…" if len(v) > 160 else v for v in values])
        state["locations"].append(location)
    # Probe just one following record, without parsing or retaining its contents.
    cursor = state["file"], state["offset"]
    state["more"] = next_record(state) is not None
    state["file"], state["offset"] = cursor
    return state


def read_detail(state, index):
    path, offset = state["locations"][index]
    with open(path, "rb") as source:
        source.seek(offset)
        item = json.loads(source.readline())
    return (f'ID: {item.get("id", index + 1)}\n\n{item.get("text", "")}',
            json.dumps(item.get("metadata", {}), ensure_ascii=False, indent=4))


class Preview:
    def __init__(self, name, visible=True):
        self.name = name
        self.state = gr.State(None)
        with gr.Accordion("数据预览 · 至多 100 行", open=False, visible=visible,
                          elem_id=f"{name}-preview") as self.section:
            gr.Markdown(HINT)
            self.table = gr.Dataframe(
                value=[], headers=["ID", "正文", "元数据"], datatype=["str"] * 3,
                interactive=False, wrap=True, max_height=320, max_chars=160,
                column_widths=["100px", "65%", "150px"], elem_classes="preview-table")
            self.status = gr.Markdown("")
            self.more = gr.Button("… 加载下 20 行", size="sm", visible=False,
                                  elem_id=f"{name}-preview-more")
            with gr.Column(visible=False, elem_id=f"{name}-detail") as self.detail_panel:
                self.detail = gr.Textbox(label="记录正文", lines=6, max_lines=12,
                                         interactive=False, buttons=["copy"])
                self.metadata = gr.Code(label="元数据", language="json", lines=3, max_lines=14,
                                         interactive=False, wrap_lines=True, buttons=["copy"])
        self.detail_outputs = [self.detail_panel, self.detail, self.metadata]
        self.outputs = [self.state, self.table, self.status, self.more] + self.detail_outputs

    @staticmethod
    def cleared():
        return [None, [], "", gr.update(visible=False), gr.update(visible=False), "", ""]

    def reset(self, visible=True):
        return [gr.update(open=False, visible=visible)] + self.cleared()

    @staticmethod
    def page(state):
        try:
            state = load_page(state)
        except (OSError, ValueError, UnicodeError) as exc:
            raise gr.Error(f"预览加载失败：{exc}") from exc
        if not state:
            return Preview.cleared()
        count = len(state["rows"])
        more = state["more"] and count < MAX_ROWS
        message = f"已加载 {count} 行"
        if not count:
            message = "暂无可预览的记录。"
        elif state["more"]:
            message += " · … 下滑继续加载" if more else " · … 已达 100 行预览上限"
        else:
            message += " · 已显示全部记录"
        return [state, state["rows"], message, gr.update(visible=more)] + [gr.skip()] * 3

    def bind(self, source, selection):
        def expand(*values):
            folder = source(*values)
            return self.page(new_preview(folder)) if folder else self.cleared()

        options = dict(show_progress="hidden", concurrency_id="view-state")
        self.section.expand(expand, selection, self.outputs, **options)
        self.section.collapse(self.cleared, outputs=self.outputs, **options)
        self.more.click(self.page, self.state, self.outputs, **options).then(
            fn=None, js=f"() => window.__previewScroll?.finish('{self.name}')")

        def detail(state, event: gr.SelectData):
            if not state or not 0 <= event.index[0] < len(state["locations"]):
                return gr.update(visible=False), "", ""
            try:
                text, metadata = read_detail(state, event.index[0])
                return gr.update(visible=True), text, metadata
            except (OSError, ValueError, UnicodeError) as exc:
                raise gr.Error(f"记录加载失败：{exc}") from exc

        self.table.select(detail, self.state, self.detail_outputs, **options).success(
            fn=None, js=f"() => requestAnimationFrame(() => document.getElementById('{self.name}-detail')?.scrollIntoView({{block: 'nearest'}}))")


SCROLL_JS = """() => {
    if (window.__previewScroll) return;
    const pending = new Map();
    window.__previewScroll = {
        finish(name) {
            const saved = pending.get(name);
            if (!saved) return;
            requestAnimationFrame(() => {
                if (saved.node.isConnected) saved.node.scrollTop = saved.top;
                pending.delete(name);
            });
        }
    };
    document.addEventListener('scroll', event => {
        const node = event.target;
        if (!(node instanceof Element) || !node.closest('.preview-table')) return;
        const section = node.closest('[id$="-preview"]');
        if (!section || node.scrollTop <= 0 ||
            node.scrollHeight - node.clientHeight - node.scrollTop > 60) return;
        const name = section.id.replace(/-preview$/, '');
        const button = document.getElementById(name + '-preview-more');
        if (pending.has(name) || !button?.getClientRects().length || button.disabled) return;
        pending.set(name, {node, top: node.scrollTop});
        button.click();
    }, true);
}"""
