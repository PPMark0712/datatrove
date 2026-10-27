"""Regression checks for WebUI controls, cached statistics and process cancellation."""

import builtins
import os
import sys
import time
from pathlib import Path

import pytest


pytest.importorskip("gradio")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webui"))
import task_manager as tm
import handlers
import app as webapp


@pytest.fixture
def task(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "TASKS_DIR", str(tmp_path / "tasks"))
    monkeypatch.setattr(tm, "WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setattr(webapp, "WORKSPACE_DIR", tm.WORKSPACE_DIR)
    Path(tm.TASKS_DIR).mkdir()
    Path(tm.WORKSPACE_DIR).mkdir()
    tm._count_rows.cache_clear()
    task = tm.Task()
    task.init()
    yield task
    tm.RUNS.clear()
    tm._count_rows.cache_clear()


def test_sampling_can_be_disabled_after_fcd_is_disabled():
    values = handlers.config_values()
    values[12] = True
    assert handlers.config_controls(values)[12]["interactive"]
    assert not handlers.config_controls(values)[13]["interactive"]
    assert not handlers.config_controls(values, busy=True)[12]["interactive"]
    values[12] = False
    assert handlers.make_config(*values)["steps"] == ["data_clean"]


def test_resume_restores_saved_config_and_preserves_other_drafts(task, monkeypatch):
    task.save_config({"steps": ["data_clean"], "workers": 4})
    task.set_status("stopped")
    draft = handlers.config_values(task)
    draft[-1] = 16
    drafts = {task.task_id: draft, "task_2": draft}
    launches = []
    monkeypatch.setattr(webapp, "run_pipeline_async", lambda *args: launches.append(args))
    ui = webapp.build_app()
    callbacks = {fn.fn.__name__: fn for fn in ui.fns.values() if fn.fn}
    restored = callbacks["resume_task"].fn(task.task_id, drafts)
    assert restored == {"task_2": draft}
    assert drafts[task.task_id][-1] == 16  # Do not mutate the input State.
    assert launches[0][1]["rerun"] is False
    assert launches[0][2] == 4
    render = callbacks["render"]
    rendered = render.fn("", task.task_id, "task", restored)
    controls = dict(zip((component._id for component in render.outputs), rendered))
    workers = next(component for component in render.outputs
                   if getattr(component, "label", None) == "最大工作进程数")
    assert controls[workers._id]["value"] == 4

    def reject(*args):
        raise ValueError("busy")

    monkeypatch.setattr(webapp, "run_pipeline_async", reject)
    with pytest.raises(Exception, match="busy"):
        callbacks["resume_task"].fn(task.task_id, drafts)
    assert drafts[task.task_id][-1] == 16


def test_result_refresh_reuses_counts_and_invalidates_changed_files(task, monkeypatch):
    source = Path(task.input_dir) / "input.jsonl"
    output = Path(task.output_dir) / "output.jsonl"
    source.write_text('{"text":"a"}\n\n{"text":"b"}\n', encoding="utf-8")
    output.write_text('{"text":"a"}\n', encoding="utf-8")
    task.save_config({"steps": []})
    task.set_status("completed")
    opened = []
    original_open = builtins.open

    def count_open(path, *args, **kwargs):
        if str(path).endswith(".jsonl"):
            opened.append(str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", count_open)
    assert handlers.result_view(task)[1] == [["输入", 2], ["最终输出", 1]]
    first_reads = len(opened)
    assert first_reads == 2
    assert tm.Task(task.task_id).count_input_rows() == 2
    assert handlers.result_view(tm.Task(task.task_id))[2] == 1
    assert len(opened) == first_reads
    with original_open(output, "a", encoding="utf-8") as stream:
        stream.write('{"text":"b"}\n')
    assert handlers.result_view(task)[2] == 2
    assert len(opened) == first_reads + 1
    output.unlink()
    assert handlers.result_view(task)[2] == 0


@pytest.mark.skipif(os.name != "nt", reason="Windows process tree regression")
def test_stop_terminates_real_worker_and_child(task, tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")

    pid_file = tmp_path / "child.pid"
    child_code = "import time; time.sleep(120)"
    parent_code = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        f"Path({str(pid_file)!r}).write_text(str(child.pid)); time.sleep(120)"
    )

    def split(task, workers):
        tm.run_script([sys.executable, "-c", parent_code], task.log_file)
        return True

    monkeypatch.setattr(tm, "step_split", split)
    thread = tm.run_pipeline_async(task, {"steps": [], "rerun": True}, workers=1)
    process = None
    child = None
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists(), task.get_log()
        child = psutil.Process(int(pid_file.read_text()))
        process = tm.RUNS[task.task_id]["proc"]
        task.stop()
        thread.join(10)
        assert not thread.is_alive()
        assert task.get_status()["status"] == "stopped"
        assert process.poll() is not None
        assert not child.is_running()
        assert task.task_id not in tm.RUNS
    finally:
        if process and process.poll() is None:
            tm.terminate_process_tree(process, force=True)
        if child and child.is_running():
            child.kill()
        thread.join(10)
