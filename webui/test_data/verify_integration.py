"""Local integration verification; excluded from version control with test_data."""
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import task_manager as tm
from handlers import result_view

root = Path(tempfile.mkdtemp(prefix="webui-integration-"))
tm.TASKS_DIR = str(root / "tasks")
tm.WORKSPACE_DIR = str(root / "workspace")
Path(tm.TASKS_DIR).mkdir()
Path(tm.WORKSPACE_DIR).mkdir()
source = Path(__file__).resolve().parents[1] / "workspace/数据集1/test.jsonl"
records = [json.loads(line) for line in source.read_text().splitlines()]


def dataset(name, rows):
    path = Path(tm.WORKSPACE_DIR) / name
    path.mkdir()
    (path / "test.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return tm.create_task_from_workspace(name)


def run(task, config):
    thread = tm.run_pipeline_async(task, config, workers=1)
    thread.join(timeout=60)
    assert not thread.is_alive(), ("timed out", task.task_dir)
    assert task.get_status()["status"] == "completed", (task.get_status(), task.get_log())
    return [json.loads(line) for path in Path(task.output_dir).glob("*.jsonl") for line in path.read_text().splitlines()]


clean = {"steps": ["data_clean"], "languages": ["zh", "en"], "rerun": True}
task = dataset("mixed", records)
output = run(task, clean)
assert len(output) == 20
assert len({r["text"] for r in output}) == 20
assert {r["metadata"]["language"] for r in output} == {"en", "zh"}
assert len({r["id"] for r in output}) == 20
print("PASS mixed cleaning: 29 -> 20; both languages; unique final records", flush=True)
print(result_view(task)[0], flush=True)

output = run(task, {**clean, "languages": ["en"]})
assert len(output) == 10 and all(r["metadata"]["language"] == "en" for r in output)
print("PASS changed-config rerun removes cached Chinese outputs", flush=True)

empty = dataset("only-japanese", [next(r for r in records if r["id"] == "dirty_lang_ja")])
assert run(empty, clean) == []
print("PASS all-language-filtered -> completed empty output", flush=True)

short = dataset("short", [{"id": "short", "text": "The public library opened a reading room beside the garden. Visitors can borrow books and study there during the afternoon."}])
assert run(short, {**clean, "languages": ["en"]}) == []
print("PASS all-quality-filtered -> completed empty output", flush=True)

stop_task = dataset("stop-resume", records)
thread = tm.run_pipeline_async(stop_task, clean, workers=1)
deadline = time.monotonic() + 5
while time.monotonic() < deadline and not tm.RUNS.get(stop_task.task_id, {}).get("proc"):
    time.sleep(.05)
process = tm.RUNS[stop_task.task_id].get("proc")
try:
    tm.run_pipeline_async(task, clean)
    raise AssertionError("concurrent launch was accepted")
except ValueError:
    pass
stop_task.stop()
thread.join(10)
assert not thread.is_alive()
assert stop_task.get_status()["status"] == "stopped"
assert not process or process.poll() is not None
assert len(run(stop_task, {**clean, "rerun": False})) == 20
print("PASS real process stop, concurrent launch rejection, resume", flush=True)

archive = root / "duplicate-names.zip"
with zipfile.ZipFile(archive, "w") as zf:
    for name in ["a/test.jsonl", "b/test.jsonl"]:
        zf.writestr(name, json.dumps(records[0]) + "\n")
staged = tm.stage_upload(files=[str(archive)])
assert len(list(Path(staged).glob("*.jsonl"))) == 2
tm.cancel_upload(staged)
bad = root / "bad.jsonl"
bad.write_text('{"id":1}\n')
try:
    tm.stage_upload(files=[str(bad)])
    raise AssertionError("invalid JSONL accepted")
except ValueError as exc:
    assert "第 1 行" in str(exc)
assert not list(Path(tm.WORKSPACE_DIR).glob(".staging_*"))
print("PASS ZIP collisions preserved; invalid record rejected; staging cleaned", flush=True)

fcd_task = dataset("english", [r for r in records if r["id"].startswith("en_")])
config = {"steps": ["fcd"], "rerun": True,
          "sample": {"method": "hard", "sample_rate": .5, "unit": "doc", "score_source": "fcd"}}
output = run(fcd_task, config)
assert len(output) == 5
assert len(tm.load_scores(fcd_task, "fcd")) == 10
assert all("fcd_score" in r["metadata"] for r in output)
with zipfile.ZipFile(fcd_task.get_output_archive()) as zf:
    assert sum(len(zf.read(name).splitlines()) for name in zf.namelist() if name.endswith(".jsonl")) == 5
print("PASS FCD + hard sampling -> 5/10; plot scores and ZIP contain actual results", flush=True)
print("ARTIFACTS", root, flush=True)
