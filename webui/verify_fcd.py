"""Run real FCD and Hard sampling in an isolated workspace and check UI data."""

import argparse
import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np

import task_manager as tm
from handlers import result_view
from preview import new_preview, load_page, read_detail


def records(folder):
    return [json.loads(line) for path in sorted(Path(folder).glob("*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify(root, source):
    root = Path(root).resolve()
    tm.TASKS_DIR, tm.WORKSPACE_DIR = str(root / "tasks"), str(root / "workspace")
    Path(tm.TASKS_DIR).mkdir(parents=True, exist_ok=True)
    dataset = Path(tm.WORKSPACE_DIR) / "FCD test data"
    dataset.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dataset / "test.jsonl")
    original = records(dataset)
    config = {"steps": ["fcd"], "workers": 1, "rerun": True,
              "fcd_params": {"freq_scaling_factor": 0.7, "w_f": 0.5, "noun_weight": 0.7}}
    tasks = []
    for name, options in (("FCD full test data", config),
                          ("FCD Hard 50% test data", {**config, "sample": {
                              "method": "hard", "sample_rate": 0.5, "unit": "doc", "score_source": "fcd"}})):
        task = tm.create_task_from_workspace(dataset.name)
        task.set_display_name(name)
        thread = tm.run_pipeline_async(task, options, workers=1)
        thread.join(timeout=300)
        if thread.is_alive():
            task.stop()
            thread.join(15)
            raise RuntimeError("FCD verification timed out")
        if task.get_status()["status"] != "completed":
            raise RuntimeError(f"{task.get_status()}\n{task.get_log()}")
        tasks.append(task)
        print(f"Completed {task.task_id}: {name}", flush=True)

    full, sampled = tasks
    scores = tm.load_scores(full)
    output = records(full.output_dir)
    assert len(scores) == len(original) == len(output)
    assert all(math.isfinite(score) and 0 <= score <= 1 for score in scores)
    assert [(row["id"], row["text"]) for row in output] == [(row["id"], row["text"]) for row in original]
    assert [row["metadata"]["fcd_score"] for row in output] == scores
    assert tm.load_scores(sampled) == scores, "Repeated FCD scoring changed"
    kept = records(sampled.output_dir)
    top = set(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:len(scores) // 2])
    expected_kept = [row for i, row in enumerate(output) if i in top]
    assert [(row["id"], row["text"], row["metadata"]["fcd_score"]) for row in kept] == [
        (row["id"], row["text"], row["metadata"]["fcd_score"]) for row in expected_kept]
    for row, original_row in zip(kept, expected_kept):
        assert all(row["metadata"].get(key) == value for key, value in original_row["metadata"].items())
    for task, expected in ((full, output), (sampled, kept)):
        summary, stats, count = result_view(task)
        assert stats == [["输入", len(original)], ["FCD 打分后（采样前）", len(original)], ["最终输出", len(expected)]]
        assert count == len(expected)
        state = load_page(new_preview(task.output_dir))
        assert len(state["rows"]) == min(20, len(expected))
        for index, row in enumerate(expected[:20]):
            text, metadata = read_detail(state, index)
            assert json.loads(metadata)["fcd_score"] == row["metadata"]["fcd_score"]
            assert text == f'ID: {row["id"]}\n\n{row["text"]}'
        with zipfile.ZipFile(task.get_output_archive()) as bundle:
            archived = [json.loads(line) for name in sorted(bundle.namelist()) if name.endswith(".jsonl")
                        for line in bundle.read(name).decode("utf-8").splitlines() if line.strip()]
        assert archived == expected

    figure = tm.plot_score_distribution(scores, "FCD test data")
    counts, edges = np.histogram(scores, bins=100)
    density = counts / counts.sum() / np.diff(edges)
    pdf, cdf = figure.axes[0].lines[0], figure.axes[1].lines[0]
    assert np.allclose(pdf.get_ydata(), density)
    assert np.isclose(np.sum(pdf.get_ydata() * np.diff(edges)), 1)
    assert np.allclose(cdf.get_ydata(), np.r_[0, np.cumsum(counts) / counts.sum()])
    figure.savefig(root / "fcd_distribution.png")
    import matplotlib.pyplot as plt
    plt.close(figure)
    report = {"full_task": full.task_id, "sampled_task": sampled.task_id,
              "input": len(original), "scored": len(scores), "sampled": len(kept),
              "mean": float(np.mean(scores)), "std": float(np.std(scores)),
              "median": float(np.median(scores)), "minimum": min(scores), "maximum": max(scores),
              "checks": ["score alignment", "repeatability", "Hard top half", "UI statistics",
                         "preview metadata", "ZIP contents", "100-bin PDF/CDF"]}
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / ".cache/fcd-acceptance")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent / "test_data/test.jsonl")
    args = parser.parse_args()
    verify(args.root, args.source)
