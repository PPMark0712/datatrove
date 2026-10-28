import os
import json
import shutil
import glob as glob_module
import subprocess
import tempfile
import threading
import time
import zipfile
import sys
import signal
from functools import lru_cache
from pathlib import Path


WEBUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WEBUI_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
TASKS_DIR = os.path.join(WEBUI_DIR, "tasks")
WORKSPACE_DIR = os.path.join(WEBUI_DIR, "workspace")
os.makedirs(TASKS_DIR, exist_ok=True)
os.makedirs(WORKSPACE_DIR, exist_ok=True)

MAX_ROWS_PER_FILE = 5000
RUN_LOCK = threading.RLock()
RUNS = {}


@lru_cache(maxsize=128)
def _count_rows(files):
    total = 0
    for path, mtime_ns, size in files:
        with open(path, encoding="utf-8", errors="replace") as source:
            total += sum(bool(line.strip()) for line in source)
    return total


def count_jsonl_rows(paths):
    """Reuse counts until a file's size or modification time changes."""
    files = []
    for path in paths:
        path = Path(path).resolve()
        info = path.stat()
        files.append((str(path), info.st_mtime_ns, info.st_size))
    return _count_rows(tuple(sorted(files)))


def terminate_process_tree(proc, force=False):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        # terminate()/kill() only stop the parent on Windows. Workers must stop too.
        result = subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode and proc.poll() is None:
            raise RuntimeError(f"无法停止任务进程树（taskkill: {result.returncode}）")
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass


class PipelineStopped(Exception):
    pass


def check_stopped(task_id):
    run = RUNS.get(task_id)
    if run and run["cancel"].is_set():
        raise PipelineStopped()


def validate_dataset_name(name):
    name = (name or "").strip()
    if not name or name in (".", "..") or any(c in name for c in "/\\\0") or name.startswith("."):
        raise ValueError("请输入有效的数据集名称，不能包含路径分隔符或以点开头")
    return name


class Task:
    def __init__(self, task_id: str = None):
        if task_id:
            if not task_id.startswith("task_") or not task_id[5:].isdigit():
                raise ValueError("无效的任务编号")
            self.task_id = task_id
        else:
            self.task_id = self._next_task_id()
        self.task_dir = os.path.join(TASKS_DIR, self.task_id)
        self.input_dir = os.path.join(self.task_dir, "input")
        self.split_dir = os.path.join(self.task_dir, "split")
        self.work_dir = os.path.join(self.task_dir, "work")
        self.output_dir = os.path.join(self.task_dir, "output")
        self.log_file = os.path.join(self.task_dir, "task.log")
        self.config_file = os.path.join(self.task_dir, "config.json")
        self.status_file = os.path.join(self.task_dir, "status.json")
        self.name_file = os.path.join(self.task_dir, "name.txt")

    @staticmethod
    def _next_task_id() -> str:
        nums = []
        if os.path.exists(TASKS_DIR):
            for d in os.listdir(TASKS_DIR):
                if os.path.isdir(os.path.join(TASKS_DIR, d)) and d.startswith("task_"):
                    try:
                        nums.append(int(d.split("_", 1)[1]))
                    except ValueError:
                        pass
        return f"task_{max(nums, default=0) + 1}"

    def init(self):
        for d in [self.task_dir, self.input_dir, self.split_dir, self.work_dir, self.output_dir]:
            os.makedirs(d, exist_ok=True)
        self.set_status("created")

    def set_status(self, status: str, step: str = "", detail: str = ""):
        if status not in ("stopped", "stopping"):
            check_stopped(self.task_id)
        info = {"status": status, "step": step, "detail": detail, "timestamp": time.time()}
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.task_dir, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(info, f)
            os.replace(tmp_path, self.status_file)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def get_status(self) -> dict:
        if not os.path.exists(self.status_file):
            return {"status": "unknown"}
        try:
            with open(self.status_file, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"status": "unknown"}

    def save_config(self, config: dict):
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def load_config(self) -> dict:
        if not os.path.exists(self.config_file):
            return {}
        with open(self.config_file, encoding="utf-8") as f:
            return json.load(f)

    def get_display_name(self) -> str:
        if os.path.exists(self.name_file):
            with open(self.name_file, "r", encoding="utf-8") as f:
                name = f.read().strip()
                if name:
                    return name
        if self.task_id.startswith("task_"):
            return f"任务 {self.task_id.split('_', 1)[1]}"
        return self.task_id

    def set_display_name(self, name: str):
        os.makedirs(self.task_dir, exist_ok=True)
        with open(self.name_file, "w", encoding="utf-8") as f:
            f.write(name.strip())

    def stop(self):
        with RUN_LOCK:
            run = RUNS.get(self.task_id)
            if not run:
                self.set_status("stopped", detail="任务已停止")
                return
            run["cancel"].set()
            self.set_status("stopping", detail="正在停止进程，请稍候")
            proc = run.get("proc")
            if proc and proc.poll() is None:
                terminate_process_tree(proc)

    def get_log(self) -> str:
        if not os.path.exists(self.log_file):
            return ""
        with open(self.log_file, "rb") as f:
            f.seek(max(0, os.path.getsize(self.log_file) - 16000))
            return f.read().decode("utf-8", errors="replace")

    def count_input_files(self) -> int:
        return len(glob_module.glob(os.path.join(self.input_dir, "*.jsonl")))

    def count_input_rows(self) -> int:
        return count_jsonl_rows(Path(self.input_dir).glob("*.jsonl"))

    def count_split_files(self) -> int:
        result_dir = os.path.join(self.split_dir, "result")
        if not os.path.exists(result_dir):
            return 0
        return len(glob_module.glob(os.path.join(result_dir, "*.jsonl")))

    def get_output_archive(self) -> str:
        if not os.path.isdir(self.output_dir) or not os.listdir(self.output_dir):
            return ""
        archive_path = os.path.join(self.task_dir, f"result_{self.task_id}")
        shutil.make_archive(archive_path, "zip", self.output_dir)
        return archive_path + ".zip"


def receive_files(files, task_id: str = None) -> Task:
    task = Task(task_id)
    task.init()
    if files is None:
        return task
    for f in files:
        dest = os.path.join(task.input_dir, os.path.basename(f.name))
        shutil.copy2(f.name, dest)
    return task


def receive_directory(dir_path: str, task_id: str = None) -> Task:
    task = Task(task_id)
    task.init()
    if not dir_path or not os.path.isdir(dir_path):
        return task
    for f in glob_module.glob(os.path.join(dir_path, "*.jsonl")):
        dest = os.path.join(task.input_dir, os.path.basename(f))
        shutil.copy2(f, dest)
    return task


def run_script(cmd: list, log_file: str, env: dict = None, timeout: int = None) -> int:
    task_id = os.path.basename(os.path.dirname(log_file))
    check_stopped(task_id)
    cmd = [sys.executable if part == "python" and i == 0 else part for i, part in enumerate(cmd)]
    full_env = os.environ.copy()
    # Captured logs are read as UTF-8 by the UI, including on Chinese Windows.
    full_env["PYTHONIOENCODING"] = "utf-8"
    full_env["DATATROVE_COLORIZE_LOGS"] = "false"
    full_env.setdefault("NLTK_DATA", os.path.join(WEBUI_DIR, ".cache", "nltk_data"))
    if env:
        full_env.update(env)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n{'='*60}\n")
        log.write(f"[CMD] {' '.join(cmd)}\n")
        log.write(f"{'='*60}\n")
        log.flush()
        with RUN_LOCK:
            check_stopped(task_id)
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                env=full_env, cwd=PROJECT_ROOT, start_new_session=True,
            )
            if task_id in RUNS:
                RUNS[task_id]["proc"] = proc
        try:
            started = time.monotonic()
            while proc.poll() is None:
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    cancelled = RUNS.get(task_id, {}).get("cancel")
                    if (cancelled and cancelled.is_set()) or (timeout and time.monotonic() - started > timeout):
                        terminate_process_tree(proc, force=True)
                        proc.wait()
        finally:
            with RUN_LOCK:
                if task_id in RUNS:
                    RUNS[task_id]["proc"] = None
        log.write(f"\n[EXIT CODE] {proc.returncode}\n")
    check_stopped(task_id)
    return proc.returncode


def step_split(task: Task, workers: int = 4):
    task.set_status("running", "split", "Splitting input files...")
    n_input = task.count_input_rows()
    if n_input == 0:
        task.set_status("error", "split", "No input data found")
        return False
    if n_input <= MAX_ROWS_PER_FILE:
        task.set_status("running", "split", f"Skip split: {n_input} rows fits in one file")
        return True
    n_tasks = task.count_input_files()
    cmd = [
        "python", os.path.join(SCRIPTS_DIR, "merge_split", "split.py"),
        "--input_path", task.input_dir,
        "--output_path", task.split_dir,
        "--max_rows_per_file", str(MAX_ROWS_PER_FILE),
        "--tasks", str(n_tasks),
        "--workers", str(workers),
    ]
    ret = run_script(cmd, task.log_file)
    if ret != 0:
        task.set_status("error", "split", f"Split failed with exit code {ret}")
        return False
    task.set_status("running", "split", f"Split done: {task.count_split_files()} files")
    return True


def get_split_input_dir(task: Task) -> str:
    result_dir = os.path.join(task.split_dir, "result")
    if os.path.exists(result_dir) and len(glob_module.glob(os.path.join(result_dir, "*.jsonl"))) > 0:
        return result_dir
    return task.input_dir


def count_tasks_and_workers(input_dir: str, max_workers: int = 16) -> tuple:
    n_files = len(glob_module.glob(os.path.join(input_dir, "*.jsonl")))
    n_tasks = max(1, n_files)
    n_workers = min(n_tasks, max_workers)
    return n_tasks, n_workers


def step_data_clean(task: Task, config: dict, workers: int = 8):
    task.set_status("running", "data_clean", "Running data cleaning...")
    input_dir = get_split_input_dir(task)
    output_dir = os.path.join(task.work_dir, "data_clean")
    n_tasks, n_workers = count_tasks_and_workers(input_dir, workers)
    languages = config.get("languages", ["zh"])
    cmd = [
        "python", os.path.join(SCRIPTS_DIR, "data_clean", "data_cleaning.py"),
        "--input_path", input_dir,
        "--output_path", output_dir,
        "--tasks", str(n_tasks),
        "--workers", str(n_workers),
        "--languages",
    ] + languages
    ret = run_script(cmd, task.log_file)
    if ret != 0:
        task.set_status("error", "data_clean", f"Data cleaning failed with exit code {ret}")
        return False, input_dir
    # Only deduplicated outputs are final; never fall back to language/quality
    # intermediates or exclusion files. Flatten all selected languages uniquely.
    final_output = os.path.join(output_dir, "final")
    shutil.rmtree(final_output, ignore_errors=True)
    os.makedirs(final_output)
    for language in languages:
        pattern = os.path.join(output_dir, "3_minhash_deduplication", language, "4_result", "output", "*.jsonl")
        for path in sorted(glob_module.glob(pattern)):
            shutil.copy2(path, os.path.join(final_output, f"{language}_{os.path.basename(path)}"))
    task.set_status("running", "data_clean", "Data cleaning done")
    return True, final_output


def step_calc_fcd(task: Task, config: dict, input_dir: str, workers: int = 8):
    task.set_status("running", "calc_fcd", "Computing FCD scores...")
    output_dir = os.path.join(task.work_dir, "fcd")
    n_tasks, n_workers = count_tasks_and_workers(input_dir, workers)
    cmd = [
        "python", os.path.join(SCRIPTS_DIR, "calc_score", "calc_fcd.py"),
        "--input_path", input_dir,
        "--output_path", output_dir,
        "--tasks", str(n_tasks),
        "--workers", str(n_workers),
    ]
    fcd_params = config.get("fcd_params", {})
    for key in ["freq_scaling_factor", "w_f", "power_mean_alpha", "agg_top_quantile", "agg_top_weight", "noun_weight"]:
        if key in fcd_params:
            cmd.extend([f"--{key}", str(fcd_params[key])])
    ret = run_script(cmd, task.log_file)
    if ret != 0:
        task.set_status("error", "calc_fcd", f"FCD calculation failed with exit code {ret}")
        return False, ""
    score_path = os.path.join(output_dir, "fcd_score")
    # Scores are rank-aligned JSON arrays; preserve that order and expose scores
    # on output records so downloads and result previews include the calculation.
    annotated = os.path.join(output_dir, "annotated")
    shutil.rmtree(annotated, ignore_errors=True)
    os.makedirs(annotated)
    for rank, source in enumerate(sorted(Path(input_dir).glob("*.jsonl"))):
        with open(os.path.join(score_path, f"{rank:05d}.json"), encoding="utf-8") as f:
            scores = json.load(f)
        index = 0
        with source.open(encoding="utf-8") as src, (Path(annotated) / source.name).open("w", encoding="utf-8") as dest:
            for line in src:
                if not line.strip():
                    continue
                doc = json.loads(line)
                if index >= len(scores):
                    raise ValueError(f"FCD 分数数量与输入不一致：{source.name}，无法生成结果")
                doc.setdefault("metadata", {})["fcd_score"] = scores[index]
                dest.write(json.dumps(doc, ensure_ascii=False) + "\n")
                index += 1
        if index != len(scores):
            raise ValueError("FCD 分数数量与输入不一致，无法生成结果")
    task.set_status("running", "calc_fcd", "FCD scores computed")
    return True, score_path


def step_calc_cdf_gc(task: Task, config: dict, input_dir: str, workers: int = 8):
    task.set_status("running", "calc_cdf_gc", "Running CDF-GC pipeline...")
    output_dir = os.path.join(task.work_dir, "cdf_gc")
    n_tasks, n_workers = count_tasks_and_workers(input_dir, workers)
    gc_params = config.get("gc_params", {})
    cmd = [
        "python", os.path.join(SCRIPTS_DIR, "data_select", "cdf_gc.py"),
        "--input_path", input_dir,
        "--output_path", output_dir,
        "--tasks", str(n_tasks),
        "--workers", str(n_workers),
        "--sample_rate", str(gc_params.get("sample_rate", 0.2)),
        "--tokenizer_path", gc_params.get("tokenizer_path", ""),
        "--language", gc_params.get("language", "zh"),
    ]
    if gc_params.get("ltp_model_path"):
        cmd.extend(["--ltp_model_path", gc_params["ltp_model_path"]])
    if gc_params.get("workers_per_gpu"):
        cmd.extend(["--workers_per_gpu", str(gc_params["workers_per_gpu"])])
    cuda_devices = gc_params.get("cuda_visible_devices", "")
    env = {"CUDA_VISIBLE_DEVICES": cuda_devices} if cuda_devices else None
    ret = run_script(cmd, task.log_file, env=env)
    if ret != 0:
        task.set_status("error", "calc_cdf_gc", f"CDF-GC failed with exit code {ret}")
        return False, ""
    result_path = os.path.join(output_dir, "2_sampling", "2_sample_result")
    task.set_status("running", "calc_cdf_gc", "CDF-GC done")
    return True, result_path


def step_sample(task: Task, config: dict, input_dir: str, score_path: str, workers: int = 8):
    sample_config = config.get("sample", {})
    method = sample_config.get("method", "cdf")
    sample_rate = sample_config.get("sample_rate", 0.2)
    unit = sample_config.get("unit", "doc")
    task.set_status("running", "sample", f"Sampling ({method}, rate={sample_rate})...")
    output_dir = os.path.join(task.work_dir, "sample")
    n_tasks, n_workers = count_tasks_and_workers(input_dir, workers)

    if method == "cdf":
        script = os.path.join(SCRIPTS_DIR, "sample", "cdf_sample.py")
        cmd = [
            "python", script,
            "--input_path", input_dir,
            "--score_path", score_path,
            "--output_path", output_dir,
            "--sample_rate", str(sample_rate),
            "--unit", unit,
            "--tasks", str(n_tasks),
            "--workers", str(n_workers),
        ]
    elif method == "hard":
        script = os.path.join(SCRIPTS_DIR, "sample", "hard_sample.py")
        cmd = [
            "python", script,
            "--input_path", input_dir,
            "--score_path", score_path,
            "--output_path", output_dir,
            "--sample_rate", str(sample_rate),
            "--unit", unit,
            "--tasks", str(n_tasks),
            "--workers", str(n_workers),
        ]
    elif method == "random":
        script = os.path.join(SCRIPTS_DIR, "sample", "random_sample.py")
        cmd = [
            "python", script,
            "--input_path", input_dir,
            "--output_path", output_dir,
            "--sample_rate", str(sample_rate),
            "--unit", unit,
            "--tasks", str(n_tasks),
            "--workers", str(n_workers),
        ]
    else:
        task.set_status("error", "sample", f"Unknown sample method: {method}")
        return False, ""

    ret = run_script(cmd, task.log_file)
    if ret != 0:
        task.set_status("error", "sample", f"Sampling failed with exit code {ret}")
        return False, ""
    result_path = os.path.join(output_dir, "result")
    task.set_status("running", "sample", "Sampling done")
    return True, result_path


def step_merge_output(task: Task, result_dir: str, workers: int = 4):
    task.set_status("running", "merge_output", "Merging output files...")
    jsonl_files = sorted(Path(result_dir).rglob("*.jsonl"))
    shutil.rmtree(task.output_dir, ignore_errors=True)
    os.makedirs(task.output_dir)
    # Stream into bounded shards, including a valid empty result.
    index = 0
    out = open(os.path.join(task.output_dir, "00000.jsonl"), "w", encoding="utf-8")
    try:
        for path in jsonl_files:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    check_stopped(task.task_id)
                    if index and index % MAX_ROWS_PER_FILE == 0:
                        out.close()
                        out = open(os.path.join(task.output_dir, f"{index // MAX_ROWS_PER_FILE:05d}.jsonl"), "w", encoding="utf-8")
                    out.write(line.rstrip("\n") + "\n")
                    index += 1
    finally:
        out.close()
    task.set_status("running", "merge_output", "Output merged")
    return True


def find_deepest_output(base_dir: str) -> str:
    for name in ["output", "result", "2_sample_result"]:
        for root, dirs, files in os.walk(base_dir):
            if os.path.basename(root) == name:
                if any(f.endswith(".jsonl") for f in files):
                    return root
    for root, dirs, files in os.walk(base_dir):
        if any(f.endswith(".jsonl") for f in files):
            return root
    return base_dir


def run_pipeline(task: Task, config: dict, workers: int = 8):
    steps = config.get("steps", [])
    current_input_dir = get_split_input_dir(task)
    score_path = ""

    try:
        if config.get("rerun", True):
            for folder in (task.split_dir, task.work_dir, task.output_dir):
                shutil.rmtree(folder, ignore_errors=True)
                os.makedirs(folder)
            Path(task.log_file).write_text("", encoding="utf-8")
        task.save_config(config)
        if not step_split(task, workers):
            return
        current_input_dir = get_split_input_dir(task)

        if "data_clean" in steps:
            ok, new_input = step_data_clean(task, config, workers)
            if not ok:
                return
            current_input_dir = new_input
            if not any(Path(current_input_dir).glob("*.jsonl")):
                step_merge_output(task, current_input_dir, workers)
                task.set_status("completed", detail="清洗完成：所有文档均被过滤，输出为 0 条")
                return

        if "fcd" in steps:
            ok, score_path = step_calc_fcd(task, config, current_input_dir, workers)
            if not ok:
                return
            current_input_dir = os.path.join(task.work_dir, "fcd", "annotated")
            sample_config = config.get("sample", {})
            if sample_config.get("method") and sample_config.get("score_source") == "fcd":
                ok, result_dir = step_sample(task, config, current_input_dir, score_path, workers)
                if not ok:
                    return
                current_input_dir = result_dir

        if "cdf_gc" in steps:
            ok, result_dir = step_calc_cdf_gc(task, config, current_input_dir, workers)
            if not ok:
                return
            current_input_dir = result_dir

        if not step_merge_output(task, current_input_dir, workers):
            return

        task.set_status("completed", detail="Pipeline finished successfully")
    except PipelineStopped:
        task.set_status("stopped", detail="任务已停止，可从已完成阶段继续")
    except Exception as e:
        # Cancellation can occur while an exception is propagating.
        if RUNS.get(task.task_id, {}).get("cancel", threading.Event()).is_set():
            task.set_status("stopped", detail="任务已停止")
            return
        task.set_status("error", detail=str(e))
        import traceback
        with open(task.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[EXCEPTION]\n{traceback.format_exc()}\n")
    finally:
        with RUN_LOCK:
            RUNS.pop(task.task_id, None)


def run_pipeline_async(task: Task, config: dict, workers: int = 8):
    with RUN_LOCK:
        if RUNS:
            raise ValueError("已有任务正在运行或停止中，请等待完成")
        task.set_status("running", detail="任务已提交")
        RUNS[task.task_id] = {"cancel": threading.Event(), "proc": None}
        thread = threading.Thread(target=run_pipeline, args=(task, config, workers), daemon=True)
        thread.start()
    return thread


def load_scores(task: Task, score_type: str = "fcd") -> list:
    if score_type == "fcd":
        score_dir = os.path.join(task.work_dir, "fcd", "fcd_score")
    elif score_type.startswith("gc:"):
        score_dir = os.path.join(task.work_dir, "cdf_gc", "1_gc_data", "6_normalized_gc")
    else:
        return []
    if not os.path.exists(score_dir):
        return []
    scores = []
    if score_type == "fcd":
        for path in sorted(Path(score_dir).glob("*.json")):
            with path.open(encoding="utf-8") as source:
                scores.extend(float(value) for value in json.load(source))
        return scores
    for f in sorted(glob_module.glob(os.path.join(score_dir, "*.jsonl"))):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    item = json.loads(line)
                    score = item.get("normalized_gc", {}).get(score_type.split(":")[-1])
                    for key in ("score", "fcd_score", "gc_score"):
                        if key in item and item[key] is not None:
                            score = item[key]
                            break
                    if score is not None:
                        scores.append(float(score))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
    return scores


def plot_score_distribution(scores: list, title: str = "Score Distribution"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not scores:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No scores available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#9ca3af")
        ax.set_facecolor("#fafafa")
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    scores = np.array(scores)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    counts, edges = np.histogram(scores, bins=100)
    probability = counts / counts.sum()
    density = probability / np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2

    ax1.plot(centers, density, color="#4f46e5", linewidth=1.5)
    ax1.set_xlabel("Score", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title(f"{title} — PDF (100 bins)", fontsize=11, fontweight="bold")
    ax1.set_xlim(edges[0], edges[-1])
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    cdf = np.cumsum(counts) / counts.sum()
    ax2.plot(edges, np.r_[0, cdf], color="#dc2626", linewidth=1.5)
    ax2.set_xlabel("Score", fontsize=10)
    ax2.set_ylabel("CDF", fontsize=10)
    ax2.set_title(f"{title} — CDF (100 bins)", fontsize=11, fontweight="bold")
    ax2.set_xlim(edges[0], edges[-1])
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    stats_text = f"n={len(scores):,}\nmean={scores.mean():.4f}\nstd={scores.std():.4f}\nmedian={np.median(scores):.4f}"
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes, fontsize=8,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0ff", alpha=0.8, edgecolor="#c7d2fe"))

    fig.tight_layout(pad=2.0)
    return fig


def delete_task(task_id: str) -> bool:
    task_dir = os.path.join(TASKS_DIR, task_id)
    if not os.path.exists(task_dir):
        return False
    shutil.rmtree(task_dir, ignore_errors=True)
    return True


def list_tasks() -> list:
    tasks = []
    if not os.path.exists(TASKS_DIR):
        return tasks
    for tid in sorted(os.listdir(TASKS_DIR)):
        task_path = os.path.join(TASKS_DIR, tid)
        if not os.path.isdir(task_path):
            continue
        t = Task(tid)
        status = t.get_status()
        tasks.append({
            "task_id": tid,
            "status": status.get("status", "unknown"),
            "step": status.get("step", ""),
            "detail": status.get("detail", ""),
            "display_name": t.get_display_name(),
        })
    return tasks


# ── Workspace management ──

def stage_upload(files=None, dir_path=None) -> str:
    """Copy uploaded files to a staging directory. Returns staging path."""
    staging = tempfile.mkdtemp(dir=WORKSPACE_DIR, prefix=".staging_")
    def target_path(name):
        target = Path(staging) / (Path(name).stem + ".jsonl")
        index = 1
        while target.exists():
            target = Path(staging) / f"{Path(name).stem}_{index}.jsonl"
            index += 1
        return target

    try:
        if files:
            for f in files:
                fname = f.name if hasattr(f, "name") else str(f)
                if fname.lower().endswith(".zip"):
                    with zipfile.ZipFile(fname) as zf:
                        for member in zf.namelist():
                            if member.lower().endswith(".jsonl") and "__MACOSX" not in member:
                                with zf.open(member) as src_f, target_path(member).open("wb") as dst_f:
                                    shutil.copyfileobj(src_f, dst_f)
                elif fname.lower().endswith(".jsonl"):
                    shutil.copy2(fname, target_path(fname))
                else:
                    raise ValueError("只支持 JSONL 文件或包含 JSONL 的 ZIP")
        elif dir_path and os.path.isdir(dir_path):
            for path in sorted(Path(dir_path).rglob("*.jsonl")):
                shutil.copy2(path, target_path(path.name))
        else:
            raise ValueError("请上传文件，或填写存在的服务器目录")
        rows = 0
        for path in Path(staging).glob("*"):
            with path.open(encoding="utf-8") as source:
                for number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path.name} 第 {number} 行不是有效 JSON") from exc
                    if not isinstance(doc, dict) or not isinstance(doc.get("text"), str) or not doc["text"].strip():
                        raise ValueError(f"{path.name} 第 {number} 行需要非空的 text 字符串")
                    if "metadata" in doc and not isinstance(doc["metadata"], dict):
                        raise ValueError(f"{path.name} 第 {number} 行的 metadata 必须是 JSON 对象")
                    rows += 1
        if not rows:
            raise ValueError("未找到有效记录，请上传包含 text 字段的 JSONL 文件")
    except Exception:
        cancel_upload(staging)
        raise
    return staging


def confirm_upload(staging_dir: str, name: str) -> str:
    """Move staged files to workspace under the given name. Returns final name."""
    name = validate_dataset_name(name)
    dest = os.path.join(WORKSPACE_DIR, name)
    if os.path.exists(dest):
        i = 1
        while os.path.exists(f"{dest}_{i}"):
            i += 1
        dest = f"{dest}_{i}"
        name = os.path.basename(dest)
    os.rename(staging_dir, dest)
    return name


def cancel_upload(staging_dir: str):
    """Remove staged files."""
    if staging_dir and os.path.exists(staging_dir):
        shutil.rmtree(staging_dir, ignore_errors=True)


def rename_dataset(old_name: str, new_name: str) -> bool:
    """Rename a workspace dataset directory."""
    old_name, new_name = validate_dataset_name(old_name), validate_dataset_name(new_name)
    if old_name == new_name:
        return False
    old_path = os.path.join(WORKSPACE_DIR, old_name)
    new_path = os.path.join(WORKSPACE_DIR, new_name)
    if not os.path.exists(old_path) or os.path.exists(new_path):
        return False
    os.rename(old_path, new_path)
    return True


def list_workspace_datasets():
    datasets = []
    if not os.path.exists(WORKSPACE_DIR):
        return datasets
    for name in sorted(os.listdir(WORKSPACE_DIR), reverse=True):
        if name.startswith("."):
            continue
        dpath = os.path.join(WORKSPACE_DIR, name)
        if not os.path.isdir(dpath):
            continue
        files = glob_module.glob(os.path.join(dpath, "*.jsonl"))
        if not files:
            continue
        datasets.append({"name": name, "n_files": len(files), "path": dpath})
    return datasets


def create_task_from_workspace(dataset_name: str) -> Task:
    src = os.path.join(WORKSPACE_DIR, validate_dataset_name(dataset_name))
    if not glob_module.glob(os.path.join(src, "*.jsonl")):
        raise ValueError("数据集不存在或没有 JSONL 文件，请重新选择")
    task = Task()
    task.init()
    src = os.path.join(WORKSPACE_DIR, dataset_name)
    if os.path.isdir(src):
        for f in glob_module.glob(os.path.join(src, "*.jsonl")):
            shutil.copy2(f, os.path.join(task.input_dir, os.path.basename(f)))
    return task


def save_output_as_dataset(task: Task, dataset_name: str) -> str:
    dataset_name = validate_dataset_name(dataset_name)
    jsonl_files = glob_module.glob(os.path.join(task.output_dir, "*.jsonl"))
    if not jsonl_files or not any(os.path.getsize(f) for f in jsonl_files):
        return ""
    dest = os.path.join(WORKSPACE_DIR, dataset_name)
    if os.path.exists(dest):
        i = 1
        while os.path.exists(f"{dest}_{i}"):
            i += 1
        dest = f"{dest}_{i}"
        dataset_name = os.path.basename(dest)
    os.makedirs(dest, exist_ok=True)
    if not jsonl_files:
        return ""
    for f in jsonl_files:
        shutil.copy2(f, os.path.join(dest, os.path.basename(f)))
    return dataset_name
