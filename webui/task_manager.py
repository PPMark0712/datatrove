import os
import json
import shutil
import uuid
import glob as glob_module
import subprocess
import threading
import time


WEBUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WEBUI_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
TASKS_DIR = os.path.join(WEBUI_DIR, "tasks")
os.makedirs(TASKS_DIR, exist_ok=True)

MAX_ROWS_PER_FILE = 5000


class Task:
    def __init__(self, task_id: str = None):
        self.task_id = task_id or str(uuid.uuid4())[:8]
        self.task_dir = os.path.join(TASKS_DIR, self.task_id)
        self.input_dir = os.path.join(self.task_dir, "input")
        self.split_dir = os.path.join(self.task_dir, "split")
        self.work_dir = os.path.join(self.task_dir, "work")
        self.output_dir = os.path.join(self.task_dir, "output")
        self.log_file = os.path.join(self.task_dir, "task.log")
        self.config_file = os.path.join(self.task_dir, "config.json")
        self.status_file = os.path.join(self.task_dir, "status.json")

    def init(self):
        for d in [self.task_dir, self.input_dir, self.split_dir, self.work_dir, self.output_dir]:
            os.makedirs(d, exist_ok=True)
        self.set_status("created")

    def set_status(self, status: str, step: str = "", detail: str = ""):
        info = {"status": status, "step": step, "detail": detail, "timestamp": time.time()}
        with open(self.status_file, "w") as f:
            json.dump(info, f)

    def get_status(self) -> dict:
        if not os.path.exists(self.status_file):
            return {"status": "unknown"}
        with open(self.status_file) as f:
            return json.load(f)

    def save_config(self, config: dict):
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def load_config(self) -> dict:
        if not os.path.exists(self.config_file):
            return {}
        with open(self.config_file) as f:
            return json.load(f)

    def get_log(self) -> str:
        if not os.path.exists(self.log_file):
            return ""
        with open(self.log_file, "r") as f:
            return f.read()

    def count_input_files(self) -> int:
        return len(glob_module.glob(os.path.join(self.input_dir, "*.jsonl")))

    def count_input_rows(self) -> int:
        total = 0
        for f in glob_module.glob(os.path.join(self.input_dir, "*.jsonl")):
            with open(f) as fh:
                total += sum(1 for _ in fh)
        return total

    def count_split_files(self) -> int:
        result_dir = os.path.join(self.split_dir, "result")
        if not os.path.exists(result_dir):
            return 0
        return len(glob_module.glob(os.path.join(result_dir, "*.jsonl")))

    def get_output_archive(self) -> str:
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


def run_script(cmd: list, log_file: str, env: dict = None) -> int:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    with open(log_file, "a") as log:
        log.write(f"\n{'='*60}\n")
        log.write(f"[CMD] {' '.join(cmd)}\n")
        log.write(f"{'='*60}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            env=full_env, cwd=PROJECT_ROOT
        )
        proc.wait()
        log.write(f"\n[EXIT CODE] {proc.returncode}\n")
    return proc.returncode


def step_split(task: Task, workers: int = 4):
    task.set_status("running", "split", "Splitting input files...")
    n_input = task.count_input_rows()
    if n_input == 0:
        task.set_status("error", "split", "No input data found")
        return False
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
    final_output = find_deepest_output(output_dir)
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
    jsonl_files = []
    for root, dirs, files in os.walk(result_dir):
        for f in files:
            if f.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, f))
    if not jsonl_files:
        task.set_status("error", "merge_output", "No jsonl files found in result directory")
        return False
    flat_dir = os.path.join(task.work_dir, "merge_input_flat")
    os.makedirs(flat_dir, exist_ok=True)
    for i, f in enumerate(jsonl_files):
        shutil.copy2(f, os.path.join(flat_dir, f"{i:06d}.jsonl"))
    cmd = [
        "python", os.path.join(SCRIPTS_DIR, "merge_split", "merge.py"),
        "--input_path", flat_dir,
        "--output_path", os.path.join(task.work_dir, "merge"),
        "--rows_per_file", str(MAX_ROWS_PER_FILE),
        "--workers", str(workers),
    ]
    ret = run_script(cmd, task.log_file)
    merge_result = os.path.join(task.work_dir, "merge", "result")
    if ret != 0 or not os.path.exists(merge_result):
        for f in jsonl_files:
            shutil.copy2(f, task.output_dir)
    else:
        for f in glob_module.glob(os.path.join(merge_result, "*.jsonl")):
            shutil.copy2(f, task.output_dir)
    task.set_status("running", "merge_output", "Output merged")
    return True


def find_deepest_output(base_dir: str) -> str:
    """Find the deepest 'output' or 'result' directory containing jsonl files."""
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
    """Execute the full pipeline based on config."""
    task.save_config(config)
    steps = config.get("steps", [])
    current_input_dir = get_split_input_dir(task)
    score_path = ""

    try:
        if not step_split(task, workers):
            return
        current_input_dir = get_split_input_dir(task)

        if "data_clean" in steps:
            ok, new_input = step_data_clean(task, config, workers)
            if not ok:
                return
            current_input_dir = new_input

        if "fcd" in steps:
            ok, score_path = step_calc_fcd(task, config, current_input_dir, workers)
            if not ok:
                return
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
    except Exception as e:
        task.set_status("error", detail=str(e))
        with open(task.log_file, "a") as f:
            import traceback
            f.write(f"\n[EXCEPTION]\n{traceback.format_exc()}\n")


def run_pipeline_async(task: Task, config: dict, workers: int = 8):
    thread = threading.Thread(target=run_pipeline, args=(task, config, workers), daemon=True)
    thread.start()
    return thread


def load_scores(task: Task, score_type: str = "fcd") -> list:
    """Load all scores from a task's score directory."""
    if score_type == "fcd":
        score_dir = os.path.join(task.work_dir, "fcd", "fcd_score")
    elif score_type == "gc":
        score_dir = os.path.join(task.work_dir, "cdf_gc", "1_gc_data", "6_normalized_gc")
    else:
        return []
    if not os.path.exists(score_dir):
        return []
    scores = []
    for f in sorted(glob_module.glob(os.path.join(score_dir, "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                try:
                    item = json.loads(line)
                    score = None
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
    """Generate PDF + CDF plot from scores. Returns a matplotlib Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not scores:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No scores available", ha="center", va="center", transform=ax.transAxes)
        return fig

    scores = np.array(scores)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # PDF (histogram + KDE-like smooth)
    ax1.hist(scores, bins=80, density=True, alpha=0.6, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Density")
    ax1.set_title(f"{title} — PDF")

    # CDF
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax2.plot(sorted_scores, cdf, color="#C44E52", linewidth=1.5)
    ax2.set_xlabel("Score")
    ax2.set_ylabel("CDF")
    ax2.set_title(f"{title} — CDF")
    ax2.grid(True, alpha=0.3)

    # Stats annotation
    stats_text = f"n={len(scores)}\nmean={scores.mean():.4f}\nstd={scores.std():.4f}\nmedian={np.median(scores):.4f}"
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes, fontsize=8,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    return fig


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
        })
    return tasks
