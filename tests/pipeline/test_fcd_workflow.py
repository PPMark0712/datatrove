"""Exercise Release shell steps with a local GitHub CLI double, without publishing."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def run_workflow(tmp_path, *, exists=False, assets=False, source_exists=True):
    yaml = pytest.importorskip("yaml")
    bash = shutil.which("bash")
    if os.name == "nt":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(candidate) if candidate.exists() else None
    if not bash:
        pytest.skip("Bash is needed to exercise GitHub Actions shell steps")
    from datatrove.pipeline.fcd.build_dict.package_data import RESOURCE_FILES, package

    data = tmp_path / "data"
    data.mkdir()
    for name in RESOURCE_FILES:
        (data / name).write_text("resource", encoding="utf-8")
    archive = package(data, tmp_path / "source.zip")
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"exists": exists, "assets": ["fcd_data.zip"] if assets else [],
                                      "source_exists": source_exists, "calls": []}))
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text('''#!/usr/bin/env python
import json, os, shutil, sys
from pathlib import Path
path = Path(os.environ["MOCK_STATE"])
state = json.loads(path.read_text())
args = sys.argv[1:]
state["calls"].append(args)
def finish(code=0):
    path.write_text(json.dumps(state))
    sys.exit(code)
command = args[1]
if command == "download":
    if not state["source_exists"]:
        finish(1)
    shutil.copyfile(os.environ["MOCK_ZIP"], Path(args[args.index("--dir")+1])/"fcd_data.zip")
elif command == "view":
    if not state["exists"]:
        finish(1)
    print("\\n".join(state["assets"]) if "--json" in args else "release")
elif command == "create":
    assert "--draft" in args and "--verify-tag" in args
    state["exists"] = True
    state["draft"] = True
elif command == "upload":
    assert Path(args[3]).is_file()
    state["assets"].append("fcd_data.zip")
elif command == "edit":
    assert "fcd_data.zip" in state["assets"]
    assert "--draft=false" in args
    state["draft"] = False
else:
    finish(2)
finish()
''', encoding="utf-8", newline="\n")
    gh.chmod(0o755)
    env = {**os.environ, "PATH": str(binary) + os.pathsep + str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
           "TARGET_TAG": "v1.0.0", "SOURCE_TAG": "fcd-data", "GH_REPO": "example/repo",
           "RESOURCE_PATH": str(tmp_path / "download/fcd_data.zip"), "RUNNER_TEMP": str(tmp_path),
           "GITHUB_ENV": str(tmp_path / "github-env"), "MOCK_STATE": str(state_file), "MOCK_ZIP": str(archive)}
    workflow = yaml.load((ROOT / ".github/workflows/fcd-data-release.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["on"]["push"]["tags"] == ["v*"]
    assert "workflow_call" in workflow["on"]
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    result = None
    for step in workflow["jobs"]["attach-data"]["steps"]:
        if "run" not in step:
            continue
        if "if" in step and env.get("CREATED_RELEASE") != "true":
            continue
        result = subprocess.run([bash, "-e", "-c", step["run"]], cwd=ROOT, env=env,
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        if result.returncode:
            break
        if Path(env["GITHUB_ENV"]).exists():
            for line in Path(env["GITHUB_ENV"]).read_text().splitlines():
                key, value = line.split("=", 1)
                env[key] = value
    return result, json.loads(state_file.read_text())


def test_new_release_is_published_only_after_upload(tmp_path):
    result, state = run_workflow(tmp_path)
    assert result.returncode == 0, result.stderr
    commands = [call[1] for call in state["calls"]]
    assert commands.index("download") < commands.index("create") < commands.index("upload") < commands.index("edit")
    assert state["draft"] is False


def test_existing_release_gets_missing_asset(tmp_path):
    result, state = run_workflow(tmp_path, exists=True)
    assert result.returncode == 0, result.stderr
    assert state["assets"] == ["fcd_data.zip"]
    assert not any(call[1] in ("create", "edit") for call in state["calls"])


def test_existing_asset_is_not_replaced(tmp_path):
    result, state = run_workflow(tmp_path, exists=True, assets=True)
    assert result.returncode == 0, result.stderr
    assert not any(call[1] in ("upload", "create", "edit") for call in state["calls"])


def test_missing_source_does_not_create_empty_release(tmp_path):
    result, state = run_workflow(tmp_path, source_exists=False)
    assert result.returncode != 0
    assert state["exists"] is False
    assert not any(call[1] in ("upload", "create", "edit") for call in state["calls"])
