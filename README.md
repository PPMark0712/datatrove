# Datatrove (Extended)

基于 [HuggingFace Datatrove](https://github.com/huggingface/datatrove) 框架扩展的数据筛选与采样工具集，用于大规模文本数据的质量评估和智能采样。

> 原始上游 README 见 [README_upstream.md](README_upstream.md)

## 安装

```bash
conda create -n datatrove python=3.12
conda activate datatrove

# 基础安装（FCD + 采样 + 合并拆分）
pip install -e .

# 含 WebUI
pip install -e ".[webui]"

# 含 GPU 功能（CDF-GC 依存分析）
pip install -e ".[webui,gpu]"

# 含数据清洗（语言过滤需要 fasttext）
pip install -e ".[processing]"
```

## 下载 FCD 词典（最新版 Release）

在仓库根目录执行以下整段命令，即可下载、校验并解压到 `src/datatrove/pipeline/fcd/build_dict/data/`。需要已安装 Python；Linux 另需 curl。ZIP 约 91 MB，整个 data 与 dist 目录均不进入 Git。

Linux / macOS：

```bash
mkdir -p dist && \
curl --fail --location --retry 3 --output dist/fcd_data.zip \
  https://github.com/PPMark0712/datatrove/releases/latest/download/fcd_data.zip && \
python3 src/datatrove/pipeline/fcd/build_dict/package_data.py --validate dist/fcd_data.zip && \
python3 -m zipfile -e dist/fcd_data.zip src/datatrove/pipeline/fcd/build_dict
```

Windows PowerShell（先激活 datatrove 环境）：

```powershell
& {
    $ErrorActionPreference = 'Stop'
    New-Item -ItemType Directory -Force -Path dist | Out-Null
    Invoke-WebRequest -Uri 'https://github.com/PPMark0712/datatrove/releases/latest/download/fcd_data.zip' -OutFile dist/fcd_data.zip
    python src/datatrove/pipeline/fcd/build_dict/package_data.py --validate dist/fcd_data.zip
    if ($LASTEXITCODE -ne 0) { throw 'FCD ZIP 校验失败，未解压' }
    Expand-Archive -LiteralPath dist/fcd_data.zip -DestinationPath src/datatrove/pipeline/fcd/build_dict -Force
}
```

这些命令使用 GitHub 标记为 Latest 的版本 Release，更新本地词典，不需要本地构建。首次使用前维护者必须上传资源源 ZIP 并发布一个带附件的版本 Release；如果下载返回 404，请检查 [Releases](https://github.com/PPMark0712/datatrove/releases)。NLTK 语料仍需单独安装。

维护者的一次性资源上传及自动发布流程见 [FCD 资源说明](src/datatrove/pipeline/fcd/build_dict/README.md)。

## 扩展功能

| 模块 | 说明 | 文档 |
|------|------|------|
| **FCD** | 英文文本词汇难度评分（词频 + WordNet 概念距离） | [docs/fcd.md](docs/fcd.md) |
| **CDF-GC** | 文本生成复杂度评分 + CDF 采样（5 维语法指标） | [docs/cdf_gc.md](docs/cdf_gc.md) |
| **Samplers** | 通用采样框架：CDF / Hard / Random，支持按文档数或 token 数 | [docs/sampler.md](docs/sampler.md) |
| **Data Cleaning** | 数据清洗：语言过滤 + 质量过滤 + MinHash 去重 | [docs/data_cleaning.md](docs/data_cleaning.md) |
| **Merge & Split** | 文件合并与拆分 | `pipeline/merge_split/` |
| **WebUI** | Web 数据筛选平台（上传 → 配置 → 处理 → 下载） | [docs/webui.md](docs/webui.md) |

## 快速使用

### 计算 FCD 分数

```bash
python scripts/calc_score/calc_fcd.py \
    --input_path /path/to/jsonl_folder \
    --output_path /path/to/output \
    --tasks 1 --workers 1
```

### CDF-GC 筛选（完整流水线）

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/data_select/cdf_gc.py \
    --input_path /path/to/jsonl_folder \
    --output_path /path/to/output \
    --tasks 64 --workers 32 \
    --workers_per_gpu 4 \
    --sample_rate 0.2 \
    --ltp_model_path /path/to/LTP/small \
    --tokenizer_path /path/to/tokenizer.json
```

### 采样

```bash
# CDF 采样（基于预计算分数）
python scripts/sample/cdf_sample.py \
    --input_path /path/to/data --score_path /path/to/scores \
    --output_path /path/to/output --sample_rate 0.2

# Hard 采样（取 top）
python scripts/sample/hard_sample.py \
    --input_path /path/to/data --score_path /path/to/scores \
    --output_path /path/to/output --sample_rate 0.2

# 随机采样
python scripts/sample/random_sample.py \
    --input_path /path/to/data \
    --output_path /path/to/output --sample_rate 0.2
```

### 文件合并与拆分

```bash
# 拆分为指定文件数
python scripts/merge_split/split.py \
    --input_path /path/to/data --output_path /path/to/output \
    --output_file_count 10

# 拆分，每文件最多 N 行
python scripts/merge_split/split.py \
    --input_path /path/to/data --output_path /path/to/output \
    --max_rows_per_file 5000

# 合并为指定文件数
python scripts/merge_split/merge.py \
    --input_path /path/to/data --output_path /path/to/output \
    --output_file_count 4
```

## 全部脚本

| 脚本 | 功能 |
|------|------|
| `scripts/calc_score/calc_fcd.py` | 计算 FCD 分数 |
| `scripts/calc_score/calc_ppl.py` | 计算困惑度 |
| `scripts/calc_score/count_tokens.py` | 统计 token 数 |
| `scripts/calc_score/fineweb_edu.py` | 计算 FineWeb-Edu 分数 |
| `scripts/data_select/cdf_gc.py` | CDF-GC 完整流水线 |
| `scripts/data_select/eta_dacp.py` | ETA-DACP 数据筛选 |
| `scripts/sample/cdf_sample.py` | CDF 采样 |
| `scripts/sample/hard_sample.py` | Hard 采样 |
| `scripts/sample/random_sample.py` | 随机采样 |
| `scripts/merge_split/split.py` | 文件拆分 |
| `scripts/merge_split/merge.py` | 文件合并 |
| `scripts/sort/sort_data_by_score.py` | 按分数排序 |
| `scripts/data_clean/data_cleaning.py` | 数据清洗 |

## 通用参数

大多数脚本共享以下参数（`merge.py` 使用独立参数，见 `--help`）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_path` | 输入 JSONL 文件夹 | — |
| `--glob_pattern` | 文件匹配模式 | 全部 |
| `--output_path` | 输出文件夹 | — |
| `--tasks` | 任务数（数据分片数） | 32 |
| `--workers` | 并行 worker 数 | 32 |
| `--limit` | 每 worker 最大处理数（测试用） | -1 |
| `--rerun` | 忽略检查点重跑 | False |
