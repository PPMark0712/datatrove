# CDF-GC

这是论文《Data-Efficient Selection via Grammatical Complexity in Continual Pre-training of Domain-Specific LLMs》的代码实现。

* **论文链接**：暂未更新。
* **支持语言**：目前仅支持中文数据筛选。如需扩展至其他语言，请修改 `src/datatrove/pipeline/cdf_gc/dependency_parser.py` 和 `part_of_speech_predictor.py`。
* **注意**：本项目基于 Hugging Face `datatrove` 框架，本 README 仅包含本项目特有的信息，不包含 `datatrove` 框架本身的说明。

## 环境配置
为了运行本项目，请先将 `datatrove` 库安装到您的 Python 环境中。具体的依赖项都列在了 `pyproject.toml` 文件里。

1、下载代码仓库
```bash
git clone https://github.com/PPMark0712/CDF-GC.git
```

2、创建并激活 Conda 环境
```bash
conda create -n cdf_gc python=3.12
conda activate cdf_gc
```

3、安装 datatrove 包

```bash
cd CDF-GC
pip install -e .[cdf_gc]
```
请注意，`.[cdf_gc]` 会安装 `pyproject.toml` 文件中为本项目指定的额外依赖项。

4、下载预训练模型

- LTP 模型：请从 Hugging Face Hub 下载 [LTP/small](https://huggingface.co/LTP/small) 模型。

- 待训练的 LLM：下载您计划训练的大型语言模型，因为后续步骤需要使用它的分词器。

## 启动

请运行以下 `bash` 脚本来启动程序：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/cdf_gc.py \
	--input_path /path/to/jsonl_folder \
	--glob_pattern *.jsonl \
	--output_path /path/to/output_folder \
	--tasks 64 \
	--workers 32 \
	--workers_per_gpu 4 \
	--sample_rate 0.2 \
	--ltp_model_path /path/to/LTP/small \
	--tokenizer_path /path/to/tokenizer.json
```

## 参数说明

| 参数 | 类型 | 描述 | 默认值 | 必填 |
| :--- | :--- | :--- | :--- | :--- |
| `--input_path` | 路径 | 输入的 JSONL 文件夹路径。 | 无 | 是 |
| `--glob_pattern` | 正则表达式 | 用于匹配输入文件的通配符，默认为当前路径下的所有文件。 | 无 | 否 |
| `--output_path` | 路径 | 输出文件夹的路径。 | 无 | 是 |
| `--tasks` | 整数 | 任务总数。输入文件将被平均分配给这些任务。 | 64 | 否 |
| `--workers` | 整数 | 用于处理任务的 CPU 核心数。 | 32 | 否 |
| `--workers_per_gpu` | 整数 | 在依存句法分析阶段，共享一个 GPU 的 worker 数量。 | 4 | 否 |
| `--sample_rate` | 浮点数 | 数据采样率。 | 0.2 | 否 |
| `--ltp_model_path` | 路径 | LTP 中文语言处理模型的路径（用于分词、词性标注等）。 | 无 | 是 |
| `--tokenizer_path` | 路径 | 分词器 JSON 文件的路径（例如 Llama tokenizer）。 | 无 | 是 |
| `--limit` | 整数 | 限制每个 worker 处理的最大样本数，常用于快速测试。 | 无 | 否 |
| `--rerun` | 布尔值 | 是否重新运行。如果设为 `False`，则会从上次的检查点继续运行。 | `False` | 否 |

## 输出目录结构

```
output_path/
├── 1_gc_data/                         # 语法复杂度（GC）相关数据
│   ├── 1_dependency_parsing/          # 依存句法分析中间结果
│   ├── 2_part_of_speech_predicting/   # 词性预测中间结果
│   ├── 3_lexical_diversity/           # 词汇多样性指标
│   ├── 4_syntactic_complexity/        # 句法复杂性指标
│   ├── 5_combined_gc/                 # 整合后的GC结果
│   └── 6_normalized_gc/               # 逐维度Min-Max归一化结果
├── 2_sampling/                        # 采样结果
│   ├── 1_probability/                 # 每条数据的采样概率
│   └── 2_sample_result/               # 最终筛选结果（您所需文件）
└── logs/                              # 日志文件
    ├── gc/                            # 语法复杂度计算日志
    │   ├── dependency_parsing/        # 依存句法分析日志
    │   ├── part_of_speech_predicting/ # 词性预测日志
    │   └── gc_calculator/             # GC数据整合日志
    └── sampling/                      # 采样日志
        ├── probability_calculator/    # 归一化与概率计算日志
        └── sample_result/             # 依概率采样日志。
```

注意：
- 如果您只关注最终输出，可以直接查看 `output_path/2_sampling/2_sample_result` 目录下的所有 `.jsonl` 文件。
- `logs/sample_result/stats.json` 文件中的 `doc_len_tokens` 字段可查看筛选前后的token数量。

## GPU 使用说明

### 资源占用

只有依存句法分析阶段需要使用 GPU，其余部分均只依赖 CPU。为了避免 GPU 长时间被占用，我们将依存句法分析设为第一个步骤，并在运行结束后立即释放 GPU 资源。该阶段的 worker 数量由可见显卡数量（通过 `CUDA_VISIBLE_DEVICES` 指定）和每个显卡的最大进程数（通过 `--workers_per_gpu` 设置）共同决定。

您需要在环境变量中指定 `CUDA_VISIBLE_DEVICES`，代码会自动使用所有可见的 GPU。

参数 `--workers_per_gpu` 允许多个进程共享同一个 GPU，默认值为 1。我们没有对显存占用进行详细测试，因此您可能需要根据实际的显存大小和利用率进行调整。经测试，4 个进程共享一张 3090 显卡可以达到较高的利用率。

### 多卡数据并行

datatrove 框架会将多个文件划分为多个任务（task），再由不同的 worker 并行处理这些任务。我们让不同的 worker 占用不同的 GPU，以实现数据并行。

考虑到 `PipelineStep.run` 方法中的 rank 是任务 ID 而非 worker ID，我们在 `src/datatrove/executor/local.py` 的第 89 行将 worker 的 local rank 传入到 pipeline step 中，以便确定当前 worker 应该使用的显卡编号。

### 推理批次

由于对单个句子进行依存句法分析的显存占用较小，我们在 `src/datatrove/pipeline/cdf_gc/dependency_parser.py` 的 `ChineseDependencyParser` 类中设置了 `batch_size` 和 `max_length` 参数，用于控制批量推理的句子数量和句子长度上限。通常无需修改这些参数，但如果显存不足，可以适当减小它们。
