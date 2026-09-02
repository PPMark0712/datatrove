# CDF-GC (Generative Complexity)

文本生成复杂度评分方法，通过语法复杂度指标衡量文本质量，结合 CDF 采样实现数据筛选。目前支持中文（依存分析基于 LTP），词性标注同时支持中英文。

## 原理

### GC 指标

GC (Generative Complexity) 由 5 个分量组成，分别从词汇多样性和句法复杂度两个维度衡量文本：

| 分量 | 维度 | 计算方式 |
|------|------|----------|
| `pos_ent` | 词汇侧 | 词性标签的信息熵 |
| `con_ent` | 词汇侧 | 实词（名词/动词/形容词/副词）的分布熵 |
| `dep_ent` | 句法侧 | 依存关系标签的信息熵 |
| `avg_dep_height` | 句法侧 | 依存树平均高度 |
| `avg_dep_dis` | 句法侧 | 平均依存距离（词与其 head 的位置差绝对值） |

5 个分量经全局 min-max 归一化后，等权平均得到最终 GC 分数。

### CDF 采样

按 GC 分数排序后，采用混合采样策略：

1. **Hard 段**（高分尾部，默认占采样量的 40%）：直接保留 GC 最高的文档（prob=1）
2. **CDF 段**（其余部分，占 60%）：采样概率正比于 CDF——GC 越高的文档被选中概率越大

这样既保证高质量文档充分覆盖，又让中等质量文档有概率入选。

## 流水线步骤

完整流水线包含以下阶段，由 `scripts/data_select/cdf_gc.py` 串联执行：

1. **依存句法分析**（GPU）：`DocumentDependencyParser` — 中文 LTP 模型，输出 words / dep_labels / parents
2. **词性标注**（CPU）：`DocumentPartOfSpeechPredictor` — 中文 jieba / 英文 nltk，同时通过 `TokensCounter` 统计 token 数
3. **指标计算**（CPU）：
   - `LexicalDiversityCalculator` → pos_ent, con_ent
   - `SyntacticComplexityCalculator` → dep_ent, avg_dep_height, avg_dep_dis
   - `GcCombiner` → 合并两侧指标
4. **归一化 + 采样概率**（CPU，单 worker）：
   - `GcNormalizer` → 全局 min-max 归一化
   - `ProbabilityCalculator` → 计算每篇文档的采样概率
5. **执行采样**（CPU）：`ProbabilitySampler` → 按概率抛硬币，输出筛选后的 JSONL

## 前置准备

- **LTP 模型**：从 Hugging Face 下载 [LTP/small](https://huggingface.co/LTP/small)
- **Tokenizer**：准备目标 LLM 的 `tokenizer.json`（用于 token 计数）

## 使用

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/data_select/cdf_gc.py \
    --input_path /path/to/jsonl_folder \
    --glob_pattern "*.jsonl" \
    --output_path /path/to/output_folder \
    --tasks 64 --workers 32 \
    --workers_per_gpu 4 \
    --sample_rate 0.2 \
    --ltp_model_path /path/to/LTP/small \
    --tokenizer_path /path/to/tokenizer.json
```

## 参数说明

| 参数 | 类型 | 说明 | 默认值 | 必填 |
|:---|:---|:---|:---|:---|
| `--input_path` | 路径 | 输入 JSONL 文件夹 | — | 是 |
| `--glob_pattern` | 通配符 | 匹配输入文件的模式 | 全部文件 | 否 |
| `--output_path` | 路径 | 输出文件夹 | — | 是 |
| `--tasks` | 整数 | 任务总数，输入文件平均分配 | 32 | 否 |
| `--workers` | 整数 | CPU worker 数 | 32 | 否 |
| `--workers_per_gpu` | 整数 | 依存分析阶段每 GPU 的 worker 数 | 1 | 否 |
| `--sample_rate` | 浮点数 | 采样率 | — | 是 |
| `--rate_for_hard_sample` | 浮点数 | 采样量中 hard sample 占比 | 0.4 | 否 |
| `--ltp_model_path` | 路径 | LTP 模型路径 | — | 否 |
| `--tokenizer_path` | 路径 | tokenizer.json 路径 | — | 是 |
| `--limit` | 整数 | 每个 worker 最大处理样本数（测试用） | -1 | 否 |
| `--rerun` | 标志 | 忽略检查点，全部重跑 | False | 否 |

## 输出目录结构

```
output_path/
├── 1_gc_data/
│   ├── 1_dependency_parsing/          # 依存句法分析中间结果
│   ├── 2_part_of_speech_predicting/   # 词性预测中间结果
│   ├── 3_lexical_diversity/           # 词汇多样性指标
│   ├── 4_syntactic_complexity/        # 句法复杂性指标
│   ├── 5_combined_gc/                 # 合并后的 GC 结果
│   └── 6_normalized_gc/              # Min-Max 归一化结果
├── 2_sampling/
│   ├── 1_probability/                # 每条数据的采样概率
│   └── 2_sample_result/              # 最终筛选结果
└── logs/
```

最终结果在 `output_path/2_sampling/2_sample_result/` 下的 `.jsonl` 文件中。

## GPU 使用说明

- 仅依存句法分析阶段需要 GPU，其余步骤只用 CPU。依存分析被安排为第一步，完成后即释放 GPU。
- 必须设置 `CUDA_VISIBLE_DEVICES`，代码会自动使用所有可见 GPU。
- `--workers_per_gpu` 允许多进程共享同一 GPU（经测试 4 进程共享一张 3090 可达较高利用率）。

## 扩展至其他语言

如需支持新语言，需要扩展以下两个文件：

- `pipeline/cdf_gc/dependency_parser.py` — 新增对应语言的依存分析器
- `pipeline/cdf_gc/part_of_speech_predictor.py` — 新增对应语言的词性标注器

## 目录结构

```
pipeline/cdf_gc/
├── gc_calculator.py             # GC 各阶段算子
├── part_of_speech_predictor.py  # 词性标注（中/英）
├── dependency_parser.py         # 依存分析（中文 LTP）
└── cdf_sampler.py               # CDF 概率采样器
```
