# FCD (Frequency-Concept Difficulty)

英文文本词汇难度评分方法。对每篇文档中的词汇从**词频难度**和**概念难度**两个维度评分，综合得到文档级难度分数。

## 原理

### 词频难度

基于 Google Ngram 词频，通过 sigmoid 函数映射到 [0,1]：

```
freq_difficulty = 1 - sigmoid(freq_scaling_factor * (log_freq - log_freq_center))
```

其中 `log_freq_center` 默认取基础词表（Oxford 5000 A1/A2）词频的第 10 分位数。

### 概念难度（仅名词）

基于 WordNet 上义词层次结构，计算每个名词 synset 到基础词集合的最短距离，再通过对数归一化：

```
concept_difficulty = log(distance + 1) / log(max_distance + 1)
```

### 词级综合

- **名词**：`score = freq_difficulty^w_f * concept_difficulty^(1-w_f)`（默认 `w_f=0.5`）
- **非名词**：`score = freq_difficulty`

### 文档级聚合

- 短文档（≤20 词）：直接 power mean（α=1.5）
- 长文档：按 top 90% 分位和其余部分加权合并（top_weight=0.7）
- 最终分数：`noun_weight * noun_difficulty + (1 - noun_weight) * non_noun_difficulty`（默认 `noun_weight=0.7`）

## 数据文件

FCD 计算依赖三个数据文件，位于 `pipeline/fcd/build_dict/data/`（未纳入 git）：

| 文件 | 说明 |
|------|------|
| `basic_words.txt` | Oxford 5000 中 CEFR A1/A2 级别的基础词汇表 |
| `dis_to_basic.txt` | WordNet synset 到基础词的最短距离 |
| `word_freq.txt` | 基于 Google Ngram 的词频表 |

获取方式见 `src/datatrove/pipeline/fcd/build_dict/README.md`。

## 核心算子

`FcdCalculator` — 读取文档文本，输出每篇文档的 FCD 难度分数。

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_folder` | — | 输出目录 |
| `freq_scaling_factor` | 0.7 | sigmoid 缩放因子 |
| `w_f` | 0.5 | 词频与概念难度的几何加权指数 |
| `power_mean_alpha` | 1.5 | power mean 的 α 参数 |
| `agg_top_quantile` | 0.9 | 长文档聚合的 top 分位阈值 |
| `agg_top_weight` | 0.7 | top 分位的权重 |
| `noun_weight` | 0.7 | 名词难度在最终分数中的权重 |

## 使用

```bash
python scripts/calc_score/calc_fcd.py \
    --input_path /path/to/jsonl_folder \
    --output_path /path/to/output_folder \
    --tasks 32 --workers 32
```

输出目录下 `fcd_score/` 包含每个 task 对应的分数文件（JSON 数组）。

## 目录结构

```
pipeline/fcd/
├── fcd_calculator.py       # 核心计算器
└── build_dict/
    ├── data/               # 运行时数据（gitignore）
    ├── get_basic_words.py  # 提取 Oxford 5000 基础词
    ├── calc_word_freq.py   # 计算词频表
    ├── calc_dis_to_basic.py # 计算 WordNet 距离
    └── download_google_ngram.py # 下载 Google Ngram 数据
```
