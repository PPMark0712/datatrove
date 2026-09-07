# 采样策略 (Samplers)

通用的数据采样框架，支持基于预计算分数的多种采样策略。采样可按文档数量或 token 数量控制。

## 三种采样策略

### CDF 采样（概率递增）

按分数排序后，采样概率与 CDF（累积分布）成正比——分数越高的文档被采中概率越大，但低分文档仍有机会入选。

实际使用的是 **CDF Balanced** 混合策略，将文档分为两段：

1. **Hard 段**（高分尾部）：直接随机抽取，占总采样量的 `rate_for_hard_sample`（默认 40%）
2. **CDF 段**（其余部分）：概率按 CDF 递增分配，占总采样量的 60%

```
采样概率
    │                    ┌────── Hard 段（随机均匀）
    │                ····│·····
    │           ····     │
    │       ···          │
    │    ··              │
    │  ·                 │
    │·                   │
    └────────────────────┴───── 文档（按分数排序）
   低分                      高分
```

这样既保证高质量文档的充分覆盖，又让中等质量文档有概率入选，避免硬截断带来的分布断裂。

**CDF-GC 中的 CDF 采样**

CDF-GC 流水线内置了一套紧耦合的 CDF 采样器（`pipeline/cdf_gc/cdf_sampler.py`），与通用版本的核心区别：

- 跨 rank 全局排序：`ProbabilityCalculator` 读取所有 rank 的 GC 数据，全局计算采样概率
- 基于 token 量分配概率：`prob = min(1, r × 累积token占比)`，而非按文档序号
- 高分段（Hard）直接设 `prob=1.0`，其余按 CDF 概率
- `ProbabilitySampler` 对每篇文档独立抛硬币决定保留

### Hard 采样（Top-K）

确定性地取分数最高（或最低）的 top-K 文档，无随机性。

- `sample_rate=0.2` 即保留排名前 20% 的文档
- `higher_is_better=True`（默认）取高分，`False` 取低分
- 简单直接，适合需要严格筛选高质量数据的场景

### Random 采样（均匀随机）

均匀随机采样，与分数无关。

- 需要先运行 `DocumentCounter` 统计文档总数
- 再由 `RandomSampler` 按 `sample_rate` 随机抽取
- 适合作为基线对照或无分数场景

## 采样单位

所有策略都支持两种采样单位：

| 单位 | 说明 |
|------|------|
| `doc` | 按文档数量。`sample_rate=0.2` → 保留 20% 的文档 |
| `token` | 按 token 总量。`sample_rate=0.2` → 保留约 20% 的 token（以完整文档为单位，不拆分） |

使用 token 模式时需提供 `--token_count_folder`（预计算的 token 数目录）。

## 使用

### CDF 采样

```bash
python scripts/sample/cdf_sample.py \
    --input_path /path/to/data \
    --score_path /path/to/scores \
    --output_path /path/to/output \
    --sample_rate 0.2 \
    --unit doc \
    --tasks 32 --workers 32
```

### Hard 采样

```bash
python scripts/sample/hard_sample.py \
    --input_path /path/to/data \
    --score_path /path/to/scores \
    --output_path /path/to/output \
    --sample_rate 0.2 \
    --unit doc \
    --tasks 32 --workers 32
```

`--lower_is_better` 可反转排序方向。

### Random 采样

```bash
python scripts/sample/random_sample.py \
    --input_path /path/to/data \
    --output_path /path/to/output \
    --sample_rate 0.2 \
    --unit doc \
    --tasks 32 --workers 32
```

不需要 `--score_path`（与分数无关）。

## 参数说明

| 参数 | 说明 | 适用策略 |
|------|------|----------|
| `--sample_rate` | 采样比例 (0,1] | 全部 |
| `--unit` | 采样单位：`doc` 或 `token` | 全部 |
| `--score_path` | 预计算分数目录 | CDF, Hard |
| `--token_count_path` | token 数目录（token 模式必填） | 全部 |
| `--seed` | 随机种子（默认 42） | CDF, Random |
| `--lower_is_better` | 分数越低越好（默认 False，即分数越高越好） | Hard |
| `--hard_sample_ratio` | Hard 段占比，默认 0.4（CDF Balanced） | CDF |

## 架构设计

采用两层抽象：

- **IndexSampler 层**：纯索引级采样逻辑，不接触文档内容。`IndexCdfSampler`、`IndexHardSampler`、`IndexRandomSampler` 可自由组合
- **BaseSampler 层**：Pipeline 集成层，读取分数文件 → 排序 → 调用 IndexSampler → 过滤文档流 → yield 命中文档

```
pipeline/samplers/
├── base.py      # BaseIndexSampler + BaseSampler 抽象基类
├── cdf.py       # IndexCdfSampler / IndexCdfBalancedSampler / CdfSampler
├── hard.py      # IndexHardSampler / HardSampler
└── random.py    # IndexRandomSampler / RandomSampler / DocumentCounter
```
