# 数据清洗 (Data Cleaning)

完整的文本数据清洗流水线，包含语言过滤、质量过滤和 MinHash 模糊去重三个阶段。每个阶段的输出作为下一个阶段的输入，被过滤的文档会保存到 `removed/` 子目录供审计。

## 整体流程

```
原始数据
  │
  ▼
[阶段 1] 语言过滤
  ├── LanguageFilter（fastText 语言检测）
  └── 置信度阈值过滤（≥ 0.65）
  │
  ▼
[阶段 2] 质量过滤（按语言独立执行）
  ├── GopherRepetitionFilter（文档内重复检测）
  ├── GopherQualityFilter（词数/符号比/格式规则）
  └── FineWebQualityFilter（行末标点/短行/去重字符比）
  │
  ▼
[阶段 3] MinHash 去重（按语言独立执行）
  ├── Stage 1: 计算 MinHash 签名
  ├── Stage 2: 桶内匹配重复文档对
  ├── Stage 3: Union-Find 聚类
  └── Stage 4: 过滤重复文档 + Token 计数 + PII 脱敏
  │
  ▼
最终输出
```

## 阶段 1：语言过滤

使用 fastText（`ft176` 后端）检测文档语言，仅保留目标语言的文档。

| 参数 | 说明 |
|------|------|
| 目标语言 | 默认 `zh`，可指定多个（如 `zh,en`） |
| 置信度阈值 | 0.65，低于此值的文档被移除 |

输出按语言分子目录存储。

## 阶段 2：质量过滤

三个过滤器串行执行，任一不通过即被移除：

### GopherRepetitionFilter

检测文档内部的重复内容（基于 Gopher 论文 arXiv:2112.11446）：

| 指标 | 默认阈值 | 说明 |
|------|----------|------|
| 重复行比例 | 0.3 | 完全相同的行占比 |
| 重复段落比例 | 0.3 | 完全相同的段落占比 |
| 重复行字符比例 | 0.2 | 重复行的字符占总字符比 |
| 重复段落字符比例 | 0.2 | 重复段落的字符占总字符比 |
| Top n-gram 字符占比 | 2-gram: 0.2, 3-gram: 0.18, 4-gram: 0.16 | 最高频 n-gram 的字符占比 |
| 重复 n-gram 字符占比 | 5-gram: 0.25 ~ 10-gram: 0.17 | 出现多次的 n-gram 字符占比 |

### GopherQualityFilter

基础文档质量规则：

| 指标 | 默认阈值 | 说明 |
|------|----------|------|
| 文档词数 | 50 ~ 100,000 | 过短或过长的文档被移除 |
| 符号词比例 | ≤ 0.1 | `#` 和 `...` 占词数的比例 |
| 项目符号行比例 | ≤ 0.9 | 以列表符号开头的行占比 |
| 省略号结尾行比例 | ≤ 0.3 | 以 `...` 结尾的行占比 |

中文数据处理时，停用词检测、平均词长等英文相关规则会被禁用。

### FineWebQualityFilter

基于 FineWeb 的额外质量规则：

| 指标 | 默认阈值 | 说明 |
|------|----------|------|
| 行末标点比例 | ≥ 0.12 | 以终止标点结尾的行占比 |
| 短行比例 | ≤ 0.67 | 长度 ≤30 字符的行占比 |
| 字符去重比例 | ≤ 0.01 | 重复行字符占总字符比 |
| 换行/词比 | ≤ 0.3 | 换行符数/词数，检测列表类页面 |

## 阶段 3：MinHash 去重

使用 MinHash LSH（Locality-Sensitive Hashing）进行模糊去重，相似度约 0.72 以上的文档会被识别为重复。

### 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 哈希函数 | SHA1, 64-bit | 精度与碰撞率的平衡 |
| 桶数 | 14 | LSH 桶数量 |
| 每桶哈希数 | 8 | 每个桶内的哈希数量 |
| n-gram | 5 | 文本分词粒度 |
| 隐式相似度阈值 | ~0.72 | `(1/14)^(1/8)` |

### 四个子步骤

1. **签名计算**：文本预处理（小写化、去标点、空白规范化）→ 生成 5-gram shingles → SHA1 哈希 → MinHash 签名
2. **桶匹配**：在每个桶内查找签名完全相同的文档对，14 个桶独立并行处理
3. **聚类**：Union-Find 算法将重复对聚合为簇，每簇仅保留一篇文档
4. **过滤输出**：根据聚类结果移除重复文档，同时进行：
   - Token 计数（使用目标 LLM 的 tokenizer）
   - PII 脱敏（邮件地址、公网 IP 替换为虚拟值）

## 使用

```bash
python scripts/data_clean/data_cleaning.py \
    --input_path /path/to/data \
    --output_path /path/to/output \
    --languages zh en \
    --tasks 64 --workers 32
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_path` | 输入数据目录 | — |
| `--output_path` | 输出目录 | — |
| `--languages` | 目标语言列表 | `zh` |
| `--tasks` | 任务数 | 32 |
| `--workers` | 并行 worker 数 | 32 |

## 输出目录结构

```
output_path/
├── 1_language_filter/
│   └── output/{language}/          # 按语言分目录
├── 2_quality_filter/
│   └── {language}/
│       ├── output/                 # 通过质量过滤的文档
│       └── removed/               # 被过滤的文档（附 filter_reason）
├── 3_minhash_deduplication/
│   └── {language}/
│       ├── 1_signatures/          # MinHash 签名
│       ├── 2_buckets/             # 桶匹配结果
│       ├── 3_clusters/            # 聚类结果
│       └── 4_result/
│           └── output/            # 最终去重结果
└── logs/
```
