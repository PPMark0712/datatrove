# FCD Data Files

FCD 计算需要以下数据文件，放置在 `build_dict/data/` 目录下：

| 文件 | 说明 |
|------|------|
| `basic_words.txt` | Oxford 5000 中 CEFR A1/A2 级别的基础词汇表 |
| `dis_to_basic.txt` | WordNet synset 到基础词的最短距离 |
| `word_freq.txt` | 基于 Google Ngram 的词频表 |

这些文件未纳入 git 版本管理（已在 `.gitignore` 中排除），需要通过以下方式获取。

## 方式一：从 Release 下载

从仓库的 GitHub Release 页面下载 `fcd_data.tar.gz`，解压到 `build_dict/data/` 目录：

```bash
cd src/datatrove/pipeline/fcd/build_dict
tar -xzf fcd_data.tar.gz -C data/
```

## 方式二：自行构建

按顺序执行 `build_dict/` 下的脚本（需要网络访问）：

```bash
cd src/datatrove/pipeline/fcd/build_dict
bash build_dict.sh
```

步骤说明：
1. `get_basic_words.py` — 从 GitHub 下载 Oxford 5000 词表，提取 A1/A2 基础词 → `data/basic_words.txt`
2. `download_google_ngram.py` — 下载 Google Ngram 数据
3. `calc_word_freq.py` — 计算词频 → `data/word_freq.txt`
4. `calc_dis_to_basic.py` — 计算 WordNet 距离 → `data/dis_to_basic.txt`
