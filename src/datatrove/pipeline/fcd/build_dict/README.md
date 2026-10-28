# FCD 本地资源与可复现构建

默认资源目录为本文件旁的 `data/`。

| 文件 | 来源 / 用途 | 当前数量 |
|---|---|---:|
| `oxford_5000.csv` | 用户提供的 Oxford 5000 / CEFR 原始 CSV | 原样复制 |
| `word_freq.txt` | 用户提供的词频表；不重新抓取 Google Ngram | 15,493,823 行 |
| `basic_words.txt` | CSV 中 CEFR 为 a1/a2、纯字母词，小写后去重排序 | 1,679 词 |
| `dis_to_basic.txt` | WordNet 3.0 名词图到基础词名词义项的最短距离 | 82,115 义项 |
| `manifest.json` | 输入、输出、脚本的 SHA-256，资源大小及 NLTK/WordNet 版本 | 构建清单 |

CEFR 构建逻辑沿用 `get_basic_words.py`。它原先的下载源是第三方仓库
`winterdl/oxford-5000-vocabulary-audio-definition` 的 `data/oxford_5000.csv`。
本地流程只使用传入文件，不验证该 CSV 是否与在线最新版本一致；词频表的上游版本也不从内容推断，按清单中的 SHA-256 标识本次输入。

## 使用本地输入复现

在仓库根目录、激活 datatrove 环境后执行：

```powershell
python src/datatrove/pipeline/fcd/build_dict/prepare_local.py --cefr-csv "C:/Users/mark/Downloads/oxford_5000.csv" --word-freq "C:/Users/mark/Desktop/word_freq.txt" --nltk-data webui/.cache/nltk_data
```

重新构建时也可将两个输入参数指向 `data/oxford_5000.csv`、`data/word_freq.txt`。
脚本不会修改源文件；先复制到 data，再逐行校验词频为正整数、所有基础词都有对应词频。
WordNet 必须事先安装到本地，缺少语料时直接失败，不隐式下载。
本次使用 NLTK 3.10.4、WordNet 3.0；使用相同输入、脚本及语料版本可复现清单中的输出哈希。
生成文本统一为 UTF-8 / LF。

距离计算沿用仓库算法：将基础词的所有名词义项作为起点，沿 hypernyms、instance_hypernyms、hyponyms、instance_hyponyms 做多源 BFS；起点距离为 0，不可达节点保留哨兵值 100。起点和最终输出均稳定排序。

大型 CSV 和词频表保留本地并被 Git 忽略；基础词表、距离表与 manifest 可纳入版本控制。
`.gitignore` 也会忽略 data 中其他大型原始资源。

## 运行 FCD 验收

```powershell
python webui/verify_fcd.py
python -m pytest tests/test_webui.py tests/pipeline/test_fcd_resources.py -q
```

验收脚本默认使用 `webui/test_data/test.jsonl` 的全部 20 条记录，禁用清洗，以单进程运行真实 FCD，再运行 FCD + Hard 50% 采样。
检查分数和记录对齐、两次打分一致、Top 半数和原顺序、UI 统计/预览元数据、ZIP 完整性、100 桶 PDF/CDF 数值。
任务、报告和图片保存在 `webui/.cache/fcd-acceptance/`，不覆盖现有演示任务。
该样本含非英文与噪声记录；本验收不代表语言过滤或文本质量评估通过，FCD 分数高也不等于文本质量高。

218 MB 词频表包含约 1,549 万词条，FCD 会将其载入 Python 字典，每个工作进程各持一份。
本机 16 GB 内存验证时使用 `workers=1`；不要直接套用旧文档的 32 进程示例。

## 历史全量构建脚本

`build_dict.sh` 串联下载 Oxford CSV、下载 Google Ngram 24 个分片、统计词频、计算距离。
这些脚本保留供参考；其中 `calc_word_freq.py` 的相对目录/输出目录与下载脚本不一致，固定 24 进程也需要先评估资源。
本次可复现入口是 `prepare_local.py`，无需执行历史下载流程。
