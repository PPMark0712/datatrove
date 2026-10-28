# FCD 数据与 Release 资源包

FCD 默认读取本目录下的 `data/`。当前资源符合加载要求：UTF-8，词频支持空格或 Tab 分隔，三个运行词典均通过实际打分验收。

| 文件 | 用途 | 当前数量 |
|---|---|---:|
| `basic_words.txt` | CEFR A1/A2、纯字母词，小写去重排序 | 1,679 词 |
| `dis_to_basic.txt` | WordNet 3.0 名词图到基础词义项的最短距离 | 82,115 义项 |
| `word_freq.txt` | 用户提供的词频表 | 15,493,823 行 |
| `oxford_5000.csv` | 重建基础词表所用的原始 CSV；运行时不读取 | 1,138,758 字节 |

整个 data 目录、打包后的 ZIP 均不进入 Git。没有 manifest 文件，也不会在构建时生成 manifest。
NLTK 运行语料仍需单独安装，资源 ZIP 不包含 NLTK、CUDA 或 Python 依赖。

## 用户：下载即可使用

从本仓库对应版本的 GitHub Release 的 Assets 区域下载 **`fcd_data.zip`**。
不要下载 GitHub 自动生成的 Source code ZIP，因为 Git 源码不包含 data。
在仓库根目录执行以下整段命令：

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


ZIP 内有 `data/` 这一层目录，解压后四个文件直接位于正确位置，无需重建词典。
下载地址指向 GitHub 标记为 Latest 的版本 Release；首次发布资源和版本前会返回 404。

也可下载指定版本的附件：

```bash
gh release download VERSION_TAG --repo PPMark0712/datatrove --pattern fcd_data.zip --dir dist
```

## 维护者：打包与首次上传

已有本地 data 时，在仓库根目录执行：

```powershell
python src/datatrove/pipeline/fcd/build_dict/package_data.py
```

生成 `dist/fcd_data.zip`，只包含上述四个文件。打包过程流式压缩、检查文件缺失/空文件和 ZIP CRC；不会载入完整词频字典，也不会生成 manifest。

CI 无法访问本地忽略的 data，所以需要**一次性上传**本地生成的 ZIP。可以在 GitHub 页面创建一个名为 `fcd-data` 的资源 Release，并将 ZIP 拖入 Assets；或者在已登录 GitHub CLI 后执行：

```powershell
gh release create fcd-data dist/fcd_data.zip --repo PPMark0712/datatrove --title "FCD data" --notes "Prebuilt FCD dictionaries" --latest=false
```

这一步由维护者明确执行；仅添加 workflow 不会把电脑上的文件自动上传。
若资源 Release 已存在，可使用 `gh release upload fcd-data dist/fcd_data.zip --repo PPMark0712/datatrove` 上传附件。

## 维护者：每个版本自动附加资源

`.github/workflows/fcd-data-release.yml` 支持推送 `v*` 标签、GitHub Release published、Actions 手动运行，以及现有 PyPI 发布流程的可复用调用：

1. 从本仓库的 `fcd-data` Release 下载现有 `fcd_data.zip`；不运行词典构建。
2. 校验 ZIP 结构、四个文件和 CRC。
3. 如果目标 Git tag 尚无 Release，先创建草稿；上传原样 ZIP 后再发布，可用于不可变 Release。若已有 Release，仅补充缺失附件；目标已有同名附件时保留它，不覆盖已发布资源。

相同目标标签的任务串行运行，避免 tag push 和 published 事件并发重复上传。手动创建的新版本应选择 Latest，才能通过 `/releases/latest/download/fcd_data.zip` 下载。

资源源 Release 可通过仓库 Actions 变量 `FCD_DATA_SOURCE_TAG` 指定其他标签。也可在 Actions 中手动运行此 workflow，填写已存在的目标 Release tag 和资源源 tag。
首次发布 `fcd-data` 资源 Release 时，该 workflow 会跳过自身。

将 workflow 和打包校验脚本合并到默认分支并推送后，再发布包含这些脚本的新版本。直接推送 `v*` 标签会自动创建带附件的 Release；现有 `pypi-release.yml` 在 PyPI 发布成功后显式调用本流程，因为使用 GITHUB_TOKEN 创建的标签不会再次触发其他 workflow。资源源附件缺失时流程会失败，不创建空的新 Release。
若启用了不可变 Release，应先创建草稿，在 Actions 手动运行此 workflow 给草稿添加资源，再发布草稿。不能依赖发布后追加附件；已包含附件的 published 事件会跳过上传。

## 可选：维护者从本地原始输入重建

普通用户不需要这一步。维护者需要更新词典时，可执行：

```powershell
python src/datatrove/pipeline/fcd/build_dict/prepare_local.py --cefr-csv "C:/Users/mark/Downloads/oxford_5000.csv" --word-freq "C:/Users/mark/Desktop/word_freq.txt" --nltk-data webui/.cache/nltk_data
```

脚本复制原始输入、校验正整数词频和基础词覆盖，生成两个派生词典，只在控制台打印数量与版本。
本次使用 NLTK 3.10.4、WordNet 3.0。基础词来自 `get_basic_words.py` 沿用的 A1/A2 筛选逻辑；其原始在线来源为第三方仓库 `winterdl/oxford-5000-vocabulary-audio-definition` 的 `data/oxford_5000.csv`。
距离沿四类上下位关系做多源 BFS；起点距离为 0，不可达哨兵为 100，结果稳定排序且使用 UTF-8 / LF。

历史 `build_dict.sh` / Google Ngram 下载脚本保留供参考，其中相对目录和固定 24 进程需要另行修正/评估；本地输入重建无需下载 Ngram。

## 验收与资源开销

```powershell
python webui/verify_fcd.py
python -m pytest tests/test_webui.py tests/pipeline/test_fcd_resources.py tests/pipeline/test_fcd_release.py -q
```

真实验收使用测试数据全部 20 条记录、禁用清洗，单进程 FCD 后再 Hard 50% 采样，得到 10 条输出。
检查分数对齐与重复运行一致性、UI 统计/预览元数据、ZIP 内容、100 桶 PDF/CDF。
样本包含非英文和噪声，FCD 高分不等于质量高。
词频表约 218 MB、1,549 万词条；每个 FCD 进程会各自加载整张表，本机 16 GB 内存验证采用 workers=1。
