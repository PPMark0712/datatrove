# WebUI 数据筛选平台

基于 Gradio 的 Web 界面，提供数据集管理 → 配置流水线 → 查看分布 → 下载结果的完整流程。支持本地和远程服务器部署。

## 安装

```bash
# 本地（仅 FCD，无 GPU）
pip install -e ".[webui]"

# 服务器（全功能，含 CDF-GC）
pip install -e ".[webui,gpu]"
```

## 启动

```bash
cd webui
bash run.sh
```

浏览器访问 `http://localhost:7860`（或 `http://服务器IP:7860`）。

### 配置

编辑 `webui/run.sh`：

```bash
# 账号密码（格式 "用户名:密码"，多个用逗号分隔）
export WEBUI_AUTH="admin:datatrove2024"

# 服务端口
export WEBUI_PORT=7860

# GPU 设备（CDF-GC 需要，按需打开）
# export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 远程访问

- 直接访问：确保服务器防火墙放行端口
- SSH 隧道：`ssh -L 7860:localhost:7860 服务器`，然后访问 `localhost:7860`

## 界面布局

采用左侧边栏 + 右侧主面板的双栏布局，顶部固定标题栏，左右独立滚动。

```
┌─────────────────────────────────────────────────┐
│  Datatrove  数据筛选与质量评估平台                │
├────────────┬────────────────────────────────────┤
│ 📂 数据集   │  任务信息 / 数据预览                 │
│  [卡片列表]  │                                    │
│  [重命名]   │  流水线配置                          │
│  [上传区域]  │   🧹 数据清洗                       │
│  [服务器导入] │   📊 FCD 打分                      │
│─────────── │   🧠 CDF-GC 打分                    │
│ 📋 任务     │   🎲 采样                           │
│  [卡片列表]  │   [Workers] [开始处理] [续传]        │
│  [新建/刷新] │                                    │
│             │  执行状态 / 运行日志                  │
│             │                                    │
│             │  结果与分析                          │
│             │   [绘制分布] [下载] [保存为数据集]     │
└────────────┴────────────────────────────────────┘
```

## 功能介绍

### 左侧边栏 — 数据集管理

| 功能 | 说明 |
|------|------|
| 数据集列表 | 显示 workspace 目录下所有数据集（名称 + 文件数），点击选中 |
| 重命名 | 选中数据集后在文本框输入新名称，点击 ✏️ 确认 |
| 上传 | 拖拽 `.jsonl` / `.zip` 文件到上传区域，或点击选择文件；上传后弹出命名窗口，默认命名"数据集N" |
| 服务器路径导入 | 填写服务器上的目录路径，自动 glob `*.jsonl` 并导入 |
| 保存结果为数据集 | 任务完成后，可在右侧结果区点击"保存为数据集"，将输出写回 workspace |

数据集存储在 `webui/workspace/<数据集名>/` 目录下，每个子目录包含若干 `.jsonl` 文件。

### 左侧边栏 — 任务管理

| 功能 | 说明 |
|------|------|
| 任务列表 | 显示所有历史任务及状态（⚪ created / 🟡 running / 🟢 completed / 🔴 error / ⚫ stopped） |
| 新建任务 | 先选中数据集，再点击"+ 新建任务"，系统自动分配递增编号（任务 1、任务 2…） |
| 切换任务 | 点击任务卡片切换；若当前任务正在运行，弹出确认窗口询问是否终止 |
| 并发限制 | 同一时刻只允许一个任务运行，新建/启动/续传时均会检查 |
| 断点续传 | 选中已有任务后点击"↻ 续传"，框架自动跳过已完成步骤 |

任务数据存储在 `webui/tasks/task_N/` 目录下，包含 `input/`、`split/`、`work/`、`output/`、`config.json`、`status.json`、`task.log`。

### 右侧主面板 — 流水线配置

可选择以下处理步骤（按勾选顺序串联执行）：

**数据清洗**
- 语言过滤 + 质量过滤 + MinHash 去重
- 参数：目标语言（逗号分隔，如 `zh,en`）

**FCD 打分**（英文，纯 CPU）
- 计算每篇文档的词汇难度分数
- 可调参数：`freq_scaling_factor`、`w_f`、`noun_weight`

**CDF-GC 打分 + 采样**（中文，需 GPU）
- 完整的 GC 计算 + CDF 采样流水线
- 可调参数：采样率、Tokenizer 路径、LTP 模型路径、GPU 设备、每 GPU workers 数

**采样**（基于 FCD 分数，需先启用 FCD）
- 三种策略可选：CDF / Hard / Random
- 可调参数：采样率、采样单位（doc/token）

**执行控制**
- 可设置并行 workers 数（1-64）
- 点击「开始处理」异步执行，不阻塞页面
- 点击「刷新状态」查看进度和日志

### 右侧主面板 — 结果与分析

| 功能 | 说明 |
|------|------|
| 分数分布图 | 选择分数类型（FCD / GC）绘制 PDF 和 CDF 图，含 n、mean、std、median 统计 |
| 下载结果 | 处理完成后获取 zip 压缩包（输出按每文件 ≤5000 行 merge） |
| 保存为数据集 | 将任务输出保存到 workspace，可用于后续任务的输入 |

## 处理流程

```
选中数据集 → 新建任务（复制数据到 task 目录）
    │
    ▼
Split（每文件 ≤5000 行）
    │
    ▼
[可选] 数据清洗（语言过滤 → 质量过滤 → 去重）
    │
    ▼
[可选] FCD 打分 → [可选] 采样（CDF/Hard/Random）
    │
    ▼
[可选] CDF-GC 打分 + CDF 采样
    │
    ▼
Merge 输出（每文件 ≤5000 行）
    │
    ▼
打包 zip 供下载 / 保存为新数据集
```

## 代码结构

```
webui/
├── app.py            # Gradio 界面布局与事件绑定（入口）
├── handlers.py       # 所有前端事件的处理函数
├── task_manager.py   # 任务生命周期、pipeline 执行、workspace 管理
├── styles.py         # 自定义 CSS
├── run.sh            # 启动脚本（环境变量配置）
├── workspace/        # 数据集存储（每个子目录 = 一个数据集）
└── tasks/            # 任务存储（每个子目录 = 一个任务）
    └── task_N/
        ├── input/        # 原始输入（从 workspace 复制）
        ├── split/        # split 后的分片
        ├── work/         # 各步骤中间结果
        ├── output/       # 最终输出
        ├── config.json   # 流水线配置（用于断点续传）
        ├── status.json   # 当前状态
        └── task.log      # 运行日志
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `app.py` | Gradio Blocks 布局定义、组件创建、事件绑定（`.click` / `.input` / `.change`）；入口 `__main__` 解析认证并启动服务 |
| `handlers.py` | 每个 UI 事件对应一个 handler 函数；负责调用 `task_manager` 并返回 Gradio 组件更新；包含数据集上传流程、任务切换守卫、pipeline 启动、结果保存等 |
| `task_manager.py` | `Task` 类封装任务目录结构和状态管理；pipeline 各步骤（split → clean → fcd → gc → sample → merge）的脚本调用；workspace 的 stage/confirm/cancel/rename 操作；`run_pipeline_async` 启动后台线程执行 |
| `styles.py` | 导出 `CUSTOM_CSS` 字符串，包含布局（sidebar + main-panel）、组件样式（badge、log、banner）、隐藏 Radio 默认标签等 |

### 关键设计

- **事件循环防护**：`task_radio` 使用 `.input()` 而非 `.change()` 绑定事件，避免编程式更新触发无限循环
- **运行互斥**：`handle_start` / `handle_resume` / `handle_create_task` 均调用 `_find_running_task()` 检查是否有任务在运行
- **切换守卫**：运行中切换任务时，`handle_switch_task` 弹出确认窗口（`switch_dialog`），用 `pending_switch_task` State 暂存目标任务 ID
- **上传流程**：文件先 stage 到 `.staging_*` 临时目录，用户确认命名后 rename 为正式目录；支持 ZIP 自动解压提取 `.jsonl`
- **后台执行**：`run_pipeline_async` 在 daemon 线程中运行，通过 `status.json` 和 `task.log` 与前端通信

## 数据溯源

输出数据中的 `id` 字段格式为 `原始文件路径/行号`，可追溯到原始输入文件和行。

## 环境要求

| 功能 | CPU | GPU |
|------|-----|-----|
| FCD 打分 | ✅ | — |
| CDF-GC | — | ✅（LTP 依存分析） |
| 采样 | ✅ | — |
| 数据清洗 | ✅ | — |
