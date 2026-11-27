# 同学本地运行指南

这个仓库包含完整 Python 源代码，**不用打开 Notebook 也能运行**。

## 1. 下载项目

```bash
git clone https://github.com/Jerryre2/spotify-popularity-analysis.git
cd spotify-popularity-analysis
```

## 2. 安装环境

先安装 Python 3.13。macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 3. 准备数据

从 [Kaggle 数据页面](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks)下载数据，或向组员取得本项目原始 `archive.zip`。

将它放在 `data/archive.zip`，不用解压。完整原始数据没有放进 Git 仓库，因此仅 clone 后还需要这一步。

```bash
python run.py check
```

如果提示数据版本不同，向组员索取原始压缩包；不要删掉校验直接套用报告结论。

## 4. 一条命令运行完整项目

```bash
python run.py all --jobs 2
```

程序依次执行：数据检查 → 模型训练和测试 → 五次划分的稳定性分析。结果保存到 `runs/`。全量训练需要一定时间和内存，不需要显卡；电脑资源紧张时改用 `--jobs 1`。

也可以分步运行：

```bash
python run.py audit
python run.py train --jobs 2
python run.py stability --jobs 2
```

稳定性分析必须在训练完成后运行。数据在其他目录时，给每条命令加 `--data "你的路径/archive.zip"`；自定义输出目录时，各阶段使用相同的 `--output`。

## 代码在哪里

- `run.py`：统一启动入口。
- `scripts/01_audit.py`：数据检查与描述性分析。
- `scripts/02_train.py`：数据分组、划分、防泄漏、模型比较、特征重要性、模型保存。
- `scripts/03_stability.py`：额外四次划分及五次结果汇总。
- 三个 `.ipynb` 文件：相同分析的 Notebook 版本，附原始运行结果。
- `reports/`：英文报告。

`run.py` 默认将新结果写到 `runs/`，不会覆盖仓库顶层已发表的实验结果。重复使用同一个输出目录会替换其中同名结果文件。
