# AdaDetectGPT 论文代码指南

## 1. 论文概述

AdaDetectGPT是一种具有统计保证的自适应LLM生成文本检测方法，发表于NeurIPS 2025会议。该方法解决了当前LLM检测方法缺乏统计保证和适应性差的问题，通过学习自适应见证函数来提高检测性能。

### 核心创新点
- **自适应检测**：从训练数据中学习最优的见证函数
- **统计保证**：提供严格的假阴性率(FNR)和真阴性率(TNR)保证
- **高性能**：在多种LLM检测任务中优于现有方法
- **灵活性**：支持带训练数据和不带训练数据两种模式

## 2. 项目结构

```
AdaDetectGPT/
├── exp_gpt3to4/         # 实验数据和结果
├── figure/              # 论文图表
├── scripts/             # 核心脚本
│   ├── detect_gpt_ada.py    # AdaDetectGPT核心实现
│   ├── local_infer_ada.py   # 本地推理脚本
│   ├── model.py             # 模型加载和处理
│   ├── nuisance_func.py     # 见证函数实现
│   └── utils.py             # 工具函数
├── setup.sh             # 安装脚本
├── requirements.txt     # 依赖列表
└── README.md            # 项目说明
```

## 3. 安装步骤

### 3.1 环境要求
- Python 3.10.8
- PyTorch 2.7.0
- CUDA兼容GPU（推荐96GB内存，如H20-NVLink）

### 3.2 快速安装

```bash
# 1. 克隆仓库（如果尚未克隆）
git clone https://github.com/Mamba413/AdaDetectGPT.git
cd AdaDetectGPT

# 2. 运行安装脚本
bash setup.sh
```

安装脚本会自动执行以下操作：
- 安装requirements.txt中的所有依赖
- 创建必要的数据目录
- 下载writingPrompts数据集
- 下载各种LLM模型（如gpt-j-6B, qwen-7b等）
- 配置NLTK数据（stopwords, wordnet等）

## 4. 简单复现步骤

### 4.1 使用预训练模型快速检测

最简单的方式是使用预训练参数直接检测文本：

```bash
# 使用预训练模型检测单段文本
python scripts/local_infer_ada.py --text "这是一段需要检测的文本，可能是人类撰写或LLM生成的。"
```

### 4.2 使用提供的数据集复现实验

项目提供了多种LLM生成的文本样本，位于`exp_gpt3to4/data/`目录。您可以使用这些数据来复现论文中的实验结果。

#### 4.2.1 复现白盒实验

```bash
# 运行基础白盒实验（评估5个基础LLM）
./exp_whitebox.sh
```

该实验对应论文中的Table 1，评估GPT-2 (1.5B)、GPT-Neo (2.7B)、OPT-2.7B、GPT-J (6B)和GPT-NeoX (20B)等模型。

#### 4.2.2 复现黑盒实验

```bash
# 运行高级黑盒实验（评估高级闭源LLM）
./exp_blackbox_advanced.sh
```

该实验对应论文中的Table 2，评估Gemini-2.5-Flash、GPT-4o和Claude-3.5-Haiku等高级闭源LLM。

#### 4.2.3 复现对抗攻击实验

```bash
# 运行对抗攻击评估实验
./exp_attack.sh
```

该实验对应论文中的Table 3，评估模型在对抗攻击下的性能。

## 5. 核心脚本详解

### 5.1 local_infer_ada.py

主要用于本地推理的脚本，支持以下参数：
- `--text`：需要检测的文本
- `--train_dataset`：训练数据集路径（多个数据集用&分隔）
- `--scoring_model_name`：用于评分的模型名称（默认：gemma-9b）
- `--sampling_model_name`：用于采样的模型名称（默认：gemma-9b-instruct）
- `--w_func`：见证函数类型（identity, pretrained, bspline, bspline_theory）

### 5.2 detect_gpt_ada.py

AdaDetectGPT的核心实现，包含以下主要功能：
- 计算文本的见证函数值
- 进行分类统计（FNR, TNR等）
- 实现自适应检测算法

### 5.3 model.py

负责模型的加载和处理：
- 加载预训练模型和分词器
- 处理模型输出
- 支持多种LLM模型

## 6. 数据集格式

训练数据集应为`.json`格式，命名为`xxx.raw_data.json`，结构如下：

```json
{
  "original": ["human-text-1", "human-text-2", "..."],  // 人类撰写的文本样本
  "sampled": ["machine-text-1", "machine-text-2", "..."]   // LLM生成的文本样本
}
```

## 7. 常见问题与解决方案

### 7.1 CUDA内存不足

**问题**：运行时出现`CUDA out of memory`错误

**解决方案**：
- 使用较小的模型（如gpt-neo-2.7B替代gpt-j-6B）
- 减少batch size
- 使用CPU模式（修改代码中的DEVICE参数为'cpu'）

### 7.2 依赖安装失败

**问题**：setup.sh执行失败，依赖安装出错

**解决方案**：
- 手动安装依赖：`pip install -r requirements.txt`
- 检查Python版本是否为3.10.8
- 确保网络连接正常（需要下载大量模型）

### 7.3 NLTK数据缺失

**问题**：运行时提示NLTK数据缺失

**解决方案**：
```bash
# 手动下载NLTK数据
python -m nltk.downloader stopwords wordnet omw-1.4 punkt_tab
```

### 7.4 模型下载失败

**问题**：setup.sh中模型下载超时或失败

**解决方案**：
- 检查网络连接
- 手动下载模型到cache目录
- 修改scripts/model.py中的模型下载路径

## 8. 实验结果解读

论文中的主要实验结果包括：

### 8.1 白盒实验（Table 1）
- 评估5个基础LLM模型
- 展示AdaDetectGPT在不同模型上的检测性能

### 8.2 黑盒实验（Table 2）
- 评估3个高级闭源LLM模型
- 与其他检测方法进行比较

### 8.3 对抗攻击实验（Table 3）
- 评估模型在对抗攻击下的鲁棒性
- 展示AdaDetectGPT的抗攻击能力

## 9. 引用

如果您使用了本项目的代码或数据，请引用以下论文：

```bibtex
@inproceedings{zhou2025adadetect,
  title={AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees},
  author={Hongyi Zhou and Jin Zhu and Pingfan Su and Kai Ye and Ying Yang and Shakeel A O B Gavioli-Akilagun and Chengchun Shi},
  booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

## 10. 联系方式

如有任何问题或建议，请在GitHub仓库中提交issue：
https://github.com/Mamba413/AdaDetectGPT/issues
# AdaDetectGPT 使用指南

## 1. 项目概述

AdaDetectGPT是一种具有统计保证的自适应LLM生成文本检测方法，发表于NeurIPS 2025会议。该方法基于Fast-DetectGPT进行扩展，通过自适应学习见证函数来提高检测性能，并提供了严格的统计保证。

### 核心功能
- **自适应检测**：从训练数据中学习最优检测策略
- **统计保证**：提供FNR（假阴性率）和TNR（真阴性率）的统计保证
- **高性能**：在多种LLM检测任务中表现优于现有方法
- **灵活使用**：支持带训练数据和不带训练数据两种使用方式

## 2. 环境安装

### 2.1 系统要求
- Python 3.10.8
- PyTorch 2.7.0
- CUDA兼容的GPU（实验使用H20-NVLink，96GB内存）

### 2.2 快速安装

```bash
# 1. 克隆仓库
# git clone https://github.com/Mamba413/AdaDetectGPT.git
# cd AdaDetectGPT

# 2. 运行安装脚本
bash setup.sh
```

安装脚本会自动：
- 安装所需依赖
- 下载必要的数据集
- 准备模型文件
- 配置NLTK数据

## 3. 简单复现步骤

### 3.1 使用预训练模型快速检测

最简单的复现方式是使用预训练参数直接检测文本：

```bash
# 使用预训练模型检测单段文本
python scripts/local_infer_ada.py --text "这是一段需要检测的文本，可能是人类撰写或LLM生成的。"
```

### 3.2 使用提供的数据集进行复现

项目提供了来自多种LLM的生成文本样本，位于`exp_gpt3to4/data/`目录。您可以使用这些数据来复现论文中的实验结果。

#### 3.2.1 复现白盒实验

```bash
# 运行基础白盒实验（评估5个基础LLM）
./exp_whitebox.sh
```

#### 3.2.2 复现黑盒实验

```bash
# 运行高级黑盒实验（评估高级闭源LLM）
./exp_blackbox_advanced.sh
```

#### 3.2.3 复现分析实验

```bash
# 运行对抗攻击评估实验
./exp_attack.sh
```

## 4. 项目结构说明

```
AdaDetectGPT/
├── exp_gpt3to4/       # 实验数据和结果
├── figure/            # 论文图表
├── scripts/           # 核心脚本
│   ├── detect_gpt_ada.py    # AdaDetectGPT核心实现
│   ├── local_infer_ada.py   # 本地推理脚本
│   └── ...                  # 其他检测方法实现
├── setup.sh           # 安装脚本
├── requirements.txt   # 依赖列表
└── README.md          # 项目说明
```

## 5. 关键脚本说明

### 5.1 local_infer_ada.py

主要用于本地推理的脚本，支持以下参数：
- `--text`：需要检测的文本
- `--train_dataset`：训练数据集路径（可选）
- `--model_name`：使用的模型名称（默认：gpt-j-6B）

### 5.2 detect_gpt_ada.py

AdaDetectGPT的核心实现，包含自适应检测算法的主要逻辑。

## 6. 数据集格式

训练数据集应为`.json`格式，命名为`xxx.raw_data.json`，结构如下：

```json
{
  "original": ["human-text-1", "human-text-2", "..."],  // 人类撰写的文本样本
  "sampled": ["machine-text-1", "machine-text-2", "..."]   // LLM生成的文本样本
}
```

## 7. 常见问题与解决方案

### 7.1 内存不足
- 问题：运行时出现CUDA内存不足错误
- 解决：尝试使用较小的模型，或减少batch size

### 7.2 依赖安装失败
- 问题：setup.sh执行失败
- 解决：手动安装requirements.txt中的依赖

### 7.3 NLTK数据缺失
- 问题：运行时提示NLTK数据缺失
- 解决：确保setup.sh正确执行，或手动下载NLTK数据

## 8. 引用

如果您使用了本项目的代码或数据，请引用以下论文：

```bibtex
@inproceedings{zhou2025adadetect,
  title={AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees},
  author={Hongyi Zhou and Jin Zhu and Pingfan Su and Kai Ye and Ying Yang and Shakeel A O B Gavioli-Akilagun and Chengchun Shi},
  booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

## 9. 联系方式

如有任何问题或建议，请在GitHub仓库中提交issue：
https://github.com/Mamba413/AdaDetectGPT/issues

---

**提示**：首次运行可能需要较长时间下载模型和数据。建议在网络良好的环境下进行安装和复现。