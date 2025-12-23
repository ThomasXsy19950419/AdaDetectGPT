# AdaDetectGPT

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue)](https://neurips.cc/)  <!-- NeurIPS 2025会议论文 -->
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)  <!-- Python版本要求 -->

This repository contains the implementation of [**AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees**](https://arxiv.org/abs/2510.01268), presented at NeurIPS 2025. Our method provides adaptive detection of LLM-generated text with statistical guarantees. We build upon and extend code from [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt).

本仓库包含在NeurIPS 2025会议上发表的论文**AdaDetectGPT**的实现，该方法提供具有统计保证的自适应LLM生成文本检测。我们的方法基于并扩展了[Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt)的代码。

## 📋 Overview
## 📋 概述

![AdaDetectGPT Workflow](figure/AdaDetectGPT.png)

Workflow of **AdaDetectGPT**. Built upon Fast-DetectGPT (Bao et al., 2024), our method adaptively learn a witness function $\hat{w}$ from training data by maximizing a lower bound on the TNR, while using normal approximation for FNR control.

**AdaDetectGPT**的工作流程：基于Fast-DetectGPT (Bao et al., 2024)，我们的方法通过最大化TNR（真阴性率）下界来自适应地从训练数据中学习见证函数$\hat{w}$，同时使用正态近似来控制FNR（假阴性率）。

## 🛠️ Installation
## 🛠️ 安装

### Requirements
### 系统要求
- Python 3.10.8  <!-- Python版本要求 -->
- PyTorch 2.7.0  <!-- PyTorch框架要求 -->
- CUDA-compatible GPU (experiments conducted on H20-NVLink with 96GB memory)  <!-- CUDA兼容GPU要求 -->

### Setup
### 安装步骤
```bash
bash setup.sh  # 执行安装脚本
```

*Note: While our experiments used high-memory GPUs, typical usage of AdaDetectGPT requires significantly less memory.*

*注意：虽然我们的实验使用了高内存GPU，但AdaDetectGPT的典型使用场景所需内存要少得多。*

## 💻 Usage
## 💻 使用方法

### With Training Data (Recommended)
### 使用训练数据（推荐）

For optimal performance, we recommend using training data. The training dataset should be a `.json` file named `xxx.raw_data.json` with the following structure:

为获得最佳性能，我们建议使用训练数据。训练数据集应为`.json`格式的文件，命名为`xxx.raw_data.json`，具有以下结构：

```json
{
  "original": ["human-text-1", "human-text-2", "..."],  // 人类撰写的文本样本
  "sampled": ["machine-text-1", "machine-text-2", "..."]   // LLM生成的文本样本
}
```

Run detection with training data:
使用训练数据运行检测：
```bash
python scripts/local_infer_ada.py \
  --text "Your text to be detected" \
  --train_dataset "train-data-file-name"  ## 多个训练数据集用&分隔
```

A quick example is: 
快速示例：
```bash
python scripts/local_infer_ada.py \
  --text "Your text to be detected" \
  --train_dataset "./exp_gpt3to4/data/essay_claude-3-5-haiku&./exp_gpt3to4/data/xsum_claude-3-5-haiku"
```

### Without Training Data
### 不使用训练数据

AdaDetectGPT can also use pretrained parameters (trained on texts from GPT-4o, Gemini-2.5, and Claude-3.5):

AdaDetectGPT也可以使用预训练参数（这些参数在GPT-4o、Gemini-2.5和Claude-3.5生成的文本上训练）：

```bash
python scripts/local_infer_ada.py --text "Your text to be detected"  # 使用预训练参数进行检测
```

## 🔬 Reproducibility
## 🔬 实验复现

We provide generated text samples from GPT-3.5-Turbo, GPT-4, GPT-4o, Gemini-2.5, and Claude-3.5 in `exp_gpt3to4/data/` for convenient reproduction. Data from GPT-3.5-Turbo and GPT-4 are sourced from [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt).

我们在`exp_gpt3to4/data/`目录中提供了来自GPT-3.5-Turbo、GPT-4、GPT-4o、Gemini-2.5和Claude-3.5的生成文本样本，方便复现实验。GPT-3.5-Turbo和GPT-4的数据来自[Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt)。

### Experiment Scripts
### 实验脚本

#### White-box Experiments
#### 白盒实验
- `./exp_whitebox.sh` - Table 1: Evaluation on 5 base LLMs
  - GPT-2 (1.5B), GPT-Neo (2.7B), OPT-2.7B, GPT-J (6B), GPT-NeoX (20B)
  - 对5个基础LLM的评估

- `./exp_whitebox_advanced.sh` - Table S7: Advanced open-source LLMs
  - Qwen-2.5 (7B), Mistral (7B), Llama3 (8B)
  - 对高级开源LLM的评估

#### Black-box Experiments
#### 黑盒实验
- `./exp_blackbox_advanced.sh` - Table 2 and Table S8: Advanced closed-source LLMs
  - Gemini-2.5-Flash, GPT-4o, Claude-3.5-Haiku
  - 对高级闭源LLM的评估

- `./exp_blackbox_simple.sh` - Table S2: Five open-source LLMs
  - 对5个开源LLM的评估

#### Analysis Experiments
#### 分析实验
- `./exp_attack.sh` - Table 3: Adversarial attack evaluation
  - 对抗攻击评估

- `./exp_normal.sh` - Data for Figure 3 and Figure S8
  - 生成图3和图S8的数据

- `./exp_sample.sh` - Training data size effects (Figure S5)
  - 训练数据大小的影响

- `./exp_tuning.sh` - Hyperparameter robustness (Figure S6)
  - 超参数鲁棒性

- `./exp_dist_shift.sh` - Distribution shift analysis (Figure S7)
  - 分布偏移分析

- `./exp_compute.sh` - Computational cost analysis (Table S9 and S10)
  - 计算成本分析

- `./exp_variance.sh` - Equal variance condition verification (Table S5)
  - 等方差条件验证

## 🎁 Additional Resources
## 🎁 其他资源

The `scripts/` directory contains implementations of various LLM detection methods from the literature. These implementations are modified from their official versions or the repo of [FastDetectGPT](https://github.com/baoguangsheng/fast-detect-gpt) to provide:
- Consistent input/output formats
- Simplified method comparison

`scripts/`目录包含了文献中各种LLM检测方法的实现。这些实现是从官方版本或[FastDetectGPT](https://github.com/baoguangsheng/fast-detect-gpt)的仓库修改而来，提供了：
- 一致的输入/输出格式
- 简化的方法比较

The provided methods are summarized below.

下表总结了提供的方法：

| Method | Script File | Paper/Website |
|--------|------------|---------------|
| **AdaDetectGPT** | `detect_gpt_ada.py` | [arXiv:2510.01268](https://arxiv.org/abs/2510.01268) |
| **Binoculars** | `detect_binoculars.py` | [arXiv:2401.12070](https://arxiv.org/abs/2401.12070) |
| **BiScope** | `detect_biscope.py` | [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/95814) |
| **DetectGPT** | `detect_gpt.py` | [arXiv:2301.11305](https://arxiv.org/abs/2301.11305) |
| **DetectLLM** | `detect_llm.py` | [arXiv:2306.05540](https://arxiv.org/abs/2306.05540) |
| **DNA-GPT** | `detect_gpt_dna.py` | [arXiv:2305.17359](https://arxiv.org/abs/2305.17359) |
| **Fast-DetectGPT** | `detect_gpt_fast.py` | [arXiv:2310.05130](https://arxiv.org/abs/2310.05130) |
| **GLTR** | `detect_gltr.py` | [arXiv:1906.04043](https://arxiv.org/abs/1906.04043) |
| **ImBD** | `detect_ImBD.py` | [arXiv:2412.10432](https://arxiv.org/abs/2412.10432) |
| **GPTZero** | `detect_gptzero.py` | [GPTZero.me](https://gptzero.me/) |
| **RADAR** | `detect_radar.py` | [arXiv:2307.03838](https://arxiv.org/abs/2307.03838) |
| **RoBERTa OpenAI Detector** | `detect_roberta.py` | [arXiv:1908.09203](https://arxiv.org/abs/1908.09203) |
| **Text Fluoroscopy** | `detect_fluoroscopy.py` | [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.885/) |

We hope these resources facilitate your research and applications in LLM-generated text detection!

我们希望这些资源能促进您在LLM生成文本检测方面的研究和应用！

## 📖 Citation
## 📖 引用

If you find this work useful, please consider citing our paper:

如果您觉得这项工作有用，请考虑引用我们的论文：

```bibtex
@inproceedings{zhou2025adadetect,
  title={AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees},
  author={Hongyi Zhou and Jin Zhu and Pingfan Su and Kai Ye and Ying Yang and Shakeel A O B Gavioli-Akilagun and Chengchun Shi},
  booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

## 📧 Contact
## 📧 联系方式

For any questions/suggestions/bugs, feel free to open an [issue](https://github.com/Mamba413/AdaDetectGPT/issues) in the repository.

如有任何问题/建议/错误，请随时在仓库中打开[issue](https://github.com/Mamba413/AdaDetectGPT/issues)。