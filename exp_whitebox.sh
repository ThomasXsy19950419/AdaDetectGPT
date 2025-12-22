#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 环境设置
echo `date`, Setup the environment ...
set -e  # 如果发生错误则退出脚本

# 准备文件夹
exp_path=exp_main  # 实验主目录
data_path=$exp_path/data  # 数据目录
res_path=$exp_path/results  # 结果目录
mkdir -p $exp_path $data_path $res_path  # 创建目录（如果不存在）

# 定义要使用的数据集和模型
datasets="xsum squad writing"  # 三个数据集：xsum(新闻摘要), squad(问答), writing(写作)
source_models="gpt2-xl opt-2.7b gpt-neo-2.7B "  # 三个基础LLM模型

# 准备数据集
for D in $datasets; do  # 遍历所有数据集
  for M in $source_models; do  # 遍历所有模型
    echo `date`, Preparing dataset ${D}_${M} ...
    # 使用data_builder.py脚本生成数据集
    python scripts/data_builder.py --dataset $D --n_samples 100 --base_model_name $M --output_file $data_path/${D}_${M}
    # 参数说明：
    # --dataset：数据集名称
    # --n_samples：样本数量（每个数据集生成100个样本）
    # --base_model_name：用于生成LLM文本的模型
    # --output_file：输出文件路径
  done
done

# 白盒设置
echo `date`, Evaluate models in the white-box setting:

# 评估AdaDetectGPT
for D in $datasets; do  # 遍历所有数据集
  # 构建训练数据集：使用其他两个数据集（交叉验证）
  train_parts=()
  for d in $datasets; do
    if [[ ${d} != ${D} ]]; then  # 排除当前测试数据集
      train_parts+=("$d")
    fi
  done

  for M in $source_models; do  # 遍历所有模型
    echo `date`, Evaluating AdaDetectGPT on ${D}_${M} ...
    # 构建训练数据集路径（两个数据集用&连接）
    train_dataset="${data_path}/${train_parts[0]}_${M}&${data_path}/${train_parts[1]}_${M}"
    # 运行AdaDetectGPT检测
    python scripts/detect_gpt_ada.py --sampling_model_name $M --scoring_model_name $M --dataset $D --train_dataset "$train_dataset" --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
    # 参数说明：
    # --sampling_model_name：采样模型名称
    # --scoring_model_name：评分模型名称
    # --dataset：数据集名称
    # --train_dataset：训练数据集
    # --dataset_file：测试数据集文件
    # --output_file：输出结果文件
  done
done


# 评估Fast-DetectGPT和快速基线方法
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating Fast-DetectGPT on ${D}_${M} ...
    # 运行Fast-DetectGPT检测（对比方法）
    python scripts/detect_gpt_fast.py --sampling_model_name $M --scoring_model_name $M --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}

    echo `date`, Evaluating baseline methods on ${D}_${M} ...
    # 运行GLTR检测（基线方法）
    python scripts/detect_gltr.py --scoring_model_name $M --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
  done
done

# 评估DNA-GPT
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating DNA-GPT on ${D}_${M} ...
    # 运行DNA-GPT检测（对比方法）
    python scripts/detect_gpt_dna.py --base_model_name $M --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
  done
done

# 评估DetectGPT及其改进版本DetectLLM
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating DetectGPT on ${D}_${M} ...
    # 运行DetectGPT检测（对比方法）
    python scripts/detect_gpt.py --scoring_model_name $M --mask_filling_model_name t5-3b --n_perturbations 100 --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
    # 参数说明：
    # --mask_filling_model_name：用于填充掩码的模型（t5-3b）
    # --n_perturbations：扰动数量（100个）
    
    # 利用DetectGPT生成的扰动来运行DetectLLM
    echo `date`, Evaluating DetectLLM methods on ${D}_${M} ...
    python scripts/detect_llm.py --scoring_model_name $M --dataset $D \
                          --dataset_file $data_path/${D}_${M}.t5-3b.perturbation_100 --output_file $res_path/${D}_${M}
  done
done
