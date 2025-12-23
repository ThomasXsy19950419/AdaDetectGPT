# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 导入必要的库
import random  # 随机数生成

import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
from torch import nn  # 神经网络模块
import tqdm  # 进度条
import argparse  # 命令行参数解析
import json  # JSON处理
from data_builder import load_data  # 加载数据的工具函数
from model import load_tokenizer, load_model  # 加载分词器和模型的工具函数
from metrics import get_roc_metrics, get_precision_recall_metrics  # 评估指标函数
from nuisance_func import BSplineTwoSample  # B样条两样本检验
from nuisance_func_human import BSplineTheory  # 理论B样条检验
from utils import load_training_data, separated_string  # 加载训练数据和字符串分割的工具函数
import time  # 时间模块，用于计时
from utils import GpuMem  # GPU内存监控工具

def get_classification_stat(logits_ref, logits_score, labels, w_func, shift_value=None):
    """
    计算分类统计量，用于判断文本是否为AI生成
    
    Args:
        logits_ref: 参考模型的logits输出
        logits_score: 评分模型的logits输出
        labels: 真实标签（输入文本的token IDs）
        w_func: witness函数，用于转换似然分数
        shift_value: 移位值，用于调整统计量
        
    Returns:
        float: 计算得到的分类统计量
    """
    assert logits_ref.shape[0] == 1  # 确保批大小为1
    assert logits_score.shape[0] == 1  # 确保批大小为1
    assert labels.shape[0] == 1  # 确保批大小为1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))  # 取较小的词汇表大小
        logits_ref = logits_ref[:, :, :vocab_size]  # 截断参考模型的logits
        logits_score = logits_score[:, :, :vocab_size]  # 截断评分模型的logits

    # 确保标签的维度与logits匹配
    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = w_func(torch.log_softmax(logits_score, dim=-1))  # 计算评分模型的转换后的对数似然
    probs_ref = torch.softmax(logits_ref, dim=-1)  # 计算参考模型的概率分布
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)  # 计算真实标签的对数似然
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)  # 计算参考分布的均值
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)  # 计算参考分布的方差

    L = torch.tensor(var_ref.shape[0])  # 获取序列长度
    # 计算标准化的统计量
    stat = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1) - shift_value * L) / var_ref.sum(dim=-1).sqrt()
    stat = stat.mean()  # 取均值
    return stat.item()  # 返回Python标量


def experiment(args):
    """
    执行AdaDetectGPT检测实验
    
    Args:
        args: 命令行参数，包含实验配置
    """
    # 加载评分模型和分词器
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)  # 加载评分模型分词器
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)  # 加载评分模型
    scoring_model.eval()  # 设置评分模型为评估模式
    
    # 如果采样模型与评分模型不同，则分别加载
    if args.sampling_model_name != args.scoring_model_name:
        sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)  # 加载采样模型分词器
        sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)  # 加载采样模型
        sampling_model.eval()  # 设置采样模型为评估模式
    
    # 加载数据
    data = load_data(args.dataset_file)  # 加载数据集
    n_samples = len(data["sampled"])  # 获取样本数量
    # 评估标准
    name = "AdaDetectGPT"  # 实验名称
    criterion_fn = get_classification_stat  # 分类统计函数

    # 选择witness函数类型
    start = time.perf_counter()  # 记录开始时间
    tracker = GpuMem()  # 创建GPU内存监控器
    
    if args.w_func == 'identity':
        # 使用恒等函数作为witness函数
        w_func = nn.Identity()
        beta = None
    else:
        bspline_args = args.config  # B样条参数
        ## 加载训练数据
        print(f"Datasets for learning BSpline: {args.train_dataset}")  # 打印用于学习B样条的数据集
        
        with tracker:
            train_data = load_training_data(args.train_dataset)  # 加载训练数据
            
            # 如果指定了子采样数量，则进行子采样
            if args.num_subsample > 0:
                args.num_subsample = min(args.num_subsample, len(train_data['original']))
                train_data['original'] = train_data['original'][:args.num_subsample]  # 人类文本子采样
                train_data['sampled'] = train_data['sampled'][:args.num_subsample]  # AI生成文本子采样
            
            # 将人类文本转换为token并移动到指定设备
            human_token_list = [scoring_tokenizer(x, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device) for x in train_data['original']]
            
            # 如果使用bspline，则还需要处理AI生成的文本
            if args.w_func == 'bspline':
                machine_token_list = [scoring_tokenizer(x, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device) for x in train_data['sampled']]
            
            # 创建并拟合B样条函数
            if args.w_func == 'bspline':
                w_func = BSplineTwoSample(bspline_args, args.device)
                w_func.fit(human_token_list, machine_token_list, scoring_model, args)  # 拟合两样本B样条
            elif args.w_func == 'bspline_theory':
                w_func = BSplineTheory(bspline_args, machine_text=False)
                w_func.fit(human_token_list, None, scoring_model, args)  # 拟合理论B样条
        
        beta = w_func.beta_hat.detach().cpu().tolist()  # 保存B样条参数
    
    pre_time = time.perf_counter() - start  # 计算准备时间
    pre_memory = tracker.memory_usage()  # 获取内存使用情况
    

    shift_value = torch.zeros(1).to(args.device)  # 移位值，初始化为0
    
    # 设置随机种子，确保结果可复现
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    results = []  # 存储结果
    eval_time_list = []  # 存储评估时间
    eval_memory_list = []  # 存储评估内存使用情况
    
    # 遍历所有样本
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]  # 人类撰写的原始文本
        sampled_text = data["sampled"][idx]  # AI生成的采样文本
        
        # 处理人类原始文本
        start = time.perf_counter()  # 记录开始时间
        with tracker:
            # 对原始文本进行分词
            tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]  # 标签为输入ID的后移一位（用于自回归模型）
            
            # 不计算梯度，提高推理速度
            with torch.no_grad():
                # 使用评分模型获取logits
                logits_score = scoring_model(**tokenized).logits[:, :-1]
                
                # 如果采样模型与评分模型相同，直接复用logits
                if args.sampling_model_name == args.scoring_model_name:
                    logits_ref = logits_score
                else:
                    # 否则使用采样模型获取logits
                    tokenized = sampling_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."  # 确保分词器匹配
                    logits_ref = sampling_model(**tokenized).logits[:, :-1]
                
                # 计算原始文本的分类统计量
                original_crit = criterion_fn(logits_ref, logits_score, labels, w_func, shift_value)
        
        eval_time_list.append(time.perf_counter() - start)  # 记录评估时间
        eval_memory_list.append(tracker.memory_usage())  # 记录内存使用情况
        
        # 处理AI生成的采样文本（类似原始文本的处理过程）
        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.sampling_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = sampling_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = sampling_model(**tokenized).logits[:, :-1]
            sampled_crit = criterion_fn(logits_ref, logits_score, labels, w_func, shift_value)
        
        # 保存当前样本的结果
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})
    eval_time = np.mean(np.array([eval_time_list]))  # 计算平均评估时间
    eval_memory = np.mean(np.array([eval_memory_list]))  # 计算平均评估内存使用
    
    # 保存结果
    results_file = f'{args.output_file}.{name}.{args.w_func}.json'  # 结果文件路径
    # 计算真实/采样文本的预测分数
    predictions = {'real': [x["original_crit"] for x in results],  # 真实文本的预测分数
                'samples': [x["sampled_crit"] for x in results]}  # AI生成文本的预测分数
    
    # 打印预测分数的统计信息
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    
    # 计算ROC曲线和PR曲线指标
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])  # ROC曲线和AUC
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])  # PR曲线和AUC
    
    # 打印评估指标
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # 组织最终结果
    results = { 'name': f'{name}_threshold',  # 实验名称
                'info': {'n_samples': n_samples},  # 样本数量
                'predictions': predictions,  # 预测结果
                'raw_results': results,  # 原始结果
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},  # ROC指标
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},  # PR指标
                'loss': 1 - pr_auc,  # 损失函数
                'beta': beta,  # B样条参数
                'bias': shift_value.detach().cpu().tolist(),  # 移位值
                'compute_info': {'pre_time': pre_time, 'eval_time': eval_time,  # 计算信息
                                    'pre_memory': pre_memory, 'eval_memory': eval_memory,}
                    }
        
    # 保存结果到JSON文件
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')  # 打印结果文件路径

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数
    parser.add_argument('--output_file', type=str, default="./exp_main/results/squad_gemma-9b", help="输出文件前缀")
    parser.add_argument('--dataset', type=str, default="squad", help="数据集名称")
    parser.add_argument('--dataset_file', type=str, default="./exp_main/data/squad_gemma-9b", help="数据集文件路径")
    parser.add_argument('--train_dataset', type=separated_string, default="./exp_main/data/xsum_gemma-9b&./exp_main/data/writing_gemma-9b", help="用于训练B样条的数据集路径")
    parser.add_argument('--num_subsample', type=int, default=-1, help="用于训练w函数的样本数量，-1表示使用所有样本")
    parser.add_argument('--sampling_model_name', type=str, default="gemma-9b", help="采样模型名称")
    parser.add_argument('--scoring_model_name', type=str, default="gemma-9b-instruct", help="评分模型名称")
    parser.add_argument('--w_func', type=str, default='bspline', choices=['identity', 'bspline', 'bspline_theory'], help="witness函数类型")
    parser.add_argument("--config", type=json.loads, default='{"start": -32, "end": 0, "n_bases": 7, "spline_order": 2, "intercept": 1}', help='B样条配置参数（JSON格式）')
    parser.add_argument('--seed', type=int, default=0, help="随机种子")
    parser.add_argument('--device', type=str, default="cuda", help="设备名称（cuda或cpu）")
    parser.add_argument('--cache_dir', type=str, default="../cache", help="模型缓存目录")
    
    # 解析命令行参数
    args = parser.parse_args()

    # 执行实验
    experiment(args)
