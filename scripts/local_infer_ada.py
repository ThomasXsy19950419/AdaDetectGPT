# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 导入必要的库
import random  # 随机数生成
import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
from torch import nn  # PyTorch神经网络模块
import argparse  # 命令行参数解析
from model import load_tokenizer, load_model  # 加载分词器和模型的工具函数
from nuisance_func import BSplineTwoSample  # B样条两样本检验
from nuisance_func_human import BSplineTheory  # 理论B样条检验
from utils import load_training_data, separated_string  # 工具函数：加载训练数据和字符串分割
import json  # JSON处理
from detect_gpt_ada import get_classification_stat  # 获取分类统计结果的函数

def load_model_and_tokenizer(args):
    """
    加载评分模型、采样模型及其对应的分词器
    
    Args:
        args: 命令行参数，包含模型名称、缓存目录等
        
    Returns:
        scoring_tokenizer: 评分模型的分词器
        scoring_model: 评分模型（用于计算文本的似然分数）
        sampling_tokenizer: 采样模型的分词器
        sampling_model: 采样模型（用于生成参考分布）
    """
    DEVICE = 'cuda'  # 使用GPU设备

    # load model 加载模型
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)  # 加载评分模型分词器
    scoring_model = load_model(args.scoring_model_name, DEVICE, args.cache_dir)  # 加载评分模型
    scoring_model.eval()  # 设置评分模型为评估模式
    
    # 如果采样模型与评分模型不同，则分别加载
    if args.sampling_model_name != args.scoring_model_name:
        sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)  # 加载采样模型分词器
        sampling_model = load_model(args.sampling_model_name, DEVICE, args.cache_dir)  # 加载采样模型
        sampling_model.eval()  # 设置采样模型为评估模式
    else:
        # 如果模型相同，直接复用
        sampling_tokenizer = scoring_tokenizer
        sampling_model = scoring_model

    return scoring_tokenizer, scoring_model, sampling_tokenizer, sampling_model

def run(args, scoring_tokenizer, scoring_model, sampling_tokenizer, sampling_model):
    """
    执行文本检测的主函数
    
    Args:
        args: 命令行参数
        scoring_tokenizer: 评分模型分词器
        scoring_model: 评分模型
        sampling_tokenizer: 采样模型分词器
        sampling_model: 采样模型
        
    Returns:
        float: 检测统计量，用于判断文本是否为AI生成
    """
    SEED = 2025  # 随机种子，确保结果可复现
    DEVICE = 'cuda'  # 使用GPU设备
    
    # evaluate criterion
    criterion_fn = get_classification_stat  # 分类统计函数，用于计算检测统计量
    
    # w function: 选择witness函数类型
    if args.w_func == 'identity':
        w_func = nn.Identity()  # 恒等函数
        beta = None  # 没有参数
    else:
        bspline_args = args.config  # B样条参数
        ## load training data 加载训练数据
        print(f"Datasets for learning BSpline: {args.train_dataset}")  # 打印用于学习B样条的数据集

        if args.w_func == 'pretrained':
            # 使用预训练的B样条参数
            w_func = BSplineTwoSample(bspline_args)
            w_func.beta_hat = torch.tensor([0.0, -0.011333, -0.037667, -0.056667, -0.281667, -0.592, 0.157833, 0.727333,]).to(DEVICE)
        else:
            # 根据选择创建不同类型的B样条函数
            if args.w_func == 'bspline':
                w_func = BSplineTwoSample(bspline_args)
            elif args.w_func == 'bspline_theory':
                w_func = BSplineTheory(bspline_args, machine_text=False)

            # 加载训练数据并拟合B样条函数
            train_data = load_training_data(args.train_dataset)
            w_func.fit(train_data, scoring_tokenizer, scoring_model, args)
        
        beta = w_func.beta_hat.detach().cpu().tolist()  # 保存B样条参数

    shift_value = torch.zeros(1).to(DEVICE)  # 移位值，初始化为0
    
    # 设置随机种子
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    text = args.text  # 获取待检测文本
    
    # 使用评分模型分词器对文本进行分词
    tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(DEVICE)
    labels = tokenized.input_ids[:, 1:]  # 标签为输入ID的后移一位（用于自回归模型）
    
    # 计算burn_in数量（忽略文本开头的tokens）
    if args.burn_in < 1.0:
        burn_in_num = int(args.burn_in * labels.size(-1))  # 比例方式
    else:
        burn_in_num = int(args.burn_in)  # 固定数量方式
    
    # 不计算梯度，提高推理速度
    with torch.no_grad():
        # 使用评分模型获取logits
        logits_score = scoring_model(**tokenized).logits[:, :-1]
        
        # 如果采样模型与评分模型相同，直接复用logits
        if args.sampling_model_name == args.scoring_model_name:
            logits_ref = logits_score
        else:
            # 否则使用采样模型获取logits
            tokenized = sampling_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(DEVICE)
            assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."  # 确保分词器匹配
            logits_ref = sampling_model(**tokenized).logits[:, :-1]
        
        # 计算检测统计量
        original_crit = criterion_fn(logits_ref, logits_score, labels, burn_in_num, w_func, shift_value)

    # 结果字典
    results = {
        "text": text,  # 输入文本
        "crit": original_crit,  # 检测统计量
        'beta': beta,  # B样条参数
    }
    
    return results['crit']


if __name__ == '__main__':
    """
    主程序入口
    解析命令行参数，加载模型，执行文本检测
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数
    parser.add_argument('--text', type=str, help="Your text goes here. It can be a long text, and the longer the text, the more reliable the result will be. 待检测的文本内容，文本越长结果越可靠")
    parser.add_argument('--train_dataset', type=separated_string, default="./exp_main/data/xsum_gpt2-xl&./exp_main/data/writing_gpt2-xl", help="用于训练B样条的数据集路径")
    parser.add_argument('--sampling_model_name', type=str, default="gpt-neo-2.7B", help="采样模型名称，用于生成参考分布")
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B", help="评分模型名称，用于计算文本的似然分数")
    parser.add_argument('--burn_in', type=float, default=0.0, help="忽略的开头token比例或数量")
    parser.add_argument('--w_func', type=str, default='bspline', choices=['identity', 'pretrained', 'bspline', 'bspline_theory'], help="选择witness函数类型")
    parser.add_argument("--config", type=json.loads, default='{"start": -32, "end": 0, "n_bases": 7, "spline_order": 2, "intercept": 1}', help='B样条配置参数')
    parser.add_argument('--cache_dir', type=str, default="../cache", help="模型缓存目录")
    
    # 解析参数
    args = parser.parse_args()

    # 加载模型和分词器
    scoring_tokenizer, scoring_model, sampling_tokenizer, sampling_model = load_model_and_tokenizer(args)
    
    # 执行检测
    result = run(args, scoring_tokenizer, scoring_model, sampling_tokenizer, sampling_model)
    
    # 打印结果
    print(f"检测结果 (统计量): {result}")
    print(f"提示: 统计量越大，文本越可能是AI生成的")