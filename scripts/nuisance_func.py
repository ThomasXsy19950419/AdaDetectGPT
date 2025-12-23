# 导入必要的库
import torch  # PyTorch深度学习框架
from BSpline import BSpline  # B样条实现
from torch import Tensor  # PyTorch张量类型
from torch import nn  # 神经网络模块
from tqdm import tqdm  # 进度条工具

class BSplineTwoSample(nn.Module):
    """
    B样条两样本检验类，用于学习区分人类文本和LLM生成文本的witness函数
    
    Args:
        bspline_args: B样条配置参数
        device: 设备名称（如'cuda'或'cpu'）
    """
    def __init__(self, bspline_args, device):
        super().__init__()
        self.bspline = BSpline(**bspline_args)  # 创建B样条对象
        self.bspline = self.bspline.to(device)  # 将B样条模型移动到指定设备
        pass
    
    def inv_sqrt_matrix(self, M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        计算矩阵的逆平方根
        
        Args:
            M: 输入矩阵，形状为(d, d)
            eps: 用于防止除以零的小值
            
        Returns:
            矩阵的逆平方根，形状为(d, d)
        """
        # 1) 特征值分解
        if M.device.type == "mps":
            # MPS设备特殊处理
            M_cpu = M.detach().to("cpu", dtype=torch.float32)
            w, Q = torch.linalg.eigh(M_cpu)
            w = w.to(M.device)
            Q = Q.to(M.device)
        else:
            w, Q = torch.linalg.eigh(M)                # w: (d,), Q: (d,d) - 特征值和特征向量
        
        # 2) 计算特征值的逆平方根
        w_inv_sqrt = (w.clamp(min=eps) ** -0.5)     # (d,) - 确保特征值不小于eps
        D_inv_sqrt = torch.diag(w_inv_sqrt)        # (d,d) - 对角矩阵
        
        # 3) 重构逆平方根矩阵
        return Q @ D_inv_sqrt @ Q.T                # (d,d) - 矩阵乘法

    def solve_beta_star(self, A: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        求解带约束的优化问题：
            max_{β}  c^T β / sqrt(β^T A β)
            s.t.    d^T β = 0,
        返回具有单位A范数的最优解β*（即β*^T A β* = 1）。

        Args:
            A: (d, d) 对称正定矩阵
            c, d: (d,) 向量
            
        Returns:
            β*: 最优解向量，形状为(d,)
        """
        REGULARIZATION = 0.01  # 正则化参数
        # 求解 B u = a
        alpha = REGULARIZATION * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)  # 正则化项

        # 求解 A v = c 和 A w = d（都在同一设备和数据类型上）
        v = torch.linalg.solve(A+alpha, c)  # shape (d,) - A^{-1} c
        w = torch.linalg.solve(A+alpha, d)  # shape (d,) - A^{-1} d

        # 计算 μ = (d^T A^{-1} c) / (d^T A^{-1} d) = (d·v)/(d·w)
        mu = torch.dot(d, v) / torch.dot(d, w)

        # 未归一化的解 β₀ = A⁻¹(c – μ d) = v – μ w
        beta0 = v - mu * w            # shape (d,)

        # 归一化使 β*^T A β* = 1
        # 首先计算范数平方 = β₀ᵀ A β₀
        norm_sq = torch.dot(beta0, (A+alpha) @ beta0)
        norm = torch.sqrt(norm_sq)

        beta_star = beta0 / norm
        return beta_star

    def compute_beta_hat(
        self,
        z_u_list,   # 人类文本的对数似然列表: list[torch.Tensor], 每个元素形状为 (1, Li)
        z_v_list,   # LLM文本的对数似然列表: list[torch.Tensor], 每个元素形状为 (1, Lj)
        constraint,
    ) -> torch.Tensor:
        """
        计算B样条系数beta_hat
        
        Args:
            z_u_list: 人类文本的对数似然列表
            z_v_list: LLM文本的对数似然列表
            constraint: 是否应用约束
            
        Returns:
            beta_hat: B样条系数，形状为(d,)
        """
        device = z_u_list[0].device  # 获取设备
        d = self.bspline.n_bases  # B样条基函数数量
        if self.bspline.add_intercept:
            d = d + 1  # 如果添加截距项，维度加1

        # 1) 收集序列长度并展平所有z值
        u_lengths = [z.shape[-1] for z in z_u_list]  # 人类文本序列长度
        v_lengths = [z.shape[-1] for z in z_v_list]  # LLM文本序列长度

        # 将所有token合并为一个长的1D张量
        all_u = torch.cat([z.squeeze(0).clamp_min(self.bspline.start) for z in z_u_list], dim=0).to(device)
        all_v = torch.cat([z.squeeze(0).clamp_min(self.bspline.start) for z in z_v_list], dim=0).to(device)

        # 2) 一次性计算B样条基函数
        all_u_feats = self.bspline(all_u)  # shape = (sum(u_lengths), d) - 人类文本的B样条特征
        all_v_feats = self.bspline(all_v)  # shape = (sum(v_lengths), d) - LLM文本的B样条特征

        # 3) 将特征分割回每个序列的张量
        # torch.split在C中实现，所以非常快
        u_feats = list(torch.split(all_u_feats, u_lengths, dim=0))  # 人类文本特征列表
        v_feats = list(torch.split(all_v_feats, v_lengths, dim=0))  # LLM文本特征列表

        # 4) 计算每个序列的特征均值
        # 堆叠的均值，形状为 (n_u, d) 和 (n_v, d)
        u_means = torch.stack([f.mean(dim=0) for f in u_feats], dim=0)  # 人类文本特征均值
        v_means = torch.stack([f.mean(dim=0) for f in v_feats], dim=0)  # LLM文本特征均值

        # 5) 构建delta向量
        delta = v_means.sum(dim=0) - u_means.sum(dim=0)     # (d,) - LLM特征均值与人类特征均值之差

        # --- 3) 计算所有序列中所有token的协方差Σ_u, Σ_v ---
        Sigma_u = torch.zeros((d, d), device=device)  # (d, d) - 人类文本特征协方差
        for i, Fu in enumerate(u_feats):
            Fu_c = Fu - Fu.mean(dim=0, keepdim=True)   # 中心化
            Sigma_u += ((Fu_c.T @ Fu_c) / (Fu_c.shape[0] - 1)) / Fu.shape[0]  # 计算协方差并加权平均
        
        Sigma_v = torch.zeros((d, d), device=device)  # (d, d) - LLM文本特征协方差
        for i, Fv in enumerate(v_feats):
            Fv_c = Fv - Fv.mean(dim=0, keepdim=True)   # 中心化
            Sigma_v += ((Fv_c.T @ Fv_c) / (Fv_c.shape[0] - 1)) / Fv.shape[0]  # 计算协方差并加权平均
        
        Sigma = Sigma_u + Sigma_v  # (d, d) - 总协方差

        # --- 4) 闭式解beta = Σ^{-1} δ，然后归一化 ---
        if constraint:
            # 应用约束条件求解
            beta_hat = self.solve_beta_star(Sigma, delta, u_means.sum(dim=0))
        else:
            # 不应用约束条件，直接求解
            Sigma = self.inv_sqrt_matrix(Sigma)  # 计算协方差的逆平方根
            beta_tilde = Sigma @ delta       # (d,) - 临时解
            beta_hat   = beta_tilde / beta_tilde.norm(p=2)  # 归一化
        
        return beta_hat

    def get_zij(self, token_list, model, args):
        """
        获取文本序列中每个token的对数似然
        
        Args:
            token_list: 分词后的文本列表
            model: 语言模型
            args: 命令行参数
            
        Returns:
            z_list: 对数似然列表，每个元素形状为(1, L)
        """
        model.eval()  # 设置模型为评估模式

        n_samples = len(token_list)  # 样本数量
        z_list = []  # 存储对数似然结果
        
        for idx in tqdm(range(n_samples)):
            tokenized = token_list[idx]  # 获取当前样本的分词结果
            labels = tokenized.input_ids[:, 1:]  # 标签为输入ID的后移一位
            
            # 不计算梯度，提高推理速度
            with torch.no_grad():
                logits_score = model(**tokenized).logits[:, :-1]  # 获取模型的logits输出
            
            # 确保标签的维度与logits匹配
            labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
            z_j_b = torch.log_softmax(logits_score, dim=-1)  # 计算对数概率
            z_j = z_j_b.gather(dim=-1, index=labels).squeeze(-1)  # 获取每个token的对数似然
            z_list.append(z_j)  # 添加到结果列表

        return z_list

    def fit(self, human_token_list, machine_token_list, model, args, constraint=False):
        """
        拟合witness函数，学习B样条系数beta_hat
        
        Args:
            human_token_list: 人类文本的分词列表
            machine_token_list: LLM生成文本的分词列表
            model: 语言模型
            args: 命令行参数
            constraint: 是否应用约束条件
        """
        print("Learning witness function...")  # 打印学习信息
        print("Fetch log-likelihood of human texts...")  # 打印处理信息
        z_ij_u = self.get_zij(human_token_list, model, args)  # 获取人类文本的对数似然
        print("Fetch log-likelihood of LLM texts...")  # 打印处理信息
        z_ij_v = self.get_zij(machine_token_list, model, args)  # 获取LLM文本的对数似然
        beta_hat = self.compute_beta_hat(z_ij_u, z_ij_v, constraint)  # 计算B样条系数
        self.beta_hat = beta_hat  # 保存学习到的系数
        print("beta_hat:", torch.round(beta_hat, decimals=3))  # 打印学习到的系数

    def forward(self, input: Tensor):
        """
        前向传播，计算输入的witness函数值
        
        Args:
            input: 输入张量，形状为(batch_size, sequence_length)
            
        Returns:
            w_value: witness函数值，形状与输入相同
        """
        input_shape = input.shape  # 保存输入形状
        device = input.device  # 获取设备
        # 将输入展平并限制最小值
        flat = input.clamp_min(self.bspline.start).reshape(-1).to(device)
        # 计算B样条特征并与beta_hat相乘
        w_value = self.bspline(flat) @ self.beta_hat
        # 恢复原始形状
        w_value = w_value.reshape(input_shape)
        return w_value

def get_ci_list(text_list, tokenizer, model, w_fun, args):
    """
    获取每个文本的置信区间
    
    Args:
        text_list: 文本列表
        tokenizer: 分词器
        model: 语言模型
        w_fun: witness函数
        args: 命令行参数
        
    Returns:
        c_list: 置信区间列表
    """
    model.eval()  # 设置模型为评估模式

    n_samples = len(text_list)  # 样本数量
    c_list = []  # 存储置信区间结果
    
    for idx in tqdm(range(n_samples)):
        original_text = text_list[idx]  # 获取当前样本
        # 对文本进行分词
        tokenized = tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]  # 标签为输入ID的后移一位
        
        # 不计算梯度，提高推理速度
        with torch.no_grad():
            logits_score = model(**tokenized).logits[:, :-1]  # 获取模型的logits输出
        
        # 确保标签的维度与logits匹配
        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        z_j_b = w_fun(torch.log_softmax(logits_score, dim=-1))  # 计算转换后的对数概率
        probs_ref = torch.softmax(logits_score, dim=-1)  # 计算概率分布
        mean_ref = (probs_ref * z_j_b).sum(dim=-1)  # 计算参考分布的均值
        z_j = z_j_b.gather(dim=-1, index=labels).squeeze(-1)  # 获取每个token的转换后对数似然
        
        ci = (z_j.mean(dim=-1) - mean_ref.mean(dim=-1))[0]  # 计算置信区间
        c_list.append(ci)  # 添加到结果列表
    
    return c_list

class ShiftLearner(nn.Module):
    """
    移位学习器类，用于学习置信区间的移位值
    """
    def __init__(self):
        super().__init__()
        pass

    def fit(self, data, tokenizer, model, w_func, args):
        """
        学习移位值
        
        Args:
            data: 数据集
            tokenizer: 分词器
            model: 语言模型
            w_func: witness函数
            args: 命令行参数
        """
        print("Learning shift...")  # 打印学习信息
        # 获取原始文本的置信区间列表
        ci_hat_list = get_ci_list(data['original'], tokenizer, model, w_func, args)
        c_hat = torch.mean(torch.tensor(ci_hat_list))  # 计算置信区间的均值
        self.c_hat = c_hat  # 保存学习到的移位值
        print("c_hat:", torch.round(c_hat, decimals=3))  # 打印学习到的移位值
