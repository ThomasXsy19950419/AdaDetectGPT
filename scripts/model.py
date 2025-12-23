# 导入必要的库
from transformers import AutoModelForCausalLM, AutoTokenizer  # 导入Transformers库的模型和分词器
import torch  # PyTorch深度学习框架
import time  # 时间模块，用于计时
import os  # 操作系统相关功能

try:
    import huggingface_hub  # Hugging Face Hub库
    huggingface_hub.login("hf_xxx")  # replace hf_xxx with your actual token
    # 登录Hugging Face Hub（需要替换为实际的访问令牌）
except:
    pass  # 如果导入失败或登录失败，忽略错误

def from_pretrained(cls, model_name, kwargs, cache_dir):
    """
    自定义的模型加载函数，优先使用本地模型
    
    Args:
        cls: 模型类（如AutoModelForCausalLM）
        model_name: 模型名称
        kwargs: 模型加载参数
        cache_dir: 缓存目录
        
    Returns:
        加载好的模型对象
    """
    # use local model if it exists
    # 优先使用本地模型，如果存在的话
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))  # 构建本地模型路径
    if os.path.exists(local_path):
        # 如果本地模型存在，则加载本地模型
        return cls.from_pretrained(local_path, **kwargs)
    # 否则从Hugging Face Hub加载模型到指定缓存目录
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

# predefined models
# 预定义的模型名称映射字典
# 键：简写名称，值：Hugging Face Hub上的完整模型名称
model_fullnames = {  'gpt2': 'gpt2',
                     'gpt2-xl': 'gpt2-xl',
                     'opt-2.7b': 'facebook/opt-2.7b',
                     'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',  # 适合8GB显卡的2.7B模型
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
                     'mgpt': 'sberbank-ai/mGPT',
                     'pubmedgpt': 'stanford-crfm/pubmedgpt',
                     'mt5-xl': 'google/mt5-xl',
                     'qwen-7b': 'Qwen/Qwen2.5-7B',
                     'qwen-7b-instruct': 'Qwen/Qwen2.5-7B-Instruct',
                     'mistralai-7b': 'mistralai/Mistral-7B-v0.1',
                     'mistralai-7b-instruct': 'mistralai/Mistral-7B-Instruct-v0.3',
                     'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
                     'llama3-8b-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
                     'falcon-7b': 'tiiuae/falcon-7b',
                     'falcon-7b-instruct': 'tiiuae/falcon-7b-instruct',
                     'llama2-13b': 'TheBloke/Llama-2-13B-fp16',
                     'llama2-13b-chat': 'meta-llama/Llama-2-13b-chat-hf', 
                     'gemma-9b': 'google/gemma-2-9b',
                     'gemma-9b-instruct': 'google/gemma-2-9b-it',
                     'bloom-7b1': 'bigscience/bloom-7b1',
                     'opt-13b': 'facebook/opt-13b',
                     'pythia-12b': 'EleutherAI/pythia-12b',
                     }
# 需要使用float16精度加载的模型列表
# 这些模型通常较大，使用float16可以减少显存占用
float16_models = ['gpt-neo-2.7B', 'gpt-j-6B', 'gpt-neox-20b', 'falcon-7b', 'falcon-7b-instruct', 
                  'qwen-7b', 'qwen-7b-instruct', 'mistralai-7b', 'mistralai-7b-instruct', 
                  'llama3-8b', 'llama3-8b-instruct', 'gemma-9b', 'gemma-9b-instruct', 
                  'llama2-13b', 'bloom-7b1', 'opt-13b', 'pythia-12b', 'llama2-13b-chat']

def get_model_fullname(model_name):
    """
    获取模型的完整名称
    
    Args:
        model_name: 模型的简写名称
        
    Returns:
        模型在Hugging Face Hub上的完整名称
    """
    # 如果模型名称在预定义字典中，则返回完整名称，否则返回原名称
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_model(model_name, device, cache_dir, torch_dtype=None):
    """
    加载语言模型
    
    Args:
        model_name: 模型的简写名称
        device: 设备名称（如'cuda'或'cpu'）
        cache_dir: 模型缓存目录
        torch_dtype: 可选，指定模型的数据类型
        
    Returns:
        加载好的模型对象
    """
    model_fullname = get_model_fullname(model_name)  # 获取模型完整名称
    print(f'Loading model {model_fullname}...')  # 打印加载信息
    model_kwargs = {}  # 模型加载参数
    
    # 如果模型在float16列表中，使用float16精度
    if model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    
    # 对于gpt-j模型，使用float16版本
    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))
    
    # 如果指定了数据类型，使用指定的类型
    if torch_dtype is not None:
        model_kwargs.update(dict(torch_dtype=torch_dtype))
    
    # 加载模型
    model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    
    # 将模型移动到指定设备
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')  # 打印移动耗时
    
    return model

def load_tokenizer(model_name, cache_dir):
    """
    加载模型对应的分词器
    
    Args:
        model_name: 模型的简写名称
        cache_dir: 分词器缓存目录
        
    Returns:
        配置好的分词器对象
    """
    model_fullname = get_model_fullname(model_name)  # 获取模型完整名称
    optional_tok_kwargs = {}  # 分词器加载参数
    
    # 对于OPT模型，使用非fast分词器
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    
    # 设置padding side为右侧
    optional_tok_kwargs['padding_side'] = 'right'
    
    # 加载分词器
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    
    # 如果分词器没有pad_token_id，设置为eos_token_id
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        # 对于某些13b模型，使用0作为pad_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    
    return base_tokenizer


if __name__ == '__main__':
    """
    主函数，用于测试模型和分词器的加载
    """
    import argparse  # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="mistralai-7b-instruct", 
                        help="模型名称")
    parser.add_argument('--cache_dir', type=str, default="../cache", 
                        help="模型缓存目录")
    args = parser.parse_args()

    # 加载分词器和模型进行测试
    load_tokenizer(args.model_name, args.cache_dir)
    load_model(args.model_name, 'cpu', args.cache_dir)
