# utils.py
import os
import pickle
import requests
import numpy as np
import torch
from torch.utils.data import Dataset

class DataLoader:
    def __init__(self, data_dir='data', dataset='shakespeare'):
        self.data_dir = data_dir
        self.dataset = dataset
        self.vocab_size = None
        self.stoi = None
        self.itos = None
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 获取数据
        input_file_path = os.path.join(data_dir, 'input.txt')
        if not os.path.exists(input_file_path):
            self.download_dataset()
            
        # 处理数据
        self.process_data()
    
    def download_dataset(self):
        """下载数据集"""
        print(f"下载 {self.dataset} 数据集...")
        if self.dataset == 'shakespeare':
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        else:  # 默认使用 shakespeare
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        
        with open(os.path.join(self.data_dir, 'input.txt'), 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)
    
    def process_data(self):
        """处理数据并创建字符映射"""
        data_path = os.path.join(self.data_dir, 'input.txt')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.read()
        
        # 获取所有唯一字符
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        
        # 创建字符到索引的映射
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # 将整个数据集编码为整数
        self.data = data
        self.encoded_data = [self.stoi[c] for c in data]
        
        # 分割训练集和验证集
        n = len(self.encoded_data)
        self.train_data = self.encoded_data[:int(n*0.9)]
        self.val_data = self.encoded_data[int(n*0.9):]
    
    def get_batch(self, split, block_size, batch_size):
        """获取一批数据"""
        data = self.train_data if split == 'train' else self.val_data
        
        # 随机选择起始位置
        ix = torch.randint(len(data) - block_size, (batch_size,))
        
        # 创建输入和目标张量
        x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
        y = torch.stack([torch.tensor(data[i+1:i+1+block_size]) for i in ix])
        
        return x, y

# 全局数据加载器
global_loader = None

def get_batch(split, block_size, batch_size, data_dir='data', dataset='shakespeare'):
    """获取一批数据的公共函数"""
    global global_loader
    if global_loader is None:
        global_loader = DataLoader(data_dir, dataset)
    return global_loader.get_batch(split, block_size, batch_size)

def encode(s):
    """编码字符串为整数列表"""
    global global_loader
    if global_loader is None:
        global_loader = DataLoader()
    return [global_loader.stoi[c] for c in s]

def decode(l):
    """解码整数列表为字符串"""
    global global_loader
    if global_loader is None:
        global_loader = DataLoader()
    return ''.join([global_loader.itos[i] for i in l])