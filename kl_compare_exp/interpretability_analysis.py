#!/usr/bin/env python3
"""
LoopLLM可解释性分析脚本
分析每一层的隐状态经过lm_head后的概率分布，计算KL散度并可视化
"""
import os
# 挂梯子加载数据集
# os.environ["http_proxy"] = "http://127.0.0.1:7891"
# os.environ["https_proxy"] = "http://127.0.0.1:7891"
import sys

# 添加LoopLLM目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import signal
import time
import psutil
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any
import warnings
from scipy import interpolate
from scipy.optimize import curve_fit
from tqdm import tqdm, trange
from dataclasses import dataclass
import hydra
from omegaconf import DictConfig, OmegaConf
warnings.filterwarnings('ignore')

from loop_llama_config import LoopLlamaConfig
from loop_llama_model import LoopLlamaForCausalLM
from transformers import LlamaConfig


def memory_monitor():
    """监控内存使用"""
    process = psutil.Process()
    memory_gb = process.memory_info().rss / 1024**3
    # print(f"当前内存使用: {memory_gb:.2f} GB")
    if memory_gb > 200:  # 如果超过200GB则警告
        print("⚠️ 内存使用过高，建议减少数据量")
    return memory_gb


def timeout_handler(signum, frame):
    """超时处理函数"""
    raise TimeoutError("操作超时")


class LoopLLMInterpreter:
    """LoopLLM可解释性分析器"""
    
    def __init__(self, model_config: LoopLlamaConfig, model_path: str, output_dir: str = "interpretability_results"):
        """初始化分析器
        
        Args:
            model_config: LoopLLM模型配置
            output_dir: 输出目录路径
        """
        self.config = model_config
        self.output_dir = output_dir
        print("正在初始化模型...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = LoopLlamaForCausalLM.from_pretrained(model_path, config=model_config, device_map="auto")
            self.model.eval()
            print("✓ 模型初始化成功")
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            raise
        
        # 存储每层的隐状态
        self.layer_hidden_states = []
        self.create_output_directory()
        
    def create_output_directory(self) -> str:
        """创建输出目录
        
        Returns:
            创建的子目录路径
        """
        # 确保基础输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建子目录名称
        if self.config.loop_layers is not None:
            # 暂时只有一个循环块
            start_layer, end_layer = self.config.loop_layers[0]
            loop_count = self.config.loop_count
            subdir_name = f"output_{loop_count}"
        else:
            subdir_name = f"output_no_loop"
        
        # 完整的子目录路径
        full_subdir_path = os.path.join(self.output_dir, subdir_name)
        
        # 创建子目录
        os.makedirs(full_subdir_path, exist_ok=True)
        print(f"输出目录: {full_subdir_path}")
        
        self.output_dir = full_subdir_path
        
    def load_data(self, num_samples: int = 50) -> List[str]:  # 默认减少到50
        """加载数据集
        
        Args:
            num_samples: 要加载的样本数量
            
        Returns:
            文本样本列表
        """
        print(f"正在加载数据集的前{num_samples}条数据...")
        
        # 首先尝试读取本地保存的pile_sample_10.json文件
        local_file = "pile_sample.json"
        if os.path.exists(local_file):
            try:
                print(f"发现本地文件 {local_file}，尝试读取...")
                with open(local_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取文本内容
                if isinstance(data, list):
                    # 如果是列表格式
                    texts = []
                    for item in data:
                        if isinstance(item, dict) and 'text' in item:
                            texts.append(item['text'])
                        elif isinstance(item, str):
                            texts.append(item)
                    
                    # 根据需要的样本数量进行截取或重复
                    if len(texts) >= num_samples:
                        texts = texts[:num_samples]
                    
                    print(f"✓ 成功从本地文件加载 {len(texts)} 条数据")
                    return texts
                    
                elif isinstance(data, dict) and 'texts' in data:
                    # 如果是字典格式，包含texts字段
                    texts = data['texts'][:num_samples]
                    print(f"✓ 成功从本地文件加载 {len(texts)} 条数据")
                    return texts
                    
                else:
                    print(f"⚠️ 本地文件格式不符合预期，回退到在线数据集，可以使用prepare_dataset.py生成pile_sample.json文件")
                    
            except Exception as e:
                print(f"⚠️ 读取本地文件失败: {e}")
                print("回退到在线数据集，可以使用prepare_dataset.py生成pile_sample.json文件")
        else:
            print(f"本地文件 {local_file} 不存在，使用在线数据集，可以使用prepare_dataset.py生成pile_sample.json文件")
        
        # 回退到原来的数据集加载方法
        # 设置超时
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60秒超时
        
        print("尝试加载EleutherAI/the_pile_deduplicated数据集...")
        # 加载数据集
        try:
            dataset = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)
            
            # 获取前num_samples条数据
            texts = []
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                texts.append(example['text'])
                
                if (i + 1) % 10 == 0:  # 更频繁的进度报告
                    print(f"已加载 {i + 1} 条数据...")
            
            signal.alarm(0)  # 取消超时
            print(f"✓ 成功加载 {len(texts)} 条数据")
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise
        return texts
   
    def forward_with_hidden_states(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """前向传播并获取所有层的隐状态
        
        Args:
            input_ids: 输入token张量
            
        Returns:
            包含所有层隐状态和输出的字典
        """
        # print("正在进行前向传播...")
        
        try:
            with torch.no_grad():
                # 获取模型输出，包括所有隐状态
                outputs = self.model(
                    input_ids=input_ids.to(self.model.device),
                    output_hidden_states=True,
                    use_cache=False
                )
                
                hidden_states = outputs.hidden_states  # 包含所有层的隐状态
                logits = outputs.logits
                
                # print(f"✓ 获得 {len(hidden_states)} 层隐状态")
                
                return {
                    'hidden_states': hidden_states,
                    'logits': logits,
                    'final_hidden_state': outputs.hidden_states[-1]
                }
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            raise
    
    def compute_layer_probabilities(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        """计算每层隐状态经过lm_head后的概率分布
        
        Args:
            hidden_states: 所有层的隐状态列表
            
        Returns:
            每层的概率分布列表
        """
        # print("正在计算每层的概率分布...")
        
        probabilities = []
        
        for i, hidden_state in enumerate(hidden_states):
            try:
                # ⭐ 关键修改：为每层hidden state应用normalization
                # 这样确保与最后一层的处理方式一致
                normalized_hidden = self.model.model.norm(hidden_state)
                
                # 添加诊断信息
                # if i == 0 or i == len(hidden_states) - 1:  # 只在第一层和最后一层打印
                #     print(f"\n第{i+1}层详细统计:")
                #     print(f"  原始hidden_state: mean={hidden_state.mean().item():.6f}, std={hidden_state.std().item():.6f}")
                #     print(f"  标准化后: mean={normalized_hidden.mean().item():.6f}, std={normalized_hidden.std().item():.6f}")
                
                # 通过lm_head获取logits
                logits = self.model.lm_head(normalized_hidden)
                
                # 添加logits统计
                # if i == 0 or i == len(hidden_states) - 1:
                #     print(f"  logits: mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")
                #     print(f"  logits范围: [{logits.min().item():.6f}, {logits.max().item():.6f}]")
                    
                #     # 🔍 检查logits的分布
                #     logits_sorted, _ = torch.sort(logits.view(-1), descending=True)
                #     print(f"  logits前10大值: {logits_sorted[:10].tolist()}")
                #     print(f"  logits后10小值: {logits_sorted[-10:].tolist()}")
                
                # 应用softmax获取概率分布
                probs = F.softmax(logits, dim=-1).cpu()
                
                # 添加概率分布统计
                # if i == 0 or i == len(hidden_states) - 1:
                #     print(f"  probs: mean={probs.mean().item():.8f}, std={probs.std().item():.8f}")
                #     print(f"  最大概率: {probs.max().item():.8f}")
                #     print(f"  最小概率: {probs.min().item():.8e}")
                    
                #     # 🔍 检查概率分布
                #     probs_sorted, indices = torch.sort(probs.view(-1), descending=True)
                #     print(f"  概率前10大值: {probs_sorted[:10].tolist()}")
                #     print(f"  对应的token indices: {indices[:10].tolist()}")
                    
                #     # 🔍 验证softmax计算
                #     print(f"  概率求和: {probs.sum(dim=-1).mean().item():.6f} (应该≈1.0)")
                    
                #     # 🔍 检查是否有异常高的logits
                #     max_logit_pos = torch.argmax(logits.view(-1))
                #     max_logit_value = logits.view(-1)[max_logit_pos]
                #     max_prob_value = probs.view(-1)[max_logit_pos]
                #     print(f"  最大logit位置: {max_logit_pos.item()}, 值: {max_logit_value.item():.6f}")
                #     print(f"  对应概率: {max_prob_value.item():.8f}")
                    
                #     print(f"  熵: {-(probs * probs.log()).sum(dim=-1).mean().item():.6f}")
                #     print(f"  理论最大熵 (log({self.config.vocab_size})): {np.log(self.config.vocab_size):.6f}")
                
                probabilities.append(probs)
                
                # if (i + 1) % 5 == 0:
                #     print(f"已处理 {i + 1} 层...")  
                    
                # 强制垃圾回收
                del logits, normalized_hidden
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
            except Exception as e:
                print(f"❌ 处理第{i+1}层时出错: {e}")
                raise
        
        return probabilities
    
    def compute_kl_divergence(self, prob_p: torch.Tensor, prob_q: torch.Tensor) -> float:
        """计算两个概率分布之间的KL散度 KL(P||Q)
        
        Args:
            prob_p: 概率分布P (真实分布或参考分布)
            prob_q: 概率分布Q (近似分布或目标分布)
            
        Returns:
            KL散度值 KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
            
        Note:
            KL散度衡量P和Q之间的差异，值越大表示差异越大
            KL(P||Q) != KL(Q||P)，不对称
        """
        # 避免数值不稳定
        eps = 1e-10
        prob_p = (prob_p + eps)
        prob_q = (prob_q + eps)
        
        # PyTorch的kl_div函数:
        # F.kl_div(input=log(Q), target=P) = KL(P||Q)
        # 所以我们传入 log(prob_q) 作为input，prob_p 作为target
        # 重新归一化
        prob_p = prob_p / prob_p.sum(dim=-1, keepdim=True)
        prob_q = prob_q / prob_q.sum(dim=-1, keepdim=True)
        
        # 计算KL散度: KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
        kl_div = torch.sum(prob_p * torch.log(prob_p / prob_q), dim=-1)
        
        return kl_div
        # return kl_div.item()
    
    def plot_layer_vs_final_kl(self, kl_divergences: List[float], save_path: str = None):
        """绘制每层与最后一层KL散度的柱状图
        
        Args:
            kl_divergences: KL散度列表
            save_path: 保存文件名（如果为None则自动生成）
        """
        
        # 如果没有指定文件名，则自动生成
        if save_path is None:
            if self.config.loop_layers is not None:
                start_layer, end_layer = self.config.loop_layers
                loop_count = self.config.loop_count
                save_path = f"layer_vs_final_kl_{start_layer}_{end_layer}_{loop_count}.png"
            else:
                save_path = f"layer_vs_final_kl_no_loop.png"
        
        # 构建完整的保存路径
        full_save_path = os.path.join(self.output_dir, save_path)
        
        plt.figure(figsize=(14, 8))
        
        num_layers = len(kl_divergences)
        # 使用相对层数（0到1之间）
        relative_layer_indices = np.linspace(0, 1, num_layers)
        
        width = 0.02  # 调整柱子宽度以适应相对层数
        bars = plt.bar(relative_layer_indices, kl_divergences, width=width, alpha=0.8, 
                      color='skyblue', edgecolor='navy', linewidth=1.5, label='KL divergence')
        
        # 添加曲线拟合
        try:
            # 使用多项式拟合（3次多项式）
            # degree = min(3, num_layers - 1)  # 确保多项式次数不超过数据点数-1
            # coeffs = np.polyfit(relative_layer_indices, kl_divergences, degree)
            # poly_func = np.poly1d(coeffs)
            
            # # 生成平滑的曲线点
            smooth_x = np.linspace(0, 1, 100)
            # smooth_y = poly_func(smooth_x)
            
            # plt.plot(smooth_x, smooth_y, 'r-', linewidth=2.5, alpha=0.8, label=f'{degree}th-degree polynomial fitting')
            
            # 同时尝试样条插值拟合
            if num_layers >= 4:  # 样条插值需要至少4个点
                spline = interpolate.UnivariateSpline(relative_layer_indices, kl_divergences, s=0, k=min(3, num_layers-1))
                spline_y = spline(smooth_x)
                plt.plot(smooth_x, spline_y, 'g--', linewidth=2, alpha=0.7, label='spline interpolation fitting')
            
        except Exception as e:
            print(f"⚠️ 曲线拟合失败: {e}")
        
        # 添加循环层标记线
        if self.config.loop_layers is not None:
            start_layer, end_layer = self.config.loop_layers
            real_end_layer = start_layer + self.config.loop_count * (end_layer - start_layer + 1)
            
            # 计算循环层在相对位置中的坐标
            start_relative = start_layer / (num_layers) if num_layers > 1 else 0
            end_relative = real_end_layer / (num_layers) if num_layers > 1 else 0
            
            # 添加循环开始和结束的垂直线
            plt.axvline(x=start_relative, color='red', linestyle='--', linewidth=2, alpha=0.8)
            if start_layer != real_end_layer:
                plt.axvline(x=end_relative, color='red', linestyle='--', linewidth=2, alpha=0.8)
            
            # 添加循环区域的背景高亮
            # plt.axvspan(start_relative, end_relative, alpha=0.1, color='red', 
            #            label=f'Loop Region (Layers {start_layer}-{end_layer})')
        
        # 设置标题和标签
        title_suffix = ""
        if self.config.loop_layers is not None:
            start_layer, end_layer = self.config.loop_layers
            loop_count = self.config.loop_count
            title_suffix = f" (loop layers: {start_layer}-{end_layer}, loop count: {loop_count})"
        
        plt.xlabel('Relative Layer Position', fontsize=14)
        plt.ylabel('KL Divergence', fontsize=14)
        plt.title(f'KL Divergence Distribution Between Each Layer and Final Layer{title_suffix}', fontsize=16, pad=20)
        
        # 设置x轴刻度和标签
        tick_positions = np.linspace(0, 1, min(11, num_layers))  # 最多显示11个刻度
        tick_labels = [f'{pos:.1f}' for pos in tick_positions]
        plt.xticks(tick_positions, tick_labels, fontsize=12)
        plt.yticks(fontsize=12)
        
        # 添加网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 添加图例
        plt.legend(fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.9)
        
        # 保存图片
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"柱状图已保存至: {full_save_path}")
    
    def plot_all_pairs_heatmap(self, kl_matrix: np.ndarray, save_path: str = None):
        """绘制所有层对KL散度的热力图
        
        Args:
            kl_matrix: KL散度矩阵
            save_path: 保存文件名（如果为None则自动生成）
        """
        
        # 如果没有指定文件名，则自动生成
        if save_path is None:
            if self.config.loop_layers is not None:
                start_layer, end_layer = self.config.loop_layers
                loop_count = self.config.loop_count
                save_path = f"all_pairs_kl_heatmap_{start_layer}_{end_layer}_{loop_count}.png"
            else:
                save_path = f"all_pairs_kl_heatmap_no_loop.png"
        
        # 构建完整的保存路径
        full_save_path = os.path.join(self.output_dir, save_path)
        
        plt.figure(figsize=(12, 10))
        
        num_layers = kl_matrix.shape[0]
        
        # 创建相对层数的刻度标签
        relative_positions = np.linspace(0, 1, num_layers)
        relative_labels = [f'{pos:.2f}' for pos in relative_positions]
        
        # 创建热力图
        mask = kl_matrix == 0  # 遮盖没有数据的部分
        
        # 使用更好的颜色映射
        heatmap = sns.heatmap(kl_matrix, 
                             annot=False, 
                             fmt='.4f', 
                             cmap='YlOrRd',
                             square=True,
                             mask=mask,
                             cbar_kws={'label': 'KL Divergence'},
                             annot_kws={'size': 9},
                             linewidths=0.5,
                             linecolor='white',
                             xticklabels=relative_labels,
                             yticklabels=relative_labels)
        
        # 添加循环层标记线
        if self.config.loop_layers is not None:
            start_layer, end_layer = self.config.loop_layers
            real_end_layer = start_layer + self.config.loop_count * (end_layer - start_layer + 1) - 1
            
            # 在热力图中，坐标是从0开始的索引
            # 添加循环层的边界线
            plt.axhline(y=start_layer + 0.5, color='red', linestyle='--', linewidth=2, alpha=0.8)
            plt.axhline(y=real_end_layer + 1.5, color='red', linestyle='--', linewidth=2, alpha=0.8)
            plt.axvline(x=start_layer + 0.5, color='red', linestyle='--', linewidth=2, alpha=0.8)
            plt.axvline(x=real_end_layer + 1.5, color='red', linestyle='--', linewidth=2, alpha=0.8)
            
            # 添加循环区域的边框高亮
            # from matplotlib.patches import Rectangle
            # rect = Rectangle((start_layer + 0.5, start_layer + 0.5), 
            #                end_layer - start_layer + 1, 
            #                end_layer - start_layer + 1,
            #                linewidth=3, edgecolor='red', facecolor='none', alpha=0.8)
            # plt.gca().add_patch(rect)
            
            # 添加文本标注
            # plt.text(start_layer + 0.5, -0.5, f'Loop Start\n(Layer {start_layer})', 
            #         ha='center', va='top', color='red', fontweight='bold', fontsize=10)
            # if start_layer != end_layer:
            #     plt.text(end_layer + 0.5, -0.5, f'Loop End\n(Layer {end_layer})', 
            #             ha='center', va='top', color='red', fontweight='bold', fontsize=10)
        
        # 设置标题和标签
        title_suffix = ""
        if self.config.loop_layers is not None:
            start_layer, end_layer = self.config.loop_layers
            loop_count = self.config.loop_count
            title_suffix = f" (loop layers: {start_layer}-{end_layer}, loop count: {loop_count})"
        
        plt.xlabel('Relative Layer Position', fontsize=14)
        plt.ylabel('Relative Layer Position', fontsize=14)
        plt.title(f'KL Divergence Heatmap Between Layer Pairs{title_suffix}', fontsize=16, pad=20)
        
        # 设置刻度
        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.9)
        
        # 保存图片
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"热力图已保存至: {full_save_path}")
    
    @torch.no_grad()
    def run_analysis(self, num_samples: int = 50, batch_size: int = 2):
        """运行完整的可解释性分析
        
        Args:
            num_samples: 数据样本数量
        """
        print("开始LoopLLM可解释性分析...")
        print("=" * 50)
        
        # 1. 加载数据
        try:
            texts = self.load_data(num_samples)
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return None
        
        # 2. 准备输入 (使用更小的batch以节省内存)
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"将处理 {total_batches} 个批次，每批次 {batch_size} 个样本")
        
        # 用于累积所有batch的KL散度结果
        all_pairs_kl_results = {}  # 存储所有层对的KL散度
        num_layers = None
        
        try:
            # 使用tqdm显示批次处理进度
            batch_progress = tqdm(range(0, len(texts), batch_size), 
                                desc="Processing batches", 
                                unit="batch",
                                total=total_batches)
            
            for i in batch_progress:
                batch_texts = texts[i:i+batch_size]
                batch_num = i//batch_size + 1
                
                # 更新进度条描述
                batch_progress.set_description(f"Processing batch {batch_num}/{total_batches}")
                
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                
                # 检查是否有超出词汇表范围的token
                # if input_ids.max().item() >= self.config.vocab_size:
                #     print(f"⚠️ 警告: 发现超出词汇表范围的token!")
                #     print(f"  最大token id: {input_ids.max().item()}")
                #     print(f"  词汇表大小: {self.config.vocab_size}")
                
                # 3. 前向传播
                outputs = self.forward_with_hidden_states(input_ids)    # 一个长度为层数的列表，每个元素为[batch_size, seq_len, hidden_size]
                
                # 处理attention_mask，去除padding部分
                non_padding_mask = attention_mask.bool()  # 确保是bool类型
                
                # 对每一层的hidden_states分别处理
                filtered_hidden_states = []
                for layer_idx, layer_hidden in enumerate(outputs['hidden_states']):
                    # layer_hidden shape: [batch_size, seq_len, hidden_size]
                    # attention_mask shape: [batch_size, seq_len]
                    
                    # 获取非padding位置的hidden states
                    batch_size_cur, seq_len, hidden_size = layer_hidden.shape
                    
                    # 展平处理
                    layer_flat = layer_hidden.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
                    mask_flat = non_padding_mask.view(-1)  # [batch_size * seq_len]
                    
                    # 选择非padding的token
                    layer_filtered = layer_flat[mask_flat]  # [num_valid_tokens, hidden_size]
                    filtered_hidden_states.append(layer_filtered)
                
                # print(f'第一层过滤后的hidden_states形状: {filtered_hidden_states[0].shape}')
                
                if num_layers is None:
                    num_layers = len(filtered_hidden_states)
                    # print(f"模型共有 {num_layers} 层")
                
                # 4. 🔥 合并计算：计算所有层对的KL散度
                batch_pairs_kl = self.compute_all_pairs_kl(filtered_hidden_states)
                
                # 5. 累积结果（对每个层对取平均）
                for (layer_i, layer_j), kl_value in batch_pairs_kl.items():
                    if (layer_i, layer_j) not in all_pairs_kl_results:
                        all_pairs_kl_results[(layer_i, layer_j)] = []
                    all_pairs_kl_results[(layer_i, layer_j)].append(kl_value)
                
                # 内存清理
                del outputs, input_ids, attention_mask, filtered_hidden_states
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
                # 如果内存使用过高，中断处理
                # if memory_monitor() > 220:
                #     print("⚠️ 内存使用过高，停止处理更多批次")
                #     break
                
        except Exception as e:
            print(f"❌ 批次处理失败: {e}")
            if not all_pairs_kl_results:
                return None
        
        print("\n" + "=" * 50)
        print("开始汇总结果...")
        
        try:
            # 6. 汇总所有batch的KL散度结果（对每个层对取平均）
            final_pairs_kl = {}
            for (layer_i, layer_j), kl_values in all_pairs_kl_results.items():
                final_pairs_kl[(layer_i, layer_j)] = np.mean(kl_values)
            
            print(f"汇总了 {len(list(all_pairs_kl_results.values())[0])} 个batch的结果")
            
            # 7. 🔥 从完整的层对KL散度中提取每层与最后一层的KL散度
            layer_vs_final_kl = []
            final_layer_idx = num_layers - 1
            
            for layer_i in range(num_layers):
                if (layer_i, final_layer_idx) in final_pairs_kl:
                    kl_value = final_pairs_kl[(layer_i, final_layer_idx)]
                elif (final_layer_idx, layer_i) in final_pairs_kl:
                    # KL散度不对称，但如果只有反向的，也可以使用
                    kl_value = final_pairs_kl[(final_layer_idx, layer_i)]
                else:
                    kl_value = 0.0  # 如果没有计算到，设为0
                
                layer_vs_final_kl.append(kl_value)
            
            # 8. 构建完整的层对KL散度矩阵
            kl_matrix = np.zeros((num_layers, num_layers))
            for (i, j), kl_value in final_pairs_kl.items():
                kl_matrix[i, j] = kl_value
            
            print(f"✓ 提取到每层与最后一层的KL散度: {len(layer_vs_final_kl)} 个值")
            print(f"✓ 构建层对KL散度矩阵: {kl_matrix.shape}")
            
        except Exception as e:
            print(f"❌ 结果汇总失败: {e}")
            return None
        
        print("\n" + "=" * 50)
        print("开始可视化...")
        
        try:
            # 9. 生成可视化图表
            self.plot_layer_vs_final_kl(layer_vs_final_kl)
            if final_pairs_kl:
                self.plot_all_pairs_heatmap(kl_matrix)
        except Exception as e:
            print(f"⚠️ 可视化失败: {e}")
        
        print("\n" + "=" * 50)
        print("保存结果到JSON文件...")
        
        try:
            # 10. 保存KL散度结果到JSON文件
            results_dict = {
                'config': {
                    'loop_layers': self.config.loop_layers,
                    'loop_count': self.config.loop_count,
                    'num_layers': num_layers,
                    'num_samples': num_samples,
                    'batch_size': batch_size
                },
                'layer_vs_final_kl': layer_vs_final_kl,
                'all_pairs_kl_matrix': kl_matrix.tolist(),  # 转换numpy数组为列表
            }
            
            # 生成JSON文件名
            if self.config.loop_layers is not None:
                start_layer, end_layer = self.config.loop_layers
                json_filename = f"all_pairs_kl_{start_layer}_{end_layer}_{self.config.loop_count}.json"
            else:
                json_filename = f"all_pairs_kl_no_loop.json"
            
            json_filepath = os.path.join(self.output_dir, json_filename)
            
            # 保存到JSON文件
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
                
            print(f"✓ KL散度结果已保存到: {json_filepath}")
            
        except Exception as e:
            print(f"⚠️ 保存JSON文件失败: {e}")
        
        print("\n" + "=" * 50)
        print("分析完成!")
        
        return {
            'layer_vs_final_kl': layer_vs_final_kl,
            'all_pairs_kl': kl_matrix,
            'pairs_kl_dict': final_pairs_kl  # 额外返回字典形式的结果
        }
    
    def compute_all_pairs_kl(self, hidden_states: List[torch.Tensor]) -> Dict[tuple, float]:
        """计算所有层对的KL散度
        
        Args:
            hidden_states: 所有层的隐状态列表
            
        Returns:
            层对KL散度字典
        """
        # print("正在计算所有层对的KL散度...")
        
        # 计算所有层的概率分布
        probabilities = self.compute_layer_probabilities(hidden_states)
        
        num_layers = len(probabilities)
        kl_results = {}
        
        # 随机采样位置以减少计算量
        num_tokens = probabilities[0].shape[0]
        # sample_size = min(50, num_tokens)  # 采样数量
        sample_size = num_tokens
        indices = torch.randperm(num_tokens)[:sample_size]
        
        # print(f"从 {num_tokens} 个token中采样 {sample_size} 个进行KL散度计算")
        
        for i in range(num_layers):
            for j in range(num_layers):
                if i == j:
                    kl_results[(i, j)] = 0.0
                else:
                    # 对采样位置求平均KL散度
                    batch_kl = []
                    
                    for idx in indices:
                        try:
                            # 计算 KL(layer_i || layer_j)
                            # probabilities[i][idx] 作为参考分布P，probabilities[j][idx] 作为目标分布Q
                            kl = self.compute_kl_divergence(
                                probabilities[i][idx],  # P: 参考分布（层i）
                                probabilities[j][idx]   # Q: 目标分布（层j）
                            )
                            batch_kl.append(kl.item() if isinstance(kl, torch.Tensor) else kl)
                        except Exception as e:
                            continue
                    
                    avg_kl = np.mean(batch_kl) if batch_kl else 0.0
                    kl_results[(i, j)] = avg_kl
        
        # print(f"所有层对KL散度计算完成，共 {len(kl_results)} 个层对")
        return kl_results


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig = None):
    """主函数"""
    print("LoopLLM可解释性分析工具")
    print("=" * 50)
    

    model_path = cfg.model.model_path

    llama_config = LlamaConfig.from_pretrained(model_path)
    config_dict = llama_config.to_dict()

    loop_layers = cfg.loop.loop_layer
    if isinstance(loop_layers, int):
        loop_layers = [loop_layers]
    elif isinstance(loop_layers, str):
        if loop_layers == "all":
            loop_layers = list(range(0, llama_config.num_hidden_layers))
        else:
            loop_layers = None

    if loop_layers is None:
        loop_llama_config = LoopLlamaConfig(
            # LoopLLM特定配置
            loop_layers=None, 
            loop_strategy="fixed_count",
            loop_count=cfg.loop.loop_count,  # 从配置中获取循环次数
            kv_cache_mode="virtual_layers",
            virtual_layer_count=cfg.loop.loop_count,  # 虚拟层数量与循环次数相同
            min_loop_count=cfg.loop.loop_count,  # 最小循环次数与循环次数相同
            max_loop_count=cfg.loop.loop_count,  # 最大循环次数
            **config_dict
        )           
        
        # 创建分析器，指定输出目录
        interpreter = LoopLLMInterpreter(loop_llama_config, model_path=model_path, output_dir=cfg.base.output_dir)
        
        # 运行分析 - 使用配置中的参数
        results = interpreter.run_analysis(num_samples=cfg.base.num_samples, batch_size=cfg.base.batch_size)
    else:
        print(f"需要循环的层: {loop_layers}, 共{len(loop_layers)}层")
        print(f"循环次数: {cfg.loop.loop_count}")
        
        for loop_layer in loop_layers:
            print(f"正在分析第{loop_layer}层")

            loop_llama_config = LoopLlamaConfig(
                # LoopLLM特定配置
                loop_layers=[(loop_layer, loop_layer)], 
                loop_strategy="fixed_count",
                loop_count=cfg.loop.loop_count,  # 从配置中获取循环次数
                kv_cache_mode="virtual_layers",
                virtual_layer_count=cfg.loop.loop_count,  # 虚拟层数量与循环次数相同
                min_loop_count=cfg.loop.loop_count,  # 最小循环次数与循环次数相同
                max_loop_count=cfg.loop.loop_count,  # 最大循环次数
                **config_dict
            )
            
            # print("模型配置:")
            # print(f"  词汇表大小: {loop_llama_config.vocab_size}")
            # print(f"  隐藏层大小: {loop_llama_config.hidden_size}")
            # print(f"  层数: {loop_llama_config.num_hidden_layers}")
            # print(f"  循环层: {loop_llama_config.loop_layers}")
            # print(f"  循环次数: {loop_llama_config.loop_count}")
                
            
            # 创建分析器，指定输出目录
            interpreter = LoopLLMInterpreter(loop_llama_config, model_path=model_path, output_dir=cfg.base.output_dir)
            
            # 运行分析 - 使用配置中的参数
            results = interpreter.run_analysis(num_samples=cfg.base.num_samples, batch_size=cfg.base.batch_size)
            
            # if results:
            #     print("\n分析结果摘要:")
            #     print(f"每层与最后一层的KL散度: {results['layer_vs_final_kl']}")
            #     print(f"KL散度矩阵形状: {results['all_pairs_kl'].shape}")
            # else:
            #     print("❌ 分析失败")


if __name__ == "__main__":
    main() 