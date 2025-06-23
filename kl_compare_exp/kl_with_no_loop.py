import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import json
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, LlamaConfig
import warnings

from loop_llama_config import LoopLlamaConfig
from loop_llama_model import LoopLlamaForCausalLM

warnings.filterwarnings('ignore')

class LoopComparer:
    """
    用于比较循环模型和非循环模型的隐状态差异的分析器。
    """
    def __init__(self, cfg: DictConfig):
        """
        初始化分析器。
        Args:
            cfg: Hydra配置对象。
        """
        self.cfg = cfg
        self.model_path = cfg.model.model_path
        self.output_dir = "comparison_results"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"输出目录: {self.output_dir}")

    def load_data(self, num_samples: int = 50) -> list[str]:
        """
        从 'pile_sample.json' 加载数据集。
        """
        local_file = "pile_sample.json"
        print(f"正在从 {local_file} 加载数据...")
        if not os.path.exists(local_file):
            raise FileNotFoundError(f"未找到数据集文件: {local_file}。请先运行 prepare_dataset.py 生成。")
        with open(local_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [item['text'] for item in data]
        return texts[:num_samples]

    def get_model_and_tokenizer(self, loop_config_kwargs: dict) -> tuple[LoopLlamaForCausalLM, AutoTokenizer]:
        """
        根据给定的循环配置初始化模型和分词器。
        """
        llama_config = LlamaConfig.from_pretrained(self.model_path)
        config_dict = llama_config.to_dict()
        
        loop_llama_config = LoopLlamaConfig(**config_dict, **loop_config_kwargs)
        
        model = LoopLlamaForCausalLM.from_pretrained(
            self.model_path, 
            config=loop_llama_config, 
            device_map="auto"
        )
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer

    @torch.no_grad()
    def _get_hidden_states_for_text(self, model, tokenizer, text):
        """Helper to get all layer hidden states for a single text sample."""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        outputs = model(input_ids=inputs.input_ids, output_hidden_states=True)
        # Return list of tensors [seq_len, hidden_size] for each layer, on CPU
        return [s[0].cpu() for s in outputs.hidden_states]      # 去掉batch

    @torch.no_grad()
    def get_baseline_outputs(self):
        """
        获取或生成基线模型（无循环）的输出。
        按文本处理，并为每个文本保存其平均隐状态和稀疏表示的最终层概率。
        """
        cache_file = os.path.join(self.output_dir, "baseline_outputs.pt")
        if os.path.exists(cache_file):
            print(f"从缓存加载基线模型输出: {cache_file}")
            return torch.load(cache_file, map_location="cpu")

        print("运行基线模型 (无循环)...")
        # 1. 配置无循环模型
        model, tokenizer = self.get_model_and_tokenizer(loop_config_kwargs={'loop_layers': None})
        
        # 2. 加载数据
        texts = self.load_data(self.cfg.base.num_samples)
        
        all_outputs = []
        for text in tqdm(texts, desc="处理基线数据"):
            hidden_states = self._get_hidden_states_for_text(model, tokenizer, text)
            
            # 6. 计算每层隐状态的平均值
            all_hidden_states = [h.reshape(-1, h.size(-1)) for h in hidden_states]
            
            all_outputs.append({
                "all_hidden_states": all_hidden_states, # 保存所有层的隐状态
            })
        
        print(f"保存基线模型输出到: {cache_file}")
        torch.save(all_outputs, cache_file)
        
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return all_outputs

    def plot_cosine_similarity(self, similarities, baseline_layers, looped_layers, loop_layer_idx, loop_count):
        """绘制余弦相似度图"""
        plt.figure(figsize=(15, 7))
        plt.plot(looped_layers, similarities, marker='o', linestyle='-')
        
        # 标记循环区域
        loop_start_layer_idx = loop_layer_idx
        loop_end_layer_idx = loop_layer_idx + loop_count - 1
        plt.axvspan(loop_start_layer_idx - 0.5, loop_end_layer_idx + 0.5, color='red', alpha=0.15)

        plt.title(f'Cosine Similarity (Looped Layer {loop_layer_idx}, Count {loop_count})', fontsize=16)
        plt.xlabel('Looped Model Layer Index', fontsize=12)
        plt.ylabel('Cosine Similarity', fontsize=12)
        plt.xticks(np.arange(0, len(looped_layers), 2))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"cosine_sim_loop_{loop_layer_idx}_{loop_count}.png")
        plt.savefig(save_path)
        plt.close() # 关闭图像，防止在notebook中重复显示
        print(f"余弦相似度图已保存至: {save_path}")

    def plot_kl_divergence(self, kl_divergences, looped_layers, loop_layer_idx, loop_count):
        """绘制KL散度图"""
        plt.figure(figsize=(15, 7))
        plt.plot(looped_layers, kl_divergences, marker='s', linestyle='--', color='green')
        
        # 标记循环区域
        loop_start_layer_idx = loop_layer_idx
        loop_end_layer_idx = loop_layer_idx + loop_count - 1
        plt.axvspan(loop_start_layer_idx - 0.5, loop_end_layer_idx + 0.5, color='red', alpha=0.15, label=f'Looped Region (Layer {loop_layer_idx})')

        plt.title(f'KL Divergence (Looped Layer {loop_layer_idx}, Count {loop_count})', fontsize=16)
        plt.xlabel('Looped Model Layer Index', fontsize=12)
        plt.ylabel('KL Divergence (vs Baseline Final Layer)', fontsize=12)
        plt.xticks(np.arange(0, len(looped_layers), 2))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"kl_div_loop_{loop_layer_idx}_{loop_count}.png")
        plt.savefig(save_path)
        plt.close() # 关闭图像，防止在notebook中重复显示
        print(f"KL散度图已保存至: {save_path}")

    def run(self):
        """
        执行完整的比较分析流程。
        """
        # 1. 获取基线模型的输出 (从缓存或新运行)
        baseline_outputs_per_text = self.get_baseline_outputs()

        # 【新逻辑】加载基线模型一次，用于即时计算所有最终层的概率
        print("正在从基线隐状态即时计算概率分布...")
        baseline_model, _ = self.get_model_and_tokenizer(loop_config_kwargs={'loop_layers': None})
        baseline_final_probs_per_text = []
        for baseline_output in tqdm(baseline_outputs_per_text, desc="计算基线概率"):
            final_hidden_state = baseline_output['all_hidden_states'][-1].to(baseline_model.device)
            norm_h = baseline_model.model.norm(final_hidden_state)
            logits = baseline_model.lm_head(norm_h)
            probs = F.softmax(logits, dim=-1).cpu()
            baseline_final_probs_per_text.append(probs)
        del baseline_model
        torch.cuda.empty_cache()
        print("✓ 基线概率计算完成。")
        
        # 2. 遍历需要测试的循环层
        loop_layers = self.cfg.loop.loop_layer
        if isinstance(loop_layers, int):
            loop_layers = [loop_layers]
        elif isinstance(loop_layers, str):
            if loop_layers == "all":
                loop_layers = list(range(0, 16))
            else:
                raise ValueError(f"Invalid loop_layers: {loop_layers}")
        loop_count = self.cfg.loop.loop_count

        for loop_layer_idx in loop_layers:
            print("\n" + "="*50)
            print(f"正在分析循环层: {loop_layer_idx}, 循环次数: {loop_count}")
            
            # 3. 加载循环模型
            loop_kwargs = {
                'loop_layers': [(loop_layer_idx, loop_layer_idx)],
                'loop_strategy': "fixed_count",
                'loop_count': loop_count,
                'kv_cache_mode': "virtual_layers",
                'virtual_layer_count': loop_count,
                'min_loop_count': loop_count,
                'max_loop_count': loop_count,
            }
            looped_model, tokenizer = self.get_model_and_tokenizer(loop_kwargs)
            texts = self.load_data(self.cfg.base.num_samples)

            # 通过一次前向传播确定模型结构
            _dummy_states = self._get_hidden_states_for_text(looped_model, tokenizer, texts[0])
            num_looped_layers = len(_dummy_states) - 1
            del _dummy_states
            torch.cuda.empty_cache()

            # 初始化指标累加器
            total_similarities = [[] for _ in range(num_looped_layers)]
            total_kls = [[] for _ in range(num_looped_layers)]

            # 4. 逐条文本进行分析
            for i, text in enumerate(tqdm(texts, desc=f"分析循环层 {loop_layer_idx}")):
                baseline_output = baseline_outputs_per_text[i]
                baseline_all_states = baseline_output['all_hidden_states']
                baseline_final_probs = baseline_final_probs_per_text[i]

                looped_hidden_states = self._get_hidden_states_for_text(looped_model, tokenizer, text)
                
                # --- 4.1 计算余弦相似度 ---
                looped_all_states = [h.reshape(-1, h.size(-1)) for h in looped_hidden_states]
                for j in range(num_looped_layers):
                    looped_state = looped_all_states[j + 1] # +1 to skip embedding
                    
                    # 确定与基线模型中哪一层进行比较
                    if j < loop_layer_idx:
                        baseline_state = baseline_all_states[j + 1]
                    elif loop_layer_idx <= j < loop_layer_idx + loop_count:
                        baseline_state = baseline_all_states[loop_layer_idx + 1]
                    else: # j >= loop_layer_idx + loop_count
                        baseline_idx = j - loop_count + 1
                        baseline_state = baseline_all_states[baseline_idx + 1]
                    
                    sim = F.cosine_similarity(looped_state.unsqueeze(0), baseline_state.unsqueeze(0), dim=-1).mean()
                    total_similarities[j].append(sim.item())
            
                # --- 4.2 计算KL散度 ---
                for j in range(num_looped_layers):
                    current_hidden_state = looped_hidden_states[j + 1].to(looped_model.device)
                    norm_h = looped_model.model.norm(current_hidden_state)
                    logits = looped_model.lm_head(norm_h)
                    looped_probs = F.softmax(logits, dim=-1)

                    # 【新逻辑】直接使用即时计算的稠密概率张量进行比较
                    kl_vals_for_layer = []
                    
                    # 确定要比较的token数量
                    num_tokens_to_compare = min(looped_probs.shape[0], baseline_final_probs.shape[0])

                    # 逐token计算KL散度
                    for k in range(num_tokens_to_compare):
                        p_k = looped_probs[k]
                        q_k = baseline_final_probs[k].to(p_k.device) # 确保在同一设备
                        
                        # 为避免log(0)问题，添加一个极小值
                        p_k_ = p_k + 1e-10
                        q_k_ = q_k + 1e-10
                        
                        kl_val = torch.sum(p_k_ * (p_k_.log() - q_k_.log()))
                        kl_vals_for_layer.append(kl_val.item())
                    
                    if kl_vals_for_layer:
                        total_kls[j].append(np.mean(kl_vals_for_layer))
            # --- 5. 平均并绘图 ---
            avg_similarities = [np.mean(s) if s else 0 for s in total_similarities]
            avg_kls = [np.mean(kl) if kl else 0 for kl in total_kls]
            
            num_baseline_layers = len(baseline_outputs_per_text[0]['all_hidden_states']) - 1
            
            self.plot_cosine_similarity(avg_similarities, list(range(num_baseline_layers)), list(range(num_looped_layers)), loop_layer_idx, loop_count)
            self.plot_kl_divergence(avg_kls, list(range(num_looped_layers)), loop_layer_idx, loop_count)

            del looped_model, tokenizer
            torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig = None):
    """主函数"""
    print("LoopLLM 与基线模型对比分析工具")
    print("=" * 50)
    comparer = LoopComparer(cfg)
    comparer.run()


if __name__ == "__main__":
    main()
