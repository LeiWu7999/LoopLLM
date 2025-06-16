import torch
import numpy as np
import math
import pandas as pd
from loop_llama_config import LoopLlamaConfig
from loop_llama_model import LoopLlamaForCausalLM
from transformers import AutoTokenizer
from transformers import LlamaConfig


def get_layer_mean_hidden(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
    # 修复bug：创建独立的tensor对象而不是多个引用
    layer_sum_list = [torch.zeros_like(outputs.hidden_states[1][0]) for _ in range(len(outputs.hidden_states[0]))]
    
    for i in range(1, len(outputs.hidden_states)):
        for j in range(len(outputs.hidden_states[i])):
            layer_sum_list[j] += outputs.hidden_states[i][j]
    layer_sum_tensor = torch.stack(layer_sum_list, dim=0)
    # 删除中间的两个维度为1的维度，从 [27, 1, 1, 2048] 变为 [27, 2048]
    layer_sum_squeezed = layer_sum_tensor.squeeze()
    # 沿着27的维度，对每一行都除以99
    layer_mean_squeezed = layer_sum_squeezed / (len(outputs.hidden_states) - 1)
    
    
    return layer_mean_squeezed


class CoEScoreInfo:
    def __init__(self, hidden_states):
        """
        初始化CoEScoreInfo类
        
        Args:
            hidden_states: 可以是以下两种格式之一:
                1. 原始格式: 列表形式的hidden states
                2. layer_mean_squeezed: torch.Tensor或numpy.array，形状为[layer_num, hidden_dim]
        """
        # 如果输入是tensor或numpy数组，且是2D的，认为是layer_mean_squeezed格式
        if isinstance(hidden_states, (torch.Tensor, np.ndarray)) and len(hidden_states.shape) == 2:
            # 转换为numpy数组
            if isinstance(hidden_states, torch.Tensor):
                self.hidden_states = hidden_states.cpu().numpy()
            else:
                self.hidden_states = hidden_states
        else:
            # 原始格式，直接赋值
            self.hidden_states = hidden_states

    def compute_CoE_Mag(self):
        hs_all_layer = self.hidden_states
        layer_num = len(hs_all_layer)

        norm_denominator = np.linalg.norm(hs_all_layer[-1] - hs_all_layer[0], ord=2)
        al_repdiff = np.array([hs_all_layer[i+1] - hs_all_layer[i] for i in range(layer_num - 1)])
        al_repdiff_norm = [np.linalg.norm(item, ord=2) / norm_denominator for item in al_repdiff]
        al_repdiff_ave = np.mean(np.array(al_repdiff_norm))
        al_repdiff_var = np.var(np.array(al_repdiff_norm))
        return al_repdiff_norm, al_repdiff_ave, al_repdiff_var


    def compute_CoE_Ang(self):
        hs_all_layer = self.hidden_states
        layer_num = len(hs_all_layer)

        al_semdiff = []
        norm_denominator = np.dot(hs_all_layer[-1], hs_all_layer[0]) / (np.linalg.norm(hs_all_layer[-1], ord=2) * np.linalg.norm(hs_all_layer[0], ord=2))
        norm_denominator = math.acos(norm_denominator)
        for i in range(layer_num - 1):
            a = hs_all_layer[i + 1]
            b = hs_all_layer[i]
            dot_product = np.dot(a, b)
            norm_a, norm_b = np.linalg.norm(a, ord=2), np.linalg.norm(b, ord=2)
            similarity = dot_product / (norm_a * norm_b)
            similarity = similarity if similarity <= 1 else 1

            arccos_sim = math.acos(similarity)
            al_semdiff.append(arccos_sim / norm_denominator)

        al_semdiff_norm = np.array(al_semdiff)
        al_semdiff_ave = np.mean(np.array(al_semdiff_norm))
        al_semdiff_var = np.var(np.array(al_semdiff_norm))
        
        return al_semdiff_norm, al_semdiff_ave, al_semdiff_var

    def compute_CoE_R(self):
        _, al_repdiff_ave, _ = self.compute_CoE_Mag()
        _, al_semdiff_ave, _ = self.compute_CoE_Ang()

        return al_repdiff_ave - al_semdiff_ave

    def compute_CoE_C(self):
        al_repdiff_norm, _, _ = self.compute_CoE_Mag()
        al_semdiff_norm, _, _ = self.compute_CoE_Ang()
        x_list = np.array([al_repdiff_norm[i] * math.cos(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
        y_list = np.array([al_repdiff_norm[i] * math.sin(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
        al_combdiff_x_ave = np.mean(x_list)
        al_combdiff_y_ave = np.mean(y_list)
        al_combdiff_x_var = np.mean(x_list)
        al_combdiff_y_var = np.mean(y_list)

        return math.sqrt(al_combdiff_x_ave ** 2 + al_combdiff_y_ave ** 2)


if __name__ == "__main__":
    # 加载数据集
    dataset = "math_train_12k"
    math_data = pd.read_csv(f'/home/ubuntu/Documents/newdisk_22T/lzq/LoopLLM_old/mi_EXP/math_train_12k.csv')
    question_list = math_data['problem'].tolist()[:100]
    # 加载模型
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    llama_config = LlamaConfig.from_pretrained(model_path)
    config_dict = llama_config.to_dict()
    loop_llama_config = LoopLlamaConfig(
        # LoopLLM特定配置
        loop_layers=[(4,5),(6,7),(8,9)], 
        loop_strategy="fixed_count",
        loop_count=[1,1,1],  # 从配置中获取循环次数
        kv_cache_mode="virtual_layers",
        virtual_layer_count=[1,1,1],  # 虚拟层数量与循环次数相同
        virtual_attention_mode="parallel",
        **config_dict
    )           
    model = LoopLlamaForCausalLM.from_pretrained(model_path, config=loop_llama_config, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    for text in question_list:
        print(text)
        layer_mean_squeezed = get_layer_mean_hidden(model, tokenizer, text)
        # 使用CoEScoreInfo类
        coe_score_info = CoEScoreInfo(layer_mean_squeezed)
        coe_r = coe_score_info.compute_CoE_R()
        coe_c = coe_score_info.compute_CoE_C()
        print(f"CoE_R: {coe_r:.4f}")
        print(f"CoE_C: {coe_c:.4f}")
        print("-" * 50)
        break