# 测试训练脚本，构建虚拟数据集测试训练函数是否正常
import sys
import os
import json
import shutil
import tempfile
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, Sequence
import torch
import argparse

# 将父目录添加到路径中，以便从其他模块导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CPT_accelerate_simple import CPT_train, load_config
from loop_llama_model import LoopLlamaForCausalLM
from loop_llama_config import LoopLlamaConfig
from transformers import AutoTokenizer, LlamaConfig

def create_dummy_dataset(output_path, num_train_samples=900, num_val_samples=100, seq_length=8192, vocab_size=1001):
    """
    创建并保存一个用于测试的虚拟数据集。
    该数据集模仿了 tokenize_and_pack.py 中打包数据的结构。
    """
    print(f"正在于 {output_path} 创建虚拟数据集...")
    
    def generate_data(num_samples):
        # 仿照 tokenize_and_pack.py 的输出结构
        data = {"input_ids": [], "document_ids": []}
        for i in range(num_samples):
            # 生成范围为 [0, vocab_size-1] 的随机整数 input_ids
            input_ids = np.random.randint(0, vocab_size, size=seq_length).tolist()
            data["input_ids"].append(input_ids)
            
            # 每个样本包含两个文档ID，在中间分割
            doc_split_point = seq_length // 2
            doc_ids = [i*2] * doc_split_point + [i*2+1] * (seq_length - doc_split_point)
            data["document_ids"].append(doc_ids)
        return data

    # 定义数据集的特征以匹配预期的格式
    features = Features({
        'input_ids': Sequence(feature=Value(dtype='int32')),
        'document_ids': Sequence(feature=Value(dtype='int32'))
    })

    # 创建训练和验证分割
    train_data = generate_data(num_train_samples)
    val_data = generate_data(num_val_samples)
    
    train_dataset = Dataset.from_dict(train_data, features=features)
    validation_dataset = Dataset.from_dict(val_data, features=features)
    
    # 创建DatasetDict并保存到磁盘
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset
    })
    
    # 如果目录已存在，先删除
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        
    dataset_dict.save_to_disk(output_path)
    print("虚拟数据集创建成功。")
    return output_path

def run_test(config_path):
    """
    运行测试的主函数。
    """
    # 为虚拟数据和模型输出使用一个临时目录
    base_temp_dir = tempfile.mkdtemp()
    temp_data_dir = os.path.join(base_temp_dir, "test_data")
    temp_output_dir_template = os.path.join(base_temp_dir, "test_output_f{}_{}")

    try:
        # 1. 加载JSON配置文件
        print(f"从 {config_path} 加载配置...")
        training_config = load_config(config_path)

        # 2. 创建虚拟数据集
        # vocab_size 应该大于等于1001以匹配请求
        # 我们使用模型词汇表大小以获得更真实的测试
        tokenizer_temp = AutoTokenizer.from_pretrained(training_config['model_config']['model_name_or_path'])
        vocab_size = tokenizer_temp.vocab_size
        del tokenizer_temp
        
        # 将总共1000条数据（900训练，100验证）写入磁盘
        create_dummy_dataset(
            temp_data_dir, 
            num_train_samples=900, 
            num_val_samples=1000, 
            seq_length=training_config['data_params']['max_length'], 
            vocab_size=vocab_size
        )

        # 3. 修改配置以进行快速测试
        # 将数据路径覆盖为我们的虚拟数据路径
        training_config['data_params']['data_name_or_path'] = temp_data_dir
        # 将输出目录重定向到临时文件夹
        training_config['training_params']['output_dir_template'] = temp_output_dir_template
        # 对于快速测试，禁用wandb报告
        training_config['training_params']['report_to'] = "none"
        # 禁用PPL评估，因为它需要完整的文本并可能减慢测试速度
        training_config['ppl_eval_config']['enabled'] = False
        
        print("配置已加载并为测试进行了修改。")
        
        loop_conf = training_config['loop_config']
        if not loop_conf['use_loop']:
            print("Warning: Loop is disabled in the configuration. Training will be performed without loop.")
            loop_conf['loop_layers'] = None
            loop_conf['loop_count'] = None
        model_conf = training_config['model_config']
        data_conf = training_config['data_params']

        # 4. 加载模型和分词器
        model_name_or_path = model_conf['model_name_or_path']
        print(f"从以下位置加载基础模型和分词器: {model_name_or_path}")

        llama_config = LlamaConfig.from_pretrained(model_name_or_path)
        config_dict = llama_config.to_dict()
        
        config = LoopLlamaConfig(
            **loop_conf,
            **config_dict
        )
        
        loop_llama_model = LoopLlamaForCausalLM.from_pretrained(model_name_or_path, config=config, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        loop_llama_model.model.training_mode = True

        # 5. 加载虚拟数据集
        print(f"从以下位置加载虚拟数据集: {data_conf['data_name_or_path']}")
        dataset = DatasetDict.load_from_disk(data_conf['data_name_or_path'])
        
        print("数据集已加载。分割:", list(dataset.keys()))
        print("训练样本数:", len(dataset['train']))
        print("验证样本数:", len(dataset['validation']))

        # 6. 开始训练
        print("开始测试训练...")
        # 为了使测试快速完成，我们只训练几个步骤
        # 这通过命令行参数或直接修改training_args来完成
        # 这里我们直接调用CPT_train，它内部会处理
        trainer = CPT_train(loop_llama_model, dataset, tokenizer, training_config)
        print("测试训练成功完成。")

    finally:
        # 清理虚拟数据和输出目录
        print(f"清理临时目录: {base_temp_dir}")
        shutil.rmtree(base_temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for CPT_accelerate_simple.py")
    parser.add_argument(
        "--config", 
        type=str, 
        default="./train_config.json",
        help="Path to the training configuration file."
    )
    args = parser.parse_args()
    
    run_test(args.config) 