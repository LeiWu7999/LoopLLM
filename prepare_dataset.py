import json
from datasets import load_dataset

def prepare_pile_dataset(num_samples=10):
    """
    读取EleutherAI/the_pile_deduplicated数据集并保存前10条到json文件
    """
    print("正在加载数据集...")
    
    # 加载数据集
    dataset = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)
    
    sample_data = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        sample_data.append(item)
        print(f"已获取第 {i+1} 条数据")
    
    # 保存到json文件
    output_file = f"pile_sample.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"成功保存前{num_samples}条数据到 {output_file}")
    print(f"总共保存了 {len(sample_data)} 条数据")

if __name__ == "__main__":
    prepare_pile_dataset(num_samples=50)
