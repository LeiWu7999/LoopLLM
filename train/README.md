# 数据集设置
```python
data_name_or_path = "openai/gsm8k" # 数据集的路径或者名称
dataset = loading_dataset(data_name_or_path,text_column_name="question") 
# 在text_column_name中传入我们要训练的字段
```
# DeepSpeed设置
运行命令配置accelerate，选择开启DeepSpeed-Zero（1-3）都可以
```python
accelerate config
```
# 启动训练
cd 到项目目录LoopLLM，执行以下命令
```python
accelerate launch CPT_accelerate_simple.py
```
# 启动TensorBoard
```
tensorboard --logdir="tensorboard文件路径" --port=6006
```