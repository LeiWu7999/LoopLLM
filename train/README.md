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
`--logdir` 参数应该指向包含 event 文件的目录，而不是具体的文件。

例如，如果您的日志在 `logs_20250618_00h32m05s` 目录中，您应该使用该目录作为路径。
```bash
tensorboard --logdir="logs_20250618_00h32m05s" --port=6006
```