#!/bin/bash

# 脚本用于在多个GPU上并行运行 LoopLLM/run.py, 每个GPU测试不同的循环层块和模型。
# 支持为每个GPU指定独立的循环起止层和模型路径，以实现更灵活的并行测试。
# GPU数量自动根据 LOOP_CONFIGS 数组长度调整（最多8个GPU）。

# --- 配置 ---
# 定义要并行测试的实验列表。
# 每个元素是一个字符串 "START_LAYER,END_LAYER,MODEL_PATH"，代表一个独立的实验配置。
# 每个配置将在一个单独的GPU上运行。
#
# 示例:
LOOP_CONFIGS=(
    "6,8,/root/loop_llm_exp/ckpts/checkpoint-100"
    "6,8,/root/loop_llm_exp/ckpts/checkpoint-200"
)
#
# 说明:
# - 上述配置将在 GPU 0 上使用 checkpoint-100 测试循环层 [6, 8]。
# - 在 GPU 1 上使用 checkpoint-200 测试循环层 [6, 8]。
# - 您可以根据需要添加更多行来利用更多GPU，或修改层和模型路径。


# 检查 LOOP_CONFIGS 数组是否包含至少1个元素
if [ ${#LOOP_CONFIGS[@]} -eq 0 ]; then
    echo "错误: LOOP_CONFIGS 数组不能为空。"
    exit 1
elif [ ${#LOOP_CONFIGS[@]} -gt 8 ]; then
    echo "警告: LOOP_CONFIGS 数组包含 ${#LOOP_CONFIGS[@]} 个元素，但只有8个GPU可用。将只使用前8个配置。"
    LOOP_CONFIGS=(${LOOP_CONFIGS[@]:0:8})  # 截取前8个元素
fi

echo "将使用 ${#LOOP_CONFIGS[@]} 个GPU进行测试"

# run.py 脚本的相对路径 (假设此bash脚本在 loop_llm_exp 目录下运行)
PYTHON_SCRIPT_PATH="./run.py"

# Hydra 配置文件的名称 (不带.yaml扩展名, 位于 LoopLLM/ppl_nll_exp/config/ 目录下)
HYDRA_CONFIG_NAME="config"

# --- 前置检查 ---
echo "执行前置检查..."
# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "错误: Python 脚本未找到于 '$PYTHON_SCRIPT_PATH'"
    echo "请确认相对于脚本运行目录 (当前为 $(pwd)) 的路径是否正确。"
    exit 1
else
    echo "Python 脚本找到: $PYTHON_SCRIPT_PATH"
fi

HYDRA_CONFIG_DIR="./config"
echo "前置检查通过。"
# --- 检查结束 ---

# 用于存储此批量运行日志的目录
LOG_DIR="parallel_run_logs"
mkdir -p "$LOG_DIR"

# Hydra 的默认输出目录基础路径 (run.py 内部逻辑会在此基础上创建时间戳和自定义子目录)
# 此处我们不直接控制Hydra的输出目录，仅记录脚本本身的输出。
# run.py中设置了 hydra.job.chdir=false, 所以run.py的 CWD 是此脚本的运行目录。

echo "准备在${#LOOP_CONFIGS[@]}个GPU上并行启动实验..."
echo "每个GPU的测试配置 (START_LAYER,END_LAYER,MODEL_PATH) 如下:"
for (( i=0; i<${#LOOP_CONFIGS[@]}; i++ )); do
    START_LAYER_TMP=$(echo ${LOOP_CONFIGS[i]} | cut -d',' -f1)
    END_LAYER_TMP=$(echo ${LOOP_CONFIGS[i]} | cut -d',' -f2)
    MODEL_PATH_TMP=$(echo ${LOOP_CONFIGS[i]} | cut -d',' -f3-)
    MODEL_NAME_TMP=$(basename "$MODEL_PATH_TMP")
    echo "  GPU $i: 模型 ${MODEL_NAME_TMP}, 测试循环层块 [${START_LAYER_TMP}, ${END_LAYER_TMP}]"
done
echo "日志文件将保存在: $LOG_DIR"
echo "Hydra的输出将保存在 'model_comparison_results/' 目录下，按模型和层配置分子目录。"
echo "-----------------------------------------------------"

# 循环遍历 GPU ID，根据 LOOP_CONFIGS 数组的实际长度
for (( GPU_INDEX=0; GPU_INDEX<${#LOOP_CONFIGS[@]}; GPU_INDEX++ ))
do
    # 获取当前GPU要测试的配置
    CURRENT_CONFIG=${LOOP_CONFIGS[$GPU_INDEX]}
    
    # 解析 "START,END,MODEL_PATH" 格式的配置
    START_LAYER=$(echo $CURRENT_CONFIG | cut -d',' -f1)
    END_LAYER=$(echo $CURRENT_CONFIG | cut -d',' -f2)
    MODEL_PATH=$(echo $CURRENT_CONFIG | cut -d',' -f3-)
    
    # 提取模型名称用于创建目录
    MODEL_NAME=$(basename "$MODEL_PATH")
    
    # 定义并创建特定于此运行的Hydra输出目录，以避免结果覆盖
    HYDRA_RUN_DIR="model_comparison_results/${MODEL_NAME}/run_L${START_LAYER}-L${END_LAYER}"
    mkdir -p "$HYDRA_RUN_DIR"
    
    # 定义此任务的日志文件路径
    LOG_FILE="$LOG_DIR/gpu${GPU_INDEX}_model_${MODEL_NAME}_layers${START_LAYER}-${END_LAYER}.log"

    echo "启动任务: GPU ${GPU_INDEX} - 模型 ${MODEL_NAME}, 块 [${START_LAYER}, ${END_LAYER}] -> 输出: ${HYDRA_RUN_DIR}, 日志: ${LOG_FILE}"

    # 构建 Hydra 的参数覆盖
    MODEL_PATH_OVERRIDE="model.base_model_name_or_path=${MODEL_PATH}"
    LOOP_LAYERS_IDX_OVERRIDE="model.loop_layers_idx=[${START_LAYER},${END_LAYER}]"
    HYDRA_OUTPUT_OVERRIDE="hydra.run.dir=${HYDRA_RUN_DIR}"

    # 设置 CUDA_VISIBLE_DEVICES 以便 python 脚本只看到一个GPU

    # 运行 python 脚本命令，并在后台执行 (&)
    # 所有输出 (stdout 和 stderr) 都重定向到日志文件
    CUDA_VISIBLE_DEVICES=$GPU_INDEX python $PYTHON_SCRIPT_PATH \
        --config-path $HYDRA_CONFIG_DIR \
        --config-name $HYDRA_CONFIG_NAME \
        gpu_id=0 \
        $MODEL_PATH_OVERRIDE \
        $LOOP_LAYERS_IDX_OVERRIDE \
        $HYDRA_OUTPUT_OVERRIDE \
        > "$LOG_FILE" 2>&1 &
        # 注意: config-path 指向包含 config.yaml 的目录
        # 其他参数可以在此继续添加覆盖

    # 短暂暂停，避免瞬间提交过多任务 (可选)
    # sleep 0.1 
done

echo "-----------------------------------------------------"
echo "所有${#LOOP_CONFIGS[@]}个任务已在后台启动。"
echo "使用 'jobs' 命令查看后台任务状态。"
echo "使用 'htop' 或 'nvidia-smi' 监控资源使用情况。"
echo "等待所有任务完成..."

wait # 等待所有后台进程完成

echo "-----------------------------------------------------"
echo "所有任务已完成!"
echo "请检查 '$LOG_DIR' 目录下的日志文件。"
echo "Hydra 的输出目录已根据模型和层配置分别存储，请检查 'model_comparison_results/' 目录。" 