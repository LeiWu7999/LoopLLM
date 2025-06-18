#!/bin/bash

# 脚本用于在多个GPU上并行运行 LoopLLM/run.py, 每个GPU测试不同的循环层块。
# 支持为每个GPU指定独立的循环起止层，以实现更灵活的并行测试。
# GPU数量自动根据 LOOP_CONFIGS 数组长度调整（最多8个GPU）。

# --- 配置 ---
# 定义要并行测试的循环层块列表。
# 每个元素是一个字符串 "START_LAYER,END_LAYER"，代表一个独立的实验配置。
# 每个配置将在一个单独的GPU上运行。
#
# 示例 1: 模拟旧的配置 LAYERS_TO_TEST=(8 9 10) 和 LOOP_BLOCK_SIZE=6
LOOP_CONFIGS=(
    "0,6"
    "1,7"
    "2,8"
    "3,9"
    "4,10"
    "5,11"
    "6,12"
    "7,13"
)
#
# 示例 2: 测试从第8层开始，不同大小的循环块 (占用8个GPU)
# LOOP_CONFIGS=("8,8" "8,9" "8,10" "8,11" "8,12" "8,13" "8,14" "8,15")
#
# 示例 3: 完全自定义的混合任务
# LOOP_CONFIGS=(
#     "8,13"  # 在 GPU 0 上测试 [8, 13] (块大小 6)
#     "9,13"  # 在 GPU 1 上测试 [9, 13] (块大小 5)
#     "10,13" # 在 GPU 2 上测试 [10, 13] (块大小 4)
#     "0,7"   # 在 GPU 3 上测试 [0, 7] (块大小 8)
# )


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
PYTHON_SCRIPT_PATH="LoopLLM/run.py"

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

HYDRA_CONFIG_DIR="ppl_nll_exp/config"
echo "前置检查通过。"
# --- 检查结束 ---

# 用于存储此批量运行日志的目录
LOG_DIR="parallel_run_logs"
mkdir -p "$LOG_DIR"

# Hydra 的默认输出目录基础路径 (run.py 内部逻辑会在此基础上创建时间戳和自定义子目录)
# 此处我们不直接控制Hydra的输出目录，仅记录脚本本身的输出。
# run.py中设置了 hydra.job.chdir=false, 所以run.py的 CWD 是此脚本的运行目录。

echo "准备在${#LOOP_CONFIGS[@]}个GPU上并行启动实验..."
echo "每个GPU的测试配置 (START_LAYER,END_LAYER) 如下:"
for (( i=0; i<${#LOOP_CONFIGS[@]}; i++ )); do
    START_LAYER_TMP=$(echo ${LOOP_CONFIGS[i]} | cut -d',' -f1)
    END_LAYER_TMP=$(echo ${LOOP_CONFIGS[i]} | cut -d',' -f2)
    echo "  GPU $i: 测试循环层块 [${START_LAYER_TMP}, ${END_LAYER_TMP}]"
done
echo "日志文件将保存在: $LOG_DIR"
echo "Hydra的输出将根据循环块大小保存在不同的目录中 (例如: loop_size_5_results/run_L2-L6/)"
echo "-----------------------------------------------------"

# 循环遍历 GPU ID，根据 LOOP_CONFIGS 数组的实际长度
for (( GPU_INDEX=0; GPU_INDEX<${#LOOP_CONFIGS[@]}; GPU_INDEX++ ))
do
    # 获取当前GPU要测试的配置
    CURRENT_CONFIG=${LOOP_CONFIGS[$GPU_INDEX]}
    
    # 解析 "START,END" 格式的配置
    START_LAYER=$(echo $CURRENT_CONFIG | cut -d',' -f1)
    END_LAYER=$(echo $CURRENT_CONFIG | cut -d',' -f2)
    
    # 计算循环块大小
    LOOP_BLOCK_SIZE=$((END_LAYER - START_LAYER + 1))
    
    # 定义并创建特定于此运行的Hydra输出目录
    HYDRA_RUN_DIR="loop_size_${LOOP_BLOCK_SIZE}_results/run_L${START_LAYER}-L${END_LAYER}"
    mkdir -p "$HYDRA_RUN_DIR"
    
    # 定义此任务的日志文件路径
    LOG_FILE="$LOG_DIR/gpu${GPU_INDEX}_layers${START_LAYER}-${END_LAYER}.log"

    echo "启动任务: GPU ${GPU_INDEX} - 块 [${START_LAYER}, ${END_LAYER}], 大小 ${LOOP_BLOCK_SIZE} -> 输出: ${HYDRA_RUN_DIR}, 日志: ${LOG_FILE}"

    # 构建 Hydra 的参数覆盖
    LOOP_LAYERS_IDX_OVERRIDE="model.loop_layers_idx=[${START_LAYER},${END_LAYER}]"
    HYDRA_OUTPUT_OVERRIDE="hydra.run.dir=${HYDRA_RUN_DIR}"

    # 设置 CUDA_VISIBLE_DEVICES 以便 python 脚本只看到一个GPU

    # 运行 python 脚本命令，并在后台执行 (&)
    # 所有输出 (stdout 和 stderr) 都重定向到日志文件
    CUDA_VISIBLE_DEVICES=$GPU_INDEX python $PYTHON_SCRIPT_PATH \
        --config-path $HYDRA_CONFIG_DIR \
        --config-name $HYDRA_CONFIG_NAME \
        gpu_id=0 \
        $LOOP_LAYERS_IDX_OVERRIDE \
        $HYDRA_OUTPUT_OVERRIDE \
        > "$LOG_FILE" 2>&1 &
        # 注意: config-path 指向包含 config.yaml 的目录
        # 如果 base_model_name_or_path 等其他参数也需要为每个运行指定，请在此处添加覆盖

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
echo "Hydra 的输出目录已根据循环块大小分别存储，请检查 'loop_size_*_results/' 格式的目录。" 