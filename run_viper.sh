#!/bin/bash

# --- SLURM 资源申请指令 ---
# 这是您告诉调度系统需要什么资源的部分

# 为您的任务命名，方便识别
#SBATCH --job-name=Viper-RL-Job

# 指定标准输出和错误日志的文件名 (%j 会被替换为任务ID)
#SBATCH --output=viper-output-%j.out
#SBATCH --error=viper-error-%j.err

# **重要**: 指定您想在哪种GPU上运行
# 可选项: L40SNodes, RTX6000Node, RTX8000Nodes, GPUNodes
#SBATCH --partition=L40SNodes

# **重要**: 申请GPU的数量 (平台限制最多4个)
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

# 申请CPU核心数和任务数
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

# --- 任务执行指令 ---
# 这部分是任务开始后实际执行的命令

echo "== SLURM 任务开始 =="
echo "任务ID: $SLURM_JOB_ID"
echo "运行节点: $(hostname)"
echo "开始时间: $(date)"
echo "工作目录: $(pwd)"
echo ""

# **核心执行命令**:
# srun: SLURM 的任务启动器
# singularity exec: 在指定的容器内执行命令
# /projects/noggins/viper_env/bin/python: 调用您虚拟环境中的Python解释器
# "$HOME/PRIVE/Projects/Viper/viper-verifiable-rl-impl/main.py": 您要运行的主程序脚本
srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif /projects/noggins/viper_env/bin/python "$HOME/PRIVE/Projects/Viper/viper-verifiable-rl-impl/main.py"

echo ""
echo "== SLURM 任务结束 =="
echo "结束时间: $(date)"
