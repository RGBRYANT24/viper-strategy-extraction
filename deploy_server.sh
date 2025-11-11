#!/bin/bash
# 服务器部署和训练脚本
# 用于在远程服务器上设置环境并启动并行训练

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 默认参数
DEPTH_MIN=5
DEPTH_MAX=25
DEPTH_STEP=5
LEAVES_MIN=25
LEAVES_MAX=150
LEAVES_STEP=25
N_ITERATIONS=10
SAMPLES_PER_ITER=50000
N_WORKERS=4
DRY_RUN=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --depth-min)
            DEPTH_MIN="$2"
            shift 2
            ;;
        --depth-max)
            DEPTH_MAX="$2"
            shift 2
            ;;
        --depth-step)
            DEPTH_STEP="$2"
            shift 2
            ;;
        --leaves-min)
            LEAVES_MIN="$2"
            shift 2
            ;;
        --leaves-max)
            LEAVES_MAX="$2"
            shift 2
            ;;
        --leaves-step)
            LEAVES_STEP="$2"
            shift 2
            ;;
        --n-iterations)
            N_ITERATIONS="$2"
            shift 2
            ;;
        --samples-per-iter)
            SAMPLES_PER_ITER="$2"
            shift 2
            ;;
        --n-workers)
            N_WORKERS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --depth-min N         最小深度 (默认: 5)"
            echo "  --depth-max N         最大深度 (默认: 25)"
            echo "  --depth-step N        深度步长 (默认: 5)"
            echo "  --leaves-min N        最小叶子节点 (默认: 25)"
            echo "  --leaves-max N        最大叶子节点 (默认: 150)"
            echo "  --leaves-step N       叶子节点步长 (默认: 25)"
            echo "  --n-iterations N      迭代次数 (默认: 10)"
            echo "  --samples-per-iter N  每轮采样数 (默认: 50000)"
            echo "  --n-workers N         并行进程数 (默认: 4)"
            echo "  --dry-run             仅测试不训练"
            echo "  -h, --help            显示此帮助信息"
            exit 0
            ;;
        *)
            echo_error "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "VIPER并行训练部署脚本"
echo "=========================================="
echo ""

# 检查Python环境
echo_info "检查Python环境..."
if ! command -v python &> /dev/null; then
    echo_error "未找到Python，请先安装Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo_info "Python版本: $PYTHON_VERSION"

# 检查必要的依赖
echo_info "检查依赖包..."
REQUIRED_PACKAGES=("numpy" "torch" "gymnasium" "sklearn" "sb3_contrib" "joblib" "pandas")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo_warn "缺少以下依赖包: ${MISSING_PACKAGES[*]}"
fi

# 检查Oracle模型
echo_info "检查Oracle模型..."
ORACLE_PATH="log/oracle_TicTacToe_ppo_aggressive.zip"
if [ ! -f "$ORACLE_PATH" ]; then
    echo_error "未找到Oracle模型: $ORACLE_PATH"
    echo_warn "请先训练或下载Oracle模型"
    exit 1
fi

echo_info "Oracle模型: $ORACLE_PATH"

# 显示训练配置
echo ""
echo "=========================================="
echo "训练配置"
echo "=========================================="
echo "深度范围: $DEPTH_MIN - $DEPTH_MAX (步长: $DEPTH_STEP)"
echo "叶子节点范围: $LEAVES_MIN - $LEAVES_MAX (步长: $LEAVES_STEP)"
echo "迭代次数: $N_ITERATIONS"
echo "每轮采样: $SAMPLES_PER_ITER"
echo "并行进程: $N_WORKERS"

# 计算总配置数
DEPTH_COUNT=$(( ($DEPTH_MAX - $DEPTH_MIN) / $DEPTH_STEP + 1 ))
LEAVES_COUNT=$(( ($LEAVES_MAX - $LEAVES_MIN) / $LEAVES_STEP + 1 ))
TOTAL_CONFIGS=$(( $DEPTH_COUNT * $LEAVES_COUNT ))

echo "总配置数: $TOTAL_CONFIGS"
echo ""

# Dry run模式
if [ "$DRY_RUN" = true ]; then
    echo_warn "DRY RUN 模式：仅测试配置，不实际训练"
    DRY_RUN_FLAG="--dry-run"
else
    DRY_RUN_FLAG=""
fi

# 创建日志目录
LOG_DIR="log/viper_parallel_training"
mkdir -p "$LOG_DIR"

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TRAIN_LOG="$LOG_DIR/train_${TIMESTAMP}.log"

echo_info "日志文件: $TRAIN_LOG"
echo ""

# 确认开始训练
if [ "$DRY_RUN" = false ]; then
    echo "=========================================="
    read -p "确认开始训练? (yes/no): " -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
        echo_warn "训练已取消"
        exit 0
    fi
fi

# 启动训练
echo "=========================================="
echo_info "启动并行训练..."
echo "=========================================="
echo ""

# 使用nohup在后台运行，以防SSH连接断开
python train_parallel_viper.py \
    --depth-range $DEPTH_MIN $DEPTH_MAX \
    --depth-step $DEPTH_STEP \
    --leaves-range $LEAVES_MIN $LEAVES_MAX \
    --leaves-step $LEAVES_STEP \
    --n-iterations $N_ITERATIONS \
    --samples-per-iter $SAMPLES_PER_ITER \
    --n-workers $N_WORKERS \
    $DRY_RUN_FLAG \
    2>&1 | tee "$TRAIN_LOG"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo_info "训练完成!"
    echo "=========================================="
    echo ""
    echo_info "日志文件: $TRAIN_LOG"
    echo_info "模型目录: log/viper_mask_ppo_tictactoe"
    echo ""
    echo_info "运行以下命令分析结果:"
    echo "  python analyze_viper_results.py --models-dir log/viper_mask_ppo_tictactoe"
else
    echo ""
    echo "=========================================="
    echo_error "训练失败 (退出码: $EXIT_CODE)"
    echo "=========================================="
    echo ""
    echo_info "查看日志: $TRAIN_LOG"
    exit $EXIT_CODE
fi