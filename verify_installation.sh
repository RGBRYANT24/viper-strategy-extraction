#!/bin/bash
# 规则提取功能验证脚本
# 在服务器上运行此脚本以验证功能是否正确安装

echo "========================================"
echo "规则提取功能安装验证"
echo "========================================"
echo ""

# 检查Python版本
echo "[1/6] 检查Python版本..."
python --version
if [ $? -ne 0 ]; then
    echo "❌ Python未安装或未在PATH中"
    exit 1
fi
echo "✅ Python版本检查通过"
echo ""

# 检查scipy是否安装
echo "[2/6] 检查scipy依赖..."
python -c "import scipy; print(f'scipy version: {scipy.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ scipy未安装"
    echo "请运行: pip install scipy==1.11.4"
    exit 1
fi
echo "✅ scipy已安装"
echo ""

# 检查其他依赖
echo "[3/6] 检查其他依赖..."
python -c "import numpy, sklearn, joblib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少必要的依赖"
    echo "请运行: pip install -r requirements.txt"
    exit 1
fi
echo "✅ 所有依赖已安装"
echo ""

# 检查文件是否存在
echo "[4/6] 检查文件完整性..."
files=(
    "model/rule_extractor.py"
    "test/test_rule_extractor.py"
    "extract_tree_rules.py"
    "demo_rule_extraction.py"
)

missing_files=()
for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ 以下文件缺失:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    exit 1
fi
echo "✅ 所有文件已就绪"
echo ""

# 运行导入测试
echo "[5/6] 测试模块导入..."
python -c "from model.rule_extractor import Rule, DecisionTreeRuleExtractor, extract_and_simplify_rules; print('模块导入成功')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 模块导入失败"
    exit 1
fi
echo "✅ 模块导入成功"
echo ""

# 运行单元测试
echo "[6/6] 运行单元测试..."
python test/test_rule_extractor.py > /tmp/test_output.txt 2>&1
test_exit_code=$?

if [ $test_exit_code -eq 0 ]; then
    echo "✅ 所有测试通过"
    # 显示测试统计
    grep -E "Ran|OK|PASSED" /tmp/test_output.txt | tail -2
else
    echo "❌ 测试失败"
    echo "详细信息请查看: /tmp/test_output.txt"
    tail -20 /tmp/test_output.txt
    exit 1
fi
echo ""

# 最终总结
echo "========================================"
echo "✅ 验证完成！"
echo "========================================"
echo ""
echo "🎉 规则提取功能已成功安装并通过所有测试"
echo ""
echo "下一步操作:"
echo "  1. 运行演示: python demo_rule_extraction.py"
echo "  2. 提取规则: python extract_tree_rules.py --tree-path <your-model>"
echo "  3. 查看文档: cat QUICK_START_RULE_EXTRACTION.md"
echo ""
echo "完整文档: RULE_EXTRACTION_README.md"
echo ""
