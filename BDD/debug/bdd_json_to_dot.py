import json

def json_to_dot(json_file, dot_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 1. 建立层级到变量名的反向映射 (例如 0 -> "x_0_0")
    # 这样图里显示的是变量名，而不是看不懂的数字
    level_to_var = {v: k for k, v in data['level_of_var'].items()}

    lines = [
        "digraph BDD {",
        "    rankdir=TB;",  # 从上到下绘制
        "    node [shape=circle];",
        "    // 终点节点样式",
        "    \"T\" [shape=box, label=\"True\", style=filled, color=lightgrey];",
        "    \"F\" [shape=box, label=\"False\", style=filled, color=lightgrey];"
    ]

    # 2. 辅助函数：处理边的 ID（处理取反的情况）
    def format_edge(u, v, style):
        # 如果 v 是字符串 (如 "T" 或 "F")
        if isinstance(v, str):
            target = f'"{v}"'
            label = ""
        else:
            # 处理 CUDD 的补码边 (负数表示取反)
            is_complemented = (v < 0)
            target_id = abs(v)
            target = f'"{target_id}"'
            # 如果是补码边，加一个标记，或者在边上画个圈
            label = ' [label="-1"]' if is_complemented else ""
        
        return f'    "{u}" -> {target} [style={style}{label}];'

    # 3. 遍历所有节点生成 DOT
    # data 中除了 level_of_var 和 roots，其他 key 都是节点 ID
    for node_id, content in data.items():
        if node_id in ["level_of_var", "roots"]:
            continue
        
        level, low, high = content
        var_name = level_to_var.get(level, f"Level {level}")
        
        # 定义节点显示内容
        lines.append(f'    "{node_id}" [label="{var_name}"];')
        
        # 连线：Low (虚线), High (实线)
        lines.append(format_edge(node_id, low, "dashed"))
        lines.append(format_edge(node_id, high, "solid"))

    # 4. 标记根节点
    for i, root in enumerate(data.get('roots', [])):
        # 根节点可能带负号
        is_neg = root < 0
        root_id = abs(root)
        label = "NOT " if is_neg else ""
        lines.append(f'    root_{i} [shape=point, style=invis];')
        lines.append(f'    root_{i} -> "{root_id}" [label="{label}root"];')

    lines.append("}")

    # 写入文件
    with open(dot_file, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"✅ 转换完成！请查看: {dot_file}")
    print("👉 现在你可以用 VS Code 打开它并使用 'Graphviz Preview' 插件查看了。")

import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BDD JSON to Graphviz DOT format.")
    parser.add_argument("input", nargs="?", default="user_rule.json", help="Input JSON file path")
    parser.add_argument("output", nargs="?", default="debug.dot", help="Output DOT file path")
    args = parser.parse_args()

    input_path = args.input
    # If file doesn't exist in current dir, check if we are in BDD/debug and file is in root
    if not os.path.exists(input_path):
        # Try looking in project root if we are running from a subdir or if file is expected in root
        # Assuming script might be run from root or BDD/debug
        potential_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", input_path)
        if os.path.exists(potential_root_path):
            input_path = potential_root_path
        elif os.path.exists(os.path.join("..", "..", input_path)): # Fallback relative check
            input_path = os.path.join("..", "..", input_path)

    if not os.path.exists(input_path):
        print(f"❌ Error: Input file '{input_path}' not found.")
        print(f"   Pwd: {os.getcwd()}")
        exit(1)

    print(f"📂 Reading from: {input_path}")
    json_to_dot(input_path, args.output)