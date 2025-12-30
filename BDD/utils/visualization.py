import json
import os
import time

def json_to_dot(json_file, dot_file):
    """
    Converts a BDD JSON dump (from dd library) to a Graphviz DOT file.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 1. Reverse mapping: level -> variable name
    level_to_var = {v: k for k, v in data['level_of_var'].items()}

    lines = [
        "digraph BDD {",
        "    rankdir=TB;",  # Top to Bottom
        "    node [shape=circle];",
        "    // Terminal nodes",
        "    \"T\" [shape=box, label=\"True\", style=filled, color=lightgrey];",
        "    \"F\" [shape=box, label=\"False\", style=filled, color=lightgrey];"
    ]

    # 2. Helper to format edges
    def format_edge(u, v, style):
        if isinstance(v, str):
            target = f'"{v}"'
            attrs = f'style={style}'
        else:
            # Handle CUDD complemented edges (negative means complement)
            is_complemented = (v < 0)
            target_id = abs(v)
            target = f'"{target_id}"'
            # Correct DOT syntax: comma separated attributes
            attrs = f'style={style}'
            if is_complemented:
                attrs += ', label="-1"'
        
        return f'    "{u}" -> {target} [{attrs}];'

    # 3. Iterate nodes
    for node_id, content in data.items():
        if node_id in ["level_of_var", "roots"]:
            continue
        
        level, low, high = content
        var_name = level_to_var.get(level, f"Level {level}")
        
        # Node definition
        lines.append(f'    "{node_id}" [label="{var_name}"];')
        
        # Edges: Low (dashed), High (solid)
        lines.append(format_edge(node_id, low, "dashed"))
        lines.append(format_edge(node_id, high, "solid"))

    # 4. Roots
    for i, root in enumerate(data.get('roots', [])):
        is_neg = root < 0
        root_id = abs(root)
        label = "NOT " if is_neg else ""
        lines.append(f'    root_{i} [shape=point, style=invis];')
        lines.append(f'    root_{i} -> "{root_id}" [label="{label}root"];')

    lines.append("}")

    # Write to file
    with open(dot_file, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"✅ DOT file generated: {dot_file}")


def generate_visualization(bdd_manager, bdd_nodes, output_base_name, output_dir=None):
    """
    Generates both JSON and DOT files for a given BDD node.
    
    Args:
        bdd_manager: The BDD manager instance.
        bdd_nodes: A single BDD node or a list of BDD nodes (roots).
        output_base_name: Base name for the output files. A timestamp and size info will be appended.
        output_dir: Directory to save the files. Defaults to 'BDD/generated_bdds'.
    """
    if output_dir is None:
        # Default to a folder relative to the project root or BDD folder
        # Trying to find a reasonable location based on current working directory
        if os.path.exists("BDD"):
            output_dir = os.path.join("BDD", "generated_bdds")
        else:
            output_dir = "generated_bdds"

    if not isinstance(bdd_nodes, list):
        bdd_nodes = [bdd_nodes]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename with timestamp and node count estimate
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Note: bdd_manager doesn't easily give exact node count of a subgraph without traversal, 
    # but we can just use timestamp for uniqueness.
    # If the user wants size, we can assume 'bdd_nodes' might be small enough or just skip it in name for now.
    
    final_name = f"{output_base_name}_{timestamp}"
    
    json_path = os.path.join(output_dir, f"{final_name}.json")
    dot_path = os.path.join(output_dir, f"{final_name}.dot")

    print(f"dumping BDD to {json_path}...")
    bdd_manager.dump(json_path, roots=bdd_nodes)
    
    print(f"Converting to DOT: {dot_path}...")
    json_to_dot(json_path, dot_path)
    
    return json_path, dot_path


def format_bdd_to_logic(bdd_manager, bdd_node, limit=20):
    """
    Converts a BDD node to a readable propositional logic string (Sum-of-Products form).
    Uses symbols: ∧ (AND), ∨ (OR), ¬ (NOT).
    
    Args:
        bdd_manager: The BDD manager.
        bdd_node: The BDD node to format.
        limit: Max number of clauses to display (to prevent huge output).
    """
    if bdd_node == bdd_manager.true:
        return "TRUE"
    if bdd_node == bdd_manager.false:
        return "FALSE"

    clauses = []
    iterator = bdd_manager.pick_iter(bdd_node)
    
    try:
        count = 0
        for assignment in iterator:
            if count >= limit:
                clauses.append("...")
                break
            
            literals = []
            # Sort keys for consistent output
            for var in sorted(assignment.keys()):
                val = assignment[var]
                if val:
                    literals.append(f"{var}")
                else:
                    literals.append(f"¬{var}")
            
            # Join literals with AND
            clause = "(" + " ∧ ".join(literals) + ")"
            clauses.append(clause)
            count += 1
            
    except Exception as e:
        return f"Error formatting logic: {e}"

    # Join clauses with OR
    return " ∨ ".join(clauses)

