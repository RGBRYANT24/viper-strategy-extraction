# Decision Tree Rules Visualization Tools

This directory contains tools for visualizing and analyzing the decision tree rules learned by VIPER.

## Files

- **`visualize_tree_rules.py`** - Interactive Chinese visualization tool
- **`export_rules_report.py`** - English report generator (for advisors/papers)
- **`test_visualization.py`** - Test cases for visualization logic
- **`example_report.txt`** - Sample output format

## Quick Start

### For Interactive Exploration (Chinese)

```bash
# View summary of top 10 rules
python visualize_tree_rules.py

# View specific rule in detail
python visualize_tree_rules.py --rule 1

# Interactive mode (explore rules one by one)
python visualize_tree_rules.py --interactive

# View all rules
python visualize_tree_rules.py --all

# Use custom JSON file
python visualize_tree_rules.py path/to/rules.json --rule 5
```

### For Generating Reports (English)

```bash
# Generate full report with all rules
python export_rules_report.py

# Generate report with top 20 rules only
python export_rules_report.py --max-rules 20

# Specify output file name
python export_rules_report.py --output my_report.txt

# Use custom JSON file
python export_rules_report.py path/to/rules.json --output report.txt
```

## Output Format

### Board State Representation

Each rule shows a 3x3 TicTacToe board with the following symbols:

```
┌─────┬─────┬─────┐
│  O  │ ?/O │  ?  │  <- Row 1
├─────┼─────┼─────┤
│ ?/X │  ?  │ ?/X │  <- Row 2
├─────┼─────┼─────┤
│  X  │  ?  │  O  │  <- Row 3
└─────┴─────┴─────┘
```

**Legend:**
- `X` = Player's piece (确定是自己的棋子)
- `O` = Opponent's piece (确定是对手的棋子)
- `.` = Empty position (确定是空位)
- `?` = Unconstrained (规则未约束此位置)
- `?/X` = Not opponent (不是对手，可能是空或自己)
- `?/O` = Not player (不是自己，可能是空或对手)

### Action Priorities

Each rule shows the priority scores and probabilities for all 9 possible actions:

```
Rank   Pos   Position        Score      Prob     Status     Mark
----------------------------------------------------------------------
1      4     Center         -0.330     88.35%   Unknown    ✓ BEST
2      6     Bottom-Left    -2.861     3.74%    Player(X)
3      3     Mid-Left       -3.039     3.44%    Unknown
...
```

- **Rank**: Priority ranking (1 = highest)
- **Score**: Logit value (higher = better)
- **Prob**: Softmax probability
- **Status**: Whether the position is occupied
- **Mark**: Indicates the best action

## Understanding the Output

### Priority Scores

Rules are sorted by priority score (higher = more important). This score indicates:
- How confident the model is in this rule
- How important this pattern is for the policy
- Rules with higher priority were more influential during training

### Support Count

Number of training samples that matched this rule:
- Higher support → more reliable rule
- Lower support → less common scenario

### Rule Validation

The tool automatically checks for potential issues:
- ✓ Rule appears valid - No obvious problems
- ⚠ Warning - Potential strategic issues (e.g., best action on occupied position)
- ❌ Error - Clear logical problems

## Use Cases

### 1. Policy Verification
Check if the learned policy makes strategic sense:
```bash
python visualize_tree_rules.py --rule 1
```

### 2. Compare with Optimal Strategy
Look at high-priority rules to see if they match MinMax decisions:
```bash
python visualize_tree_rules.py --top 5
```

### 3. Generate Report for Advisor
Create a comprehensive English report:
```bash
python export_rules_report.py --max-rules 30 --output advisor_report.txt
```

### 4. Find Specific Board States
Use interactive mode to search for interesting patterns:
```bash
python visualize_tree_rules.py --interactive
```

## Tips

1. **Start with high-priority rules** - They represent the most important patterns
2. **Check for unconstrained positions** - Many `?` means the rule is very general
3. **Look at support counts** - High support = more reliable
4. **Compare action probabilities** - Large gap means confident decision
5. **Watch for warnings** - They indicate potential issues with the learned policy

## Example Workflow

```bash
# 1. First, get an overview
python visualize_tree_rules.py --top 10

# 2. Examine specific interesting rules
python visualize_tree_rules.py --rule 1
python visualize_tree_rules.py --rule 2

# 3. Generate report for presentation
python export_rules_report.py --max-rules 20 --output viper_policy_report.txt

# 4. Open the report
cat viper_policy_report.txt
# or
open viper_policy_report.txt  # macOS
# or
nano viper_policy_report.txt  # Linux
```

## Troubleshooting

### File not found error
Make sure the JSON file path is correct. Default path is:
```
log/viper_mask_ppo_tictactoe/viper_mask_ppo_tree_20251107_001324_iter10_samples50000_depth10_leaves50.rules.json
```

You can specify a different path:
```bash
python visualize_tree_rules.py your/path/to/rules.json
```

### Unicode display issues
If the box-drawing characters don't display correctly, your terminal may not support UTF-8. Try:
```bash
export LANG=en_US.UTF-8
```

### Too much output
Limit the number of rules:
```bash
python export_rules_report.py --max-rules 10
```

## For Your Advisor

The English report (`export_rules_report.py`) is specifically designed for academic presentation:

✓ Professional English formatting
✓ Clear explanations of board states
✓ Statistical summaries
✓ Sorted by importance (priority)
✓ Easy to read and understand
✓ Suitable for papers/presentations

Generate with:
```bash
python export_rules_report.py --max-rules 25 --output policy_analysis.txt
```

This creates a comprehensive document showing the learned policy in interpretable form, perfect for demonstrating the explainability of VIPER compared to black-box neural networks.
