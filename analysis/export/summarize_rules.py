"""
Summarize and analyze the decision tree rules
"""
import re
from collections import Counter

def parse_rules_file(filename='tree_rules_all.txt'):
    """Parse the rules file and extract statistics"""

    with open(filename, 'r') as f:
        content = f.read()

    # Extract all rules
    rules = []
    current_rule = {}

    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for rule header
        if line.startswith('RULE #'):
            # Extract rule number and stats
            match = re.search(r'RULE #(\d+) \(used in (\d+) games, ([\d.]+)% confidence\)', line)
            if match:
                current_rule = {
                    'number': int(match.group(1)),
                    'games': int(match.group(2)),
                    'confidence': float(match.group(3))
                }

        # Extract conditions
        elif line.strip().startswith('•'):
            if 'conditions' not in current_rule:
                current_rule['conditions'] = []
            current_rule['conditions'].append(line.strip()[2:])  # Remove bullet

        # Extract action
        elif '→ Place at Position' in line:
            match = re.search(r'Position (\d+)', line)
            if match:
                current_rule['action'] = int(match.group(1))
                rules.append(current_rule)
                current_rule = {}

        i += 1

    return rules


def analyze_rules(rules):
    """Analyze the rules for patterns"""
    print("=" * 80)
    print("RULE ANALYSIS")
    print("=" * 80)

    # Total stats
    total_rules = len(rules)
    total_games = sum(r['games'] for r in rules)

    print(f"\nTotal Rules: {total_rules}")
    print(f"Total Games Covered: {total_games:,}")
    print(f"Average Games per Rule: {total_games / total_rules:.1f}")

    # Coverage analysis
    print("\n" + "─" * 80)
    print("COVERAGE ANALYSIS")
    print("─" * 80)

    cumulative = 0
    for threshold in [10, 20, 50, 80, 90, 95]:
        target = total_games * threshold / 100
        for i, rule in enumerate(rules, 1):
            cumulative += rule['games']
            if cumulative >= target:
                print(f"  Top {i:2} rules cover {threshold}% of games")
                break
        cumulative = 0

    # Action distribution
    print("\n" + "─" * 80)
    print("ACTION DISTRIBUTION (Which positions are chosen most)")
    print("─" * 80)

    action_counts = Counter()
    action_games = Counter()

    for rule in rules:
        action = rule['action']
        action_counts[action] += 1
        action_games[action] += rule['games']

    position_names = [
        "TopLeft(0)", "TopCenter(1)", "TopRight(2)",
        "MidLeft(3)", "Center(4)", "MidRight(5)",
        "BotLeft(6)", "BotCenter(7)", "BotRight(8)"
    ]

    print(f"\n{'Position':<20} {'# Rules':>10} {'# Games':>12} {'Avg/Rule':>12}")
    print("─" * 60)

    for pos in sorted(action_games.keys(), key=lambda x: action_games[x], reverse=True):
        name = position_names[pos]
        num_rules = action_counts[pos]
        num_games = action_games[pos]
        avg = num_games / num_rules
        bar = "█" * int(num_games / total_games * 50)
        print(f"{name:<20} {num_rules:10} {num_games:12,} {avg:12.1f}  {bar}")

    # Rule complexity
    print("\n" + "─" * 80)
    print("RULE COMPLEXITY (Number of conditions)")
    print("─" * 80)

    complexity = [len(r['conditions']) for r in rules]
    print(f"  Average conditions per rule: {sum(complexity) / len(complexity):.1f}")
    print(f"  Min conditions: {min(complexity)}")
    print(f"  Max conditions: {max(complexity)}")

    complexity_dist = Counter(complexity)
    print(f"\n  Distribution:")
    for num_cond in sorted(complexity_dist.keys()):
        count = complexity_dist[num_cond]
        bar = "█" * (count // 2)
        print(f"    {num_cond:2} conditions: {count:3} rules {bar}")

    # Most important rules
    print("\n" + "─" * 80)
    print("TOP 10 MOST USED RULES")
    print("─" * 80)

    for i, rule in enumerate(rules[:10], 1):
        pct = rule['games'] / total_games * 100
        print(f"\n  Rule #{rule['number']} ({rule['games']} games, {pct:.1f}% of total)")
        print(f"    Conditions: {len(rule['conditions'])}")
        print(f"    Action: Position {rule['action']} ({position_names[rule['action']]})")

        # Show first few conditions
        print(f"    Key conditions:")
        for cond in rule['conditions'][:3]:
            print(f"      • {cond}")
        if len(rule['conditions']) > 3:
            print(f"      ... and {len(rule['conditions']) - 3} more")


def find_strategic_patterns(rules):
    """Find strategic patterns in rules"""
    print("\n" + "=" * 80)
    print("STRATEGIC PATTERNS")
    print("=" * 80)

    # Find "opening" rules (few conditions)
    opening_rules = [r for r in rules if len(r['conditions']) <= 5]
    print(f"\nSimple rules (≤5 conditions): {len(opening_rules)}")
    print("These likely represent opening/early game:")

    for rule in opening_rules[:5]:
        print(f"  → Rule #{rule['number']}: {rule['games']} games → Position {rule['action']}")

    # Find "blocking" rules (mention opponent)
    blocking_rules = []
    for rule in rules:
        if any('Opponent' in c for c in rule['conditions']):
            blocking_rules.append(rule)

    print(f"\nRules that check opponent positions: {len(blocking_rules)}")
    print(f"({len(blocking_rules) / len(rules) * 100:.1f}% of all rules)")

    # Find center-related rules
    center_rules = [r for r in rules if r['action'] == 4]
    total_center_games = sum(r['games'] for r in center_rules)

    print(f"\nRules that choose Center(4): {len(center_rules)}")
    print(f"Total games: {total_center_games} ({total_center_games / sum(r['games'] for r in rules) * 100:.1f}% of all)")
    print(f"Top center rule: Rule #{center_rules[0]['number']} with {center_rules[0]['games']} games")


if __name__ == "__main__":
    print("Loading rules from tree_rules_all.txt...")

    try:
        rules = parse_rules_file('tree_rules_all.txt')
        print(f"✓ Loaded {len(rules)} rules")

        analyze_rules(rules)
        find_strategic_patterns(rules)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

    except FileNotFoundError:
        print("Error: tree_rules_all.txt not found")
        print("Please run: python export_tree_text.py first")
