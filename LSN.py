import math
import random
import matplotlib
import statistics

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



class LSNNode:
    def __init__(self, node_id):
        self.id = node_id
        # Shortcuts are separated into left and right dictionaries.
        # The key is the logarithmic interval, the value is the closest known node.
        self.shortcuts_left, self.shortcuts_right = {}, {}
        self.inbox, self.proposed_edges = set(), set()

    @property
    def shortcuts(self):
        return set(self.shortcuts_left.values()).union(self.shortcuts_right.values())

    def __hash__(self): return hash(self.id)

    def __eq__(self, other): return isinstance(other, LSNNode) and self.id == other.id


def run_lsn_round(nodes, current_edges):
    # Build a temporary map of the current graph
    neighbors_map = {n: set() for n in nodes}
    for edge in current_edges:
        u, v = tuple(edge)
        neighbors_map[u].add(v)
        neighbors_map[v].add(u)

    # Phase 1: Information Exchange
    # Nodes broadcast their current shortcuts and their own ID to immediate neighbors.
    for u in nodes:
        message = u.shortcuts.copy()
        message.add(u)
        for v in neighbors_map[u].union(u.shortcuts):
            if v != u: v.inbox.update(message)

    # Phase 2: Update Shortcuts
    # Calculate the logarithmic distance to all newly discovered nodes in the inbox.
    for u in nodes:
        known_pool = u.inbox.union(neighbors_map[u])
        for x in known_pool:
            if x.id == u.id: continue
            dist = abs(u.id - x.id)
            # Base-two logarithm determines the interval bucket
            interval = int(math.log2(dist)) + 1 if dist > 0 else 0

            # Keep only one node per interval
            if x.id < u.id and interval not in u.shortcuts_left:
                u.shortcuts_left[interval] = x
            elif x.id > u.id and interval not in u.shortcuts_right:
                u.shortcuts_right[interval] = x

    # Phase 3: Edge Proposals
    # Evaluate known peers to propose edges that sort the line.
    for u in nodes:
        u.proposed_edges.clear()
        u_remembers = neighbors_map[u].union(u.shortcuts).union({u})
        for v in neighbors_map[u]:
            if v.id < u.id:
                # Propose edge to the smallest known node that is larger than the left neighbor
                candidates = {w for w in u_remembers if w.id > v.id}
                if candidates:
                    w = min(candidates, key=lambda x: x.id)
                    u.proposed_edges.add(frozenset({v, w}))
            elif v.id > u.id:
                # Propose edge to the largest known node that is smaller than the right neighbor
                candidates = {w for w in u_remembers if w.id < v.id}
                if candidates:
                    w = max(candidates, key=lambda x: x.id)
                    u.proposed_edges.add(frozenset({v, w}))

    # Clear inbox for the next discrete time step
    for u in nodes:
        u.inbox.clear()

    # Collect all proposals globally to prevent mid-round topology changes
    next_edges = set()
    for u in nodes:
        next_edges.update(u.proposed_edges)

    return next_edges


#simulation
def calculate_correctness(nodes, current_edges):
    sorted_nodes = sorted(nodes, key=lambda n: n.id)
    expected_edges = set(frozenset({sorted_nodes[i], sorted_nodes[i + 1]}) for i in range(len(sorted_nodes) - 1))
    if current_edges == expected_edges: return 100.0
    correct_links = len(current_edges.intersection(expected_edges))
    total_relevant = max(len(current_edges), len(expected_edges))
    return (correct_links / total_relevant) * 100.0 if total_relevant > 0 else 0


def run_single_simulation(nodes, current_edges, max_rounds=1000):
    for round_idx in range(1, max_rounds + 1):
        current_edges = run_lsn_round(nodes, current_edges)
        if calculate_correctness(nodes, current_edges) == 100.0:
            return round_idx
    return max_rounds


def clear_node_state(nodes):
    for u in nodes:
        u.shortcuts_left.clear()
        u.shortcuts_right.clear()
        u.inbox.clear()
        u.proposed_edges.clear()


def build_star_topology(n):
    nodes = [LSNNode(i) for i in range(n)]
    hub = nodes[n // 2]
    edges = {frozenset({hub, node}) for node in nodes if node != hub}
    clear_node_state(nodes)
    for u, v in edges: u.neighbors.add(v); v.neighbors.add(u)
    return nodes, edges


def build_jumbled_ring_topology(n, seed):
    nodes = [LSNNode(i) for i in range(n)]
    shuffled = nodes.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    edges = {frozenset({shuffled[i], shuffled[(i + 1) % len(shuffled)]}) for i in range(len(shuffled))}
    clear_node_state(nodes)
    for u, v in edges: u.neighbors.add(v); v.neighbors.add(u)
    return nodes, edges


def build_shuffled_tree_topology(n, seed):
    ids = list(range(n))
    random.seed(seed)
    random.shuffle(ids)
    nodes = [LSNNode(id) for id in ids]
    edges = {frozenset({nodes[i], nodes[(i - 1) // 2]}) for i in range(1, n)}
    clear_node_state(nodes)
    for u, v in edges: u.neighbors.add(v); v.neighbors.add(u)
    return nodes, edges


def build_theoretical_worst_case(n):
    """
    Theorem 2.1.
    """
    nodes = [LSNNode(i) for i in range(n)]
    edges = set()
    for i in range(1, n - 1):
        edges.add(frozenset({nodes[i], nodes[i + 1]}))
    edges.add(frozenset({nodes[n - 1], nodes[0]}))

    clear_node_state(nodes)
    for u, v in edges:
        u.neighbors.add(v)
        v.neighbors.add(u)
    return nodes, edges



def run_benchmarks():
    sizes = [16, 32, 64, 128, 256, 512]
    iterations = 10

    results_mean = {"Star Graph": [], "Jumbled Ring": [], "Shuffled Tree": [], "Th. 2.1 (PL-Worst)": []}
    results_std = {"Star Graph": [], "Jumbled Ring": [], "Shuffled Tree": [], "Th. 2.1 (PL-Worst)": []}

    print("=" * 145)
    print(f"MONTE-CARLO-EVALUATION ({iterations} rounds per N)")
    print("=" * 145)
    print(
        f"{'N':<5} | {'Star (Avg±Std)':<16} | {'Ring (Avg±Std)':<16} | {'Tree (Avg±Std)':<16} | {'Th. 2.1 (Avg±Std)':<18} | {'ceil(log2^2 N)':<14} | {'Empirisches c':<15}")
    print("-" * 145)

    c_values = []

    for n in sizes:
        runs_star, runs_ring, runs_tree, runs_worst = [], [], [], []

        for i in range(iterations):
            seed = 42 + i
            runs_star.append(run_single_simulation(*build_star_topology(n)))
            runs_ring.append(run_single_simulation(*build_jumbled_ring_topology(n, seed)))
            runs_tree.append(run_single_simulation(*build_shuffled_tree_topology(n, seed)))
            runs_worst.append(run_single_simulation(*build_theoretical_worst_case(n)))

        mean_star = statistics.mean(runs_star)
        mean_ring = statistics.mean(runs_ring)
        mean_tree = statistics.mean(runs_tree)
        mean_worst = statistics.mean(runs_worst)

        std_star = statistics.stdev(runs_star) if iterations > 1 else 0
        std_ring = statistics.stdev(runs_ring) if iterations > 1 else 0
        std_tree = statistics.stdev(runs_tree) if iterations > 1 else 0
        std_worst = statistics.stdev(runs_worst) if iterations > 1 else 0

        results_mean["Star Graph"].append(mean_star)
        results_std["Star Graph"].append(std_star)

        results_mean["Jumbled Ring"].append(mean_ring)
        results_std["Jumbled Ring"].append(std_ring)

        results_mean["Shuffled Tree"].append(mean_tree)
        results_std["Shuffled Tree"].append(std_tree)

        results_mean["Th. 2.1 (PL-Worst)"].append(mean_worst)
        results_std["Th. 2.1 (PL-Worst)"].append(std_worst)

        # Änderung: Wir testen gegen O(log^2 n) statt O(log n)
        theo_log_sq = math.ceil(math.log2(n) ** 2)
        avg_case_max = max(mean_ring, mean_tree, mean_star)
        c_local = avg_case_max / (math.log2(n) ** 2)
        c_values.append(c_local)

        print(
            f"{n:<5} | {mean_star:>4.1f} ± {std_star:<5.2f}    | {mean_ring:>4.1f} ± {std_ring:<5.2f}    | {mean_tree:>4.1f} ± {std_tree:<5.2f}    | {mean_worst:>4.1f} ± {std_worst:<5.2f}      | {theo_log_sq:<14} | c ≈ {c_local:.2f}")

    print("=" * 145)

    c_avg = sum(c_values) / len(c_values)

    plot_results(sizes, results_mean, results_std, c_avg)


def plot_results(sizes, results_mean, results_std, c_avg):

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax1.errorbar(sizes, results_mean["Jumbled Ring"], yerr=results_std["Jumbled Ring"],
                 marker='s', color='#d62728', linewidth=2.5, markersize=8, capsize=5, label='Jumbled Ring')

    ax1.errorbar(sizes, results_mean["Shuffled Tree"], yerr=results_std["Shuffled Tree"],
                 marker='^', color='#ff7f0e', linewidth=2.5, markersize=8, capsize=5, label='Shuffled Tree')

    ax1.errorbar(sizes, results_mean["Star Graph"], yerr=results_std["Star Graph"],
                 marker='o', color='#1f77b4', linewidth=2, markersize=7, capsize=5, label='Star Graph')

    x_smooth = [sizes[0] + i * (sizes[-1] - sizes[0]) / 100.0 for i in range(101)]
    theo_log_sq_smooth = [math.log2(x) ** 2 for x in x_smooth]
    emp_log_sq_smooth = [c_avg * (math.log2(x) ** 2) for x in x_smooth]

    ax1.plot(x_smooth, theo_log_sq_smooth, color='gray', linestyle='-.', linewidth=2,
             label=rf'Theory: $\log_2^2(n)$')
    ax1.plot(x_smooth, emp_log_sq_smooth, color='black', linestyle=':', linewidth=2.5,
             label=rf'Empiric (Max Avg): $y \approx {c_avg:.2f} \cdot \log_2^2(n)$')

    ax1.set_xlabel('Number of Nodes (N)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Number of Rounds', fontweight='bold', fontsize=12)
    ax1.set_title(r'LSN Average-Cases: Visible $\mathcal{O}(\log^2 n)$ Curvature', fontweight='bold', fontsize=14)

    ax1.set_xticks(sizes)
    ax1.grid(True, which="both", linestyle='--', alpha=0.6)
    ax1.set_facecolor('#fafafa')
    ax1.legend(fontsize=11)
    fig1.tight_layout()


    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(sizes, results_mean["Th. 2.1 (PL-Worst)"], marker='D', color='purple', linewidth=2.5, markersize=8,
             label='Theorem 2.1 Worst-Case Topology')
    ax2.plot(sizes, results_mean["Jumbled Ring"], marker='s', color='#d62728', linewidth=2.5, markersize=8,
             label='Jumbled Ring (Average)')

    theo_linear = [n for n in sizes]
    ax2.plot(sizes, theo_linear, color='black', linestyle='-.', linewidth=2,
             label=r'Pure Linearization (PL) Bound $\mathcal{O}(n)$')

    ax2.set_xlabel('Number of Nodes (N) [Linear Scale]', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Rounds to 100% Stabilization', fontweight='bold', fontsize=12)
    ax2.set_title(r'LSN Outperforms PL Worst-Case (Theorem 2.1)', fontweight='bold', fontsize=14)

    ax2.set_xticks(sizes)
    ax2.grid(True, which="both", linestyle='--', alpha=0.6)
    ax2.set_facecolor('#fafafa')
    ax2.legend(fontsize=11)
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    run_benchmarks()