import math
import random
import time
import statistics
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt



class SkipPlusNode:
    def __init__(self, node_id, vector_len=24):
        self.id = node_id
        self.vector = tuple(random.choice([0, 1]) for _ in range(vector_len))
        self.neighbors = set()

    def get_bit(self, idx):
        return self.vector[idx] if idx < len(self.vector) else 0


def build_perfect_skip_plus(n):
    nodes = [SkipPlusNode(i) for i in range(n)]
    nodes.sort(key=lambda x: x.id)

    for i in range(24):
        groups = {}
        for node in nodes:
            pfx = node.vector[:i]
            if pfx not in groups:
                groups[pfx] = []
            groups[pfx].append(node)

        for group in groups.values():
            if len(group) <= 1:
                continue

            for idx, u in enumerate(group):
                u_bit = u.get_bit(i)
                left_idx = idx - 1
                while left_idx >= 0:
                    if group[left_idx].get_bit(i) != u_bit:
                        break
                    left_idx -= 1
                if left_idx < 0: left_idx = 0

                right_idx = idx + 1
                while right_idx < len(group):
                    if group[right_idx].get_bit(i) != u_bit:
                        break
                    right_idx += 1
                if right_idx >= len(group): right_idx = len(group) - 1

                for target_idx in range(left_idx, right_idx + 1):
                    if target_idx != idx:
                        v = group[target_idx]
                        u.neighbors.add(v)
                        v.neighbors.add(u)
    return nodes


def greedy_route(start_node, target_id):
    current = start_node
    hops = 0
    visited = set()

    while current.id != target_id:
        visited.add(current.id)
        best_neighbor = current
        best_dist = abs(current.id - target_id)

        for neighbor in current.neighbors:
            dist = abs(neighbor.id - target_id)
            if dist < best_dist and neighbor.id not in visited:
                best_dist = dist
                best_neighbor = neighbor

        if best_neighbor == current:
            return float('inf')

        current = best_neighbor
        hops += 1

    return hops



def evaluate_extreme_routing():
    sizes = [1000, 5000, 10000, 50000, 100000]
    avg_hops_history = []
    max_hops_history = []


    for n in sizes:
        iter_avg_hops = []
        iter_max_hops = []

        start_time = time.time()


        for seed in range(3):
            random.seed(42 + seed)
            nodes_list = build_perfect_skip_plus(n)

            total_hops = 0
            local_max = 0
            tests = 5000

            for _ in range(tests):
                source = random.choice(nodes_list)
                target = random.choice(nodes_list)
                while source.id == target.id:
                    target = random.choice(nodes_list)

                hops = greedy_route(source, target.id)
                total_hops += hops
                local_max = max(local_max, hops)

            iter_avg_hops.append(total_hops / tests)
            iter_max_hops.append(local_max)

        total_time = time.time() - start_time


        final_avg = statistics.mean(iter_avg_hops)

        final_max = max(iter_max_hops)

        avg_hops_history.append(final_avg)
        max_hops_history.append(final_max)

        print(
            f"N={n:<7} | Zeit: {total_time:.1f}s | Avg Hops: {final_avg:.2f} | Max Hops (Sampled): {final_max} | log2(N): {math.log2(n):.2f}")


    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sizes, avg_hops_history, marker='o', color='#1f77b4', linewidth=2.5, markersize=8,
            label='Average Path Length (Avg Hops)')
    ax.plot(sizes, max_hops_history, marker='^', color='#d62728', linewidth=2, markersize=7, linestyle='--',
            label='Sampled Maximum Path Length (Worst-Case)')

    theo_log = [math.log2(x) for x in sizes]
    ax.plot(sizes, theo_log, color='black', linestyle=':', linewidth=2, label=rf'Theoretical Bound: $\log_2(n)$')

    ax.set_title(r'SKIP+ Routing Efficiency under Large Scaling ($N \leq 100,000$)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Number of Nodes (N)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Hops', fontweight='bold', fontsize=12)

    ax.set_xscale('log', base=10)
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.grid(True, which="both", linestyle='--', alpha=0.6)
    ax.set_facecolor('#fafafa')
    ax.legend(fontsize=11)

    fig.tight_layout()

    filename = "skip_routing_100k_en.png"
    plt.savefig(filename, dpi=300)


if __name__ == '__main__':
    evaluate_extreme_routing()