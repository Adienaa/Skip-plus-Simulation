import math
import random
import matplotlib
import statistics

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


#LSN
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


#skip+
class SkipPlusNode:
    def __init__(self, node_id, vector_len=24):
        self.id = node_id
        # Random binary vector
        self.vector = tuple(random.choice([0, 1]) for _ in range(vector_len))
        self.neighbors, self.pings_received = set(), set()
        self.inbox_neighbors, self.insert_requests, self.remove_requests = set(), set(), set()
        self.ranges, self.current_stable, self.stable_levels, self.prev_stable_levels, self.F = {}, set(), {}, {}, {}

        # Persona pattern: Isolates LSN states for different prefix groups
        self.lsn_personas = {}

    def get_lsn_persona(self, pfx):
        if pfx not in self.lsn_personas: self.lsn_personas[pfx] = LSNNode(self.id)
        return self.lsn_personas[pfx]

    def shared_prefix_length(self, other):
        c = 0
        for b1, b2 in zip(self.vector, other.vector):
            if b1 == b2:
                c += 1
            else:
                break
        return c

    def get_bit(self, idx):
        return self.vector[idx] if idx < len(self.vector) else 0

    def calculate_ranges(self, inbox_neighbors):
        # Calculate boundaries based on the closest known predecessors and successors
        self.ranges = {}
        for i in range(len(self.vector) + 1):
            my_pfx = self.vector[:i]
            cands = [w for w in inbox_neighbors if w.vector[:i] == my_pfx]

            l0 = [w.id for w in cands if w.id < self.id and w.get_bit(i) == 0]
            l1 = [w.id for w in cands if w.id < self.id and w.get_bit(i) == 1]
            low = min(max(l0) if l0 else float('-inf'), max(l1) if l1 else float('-inf'))

            r0 = [w.id for w in cands if w.id > self.id and w.get_bit(i) == 0]
            r1 = [w.id for w in cands if w.id > self.id and w.get_bit(i) == 1]
            high = max(min(r0) if r0 else float('inf'), min(r1) if r1 else float('inf'))
            self.ranges[i] = (low, high)

    def calculate_stability(self, inbox_neighbors):
        # A neighbor is stable if it falls within the calculated range for a shared prefix
        self.stable_levels = {}
        for v in inbox_neighbors:
            c = self.shared_prefix_length(v)
            lowest_lvl = float('inf')
            for i in range(c + 1):
                low, high = self.ranges.get(i, (float('-inf'), float('inf')))
                if low <= v.id <= high and i < lowest_lvl: lowest_lvl = i
            if lowest_lvl != float('inf'): self.stable_levels[v] = lowest_lvl
        self.current_stable = set(self.stable_levels.keys())



def calculate_global_correctness(nodes_list):
    """
    Evaluate the actual network topology against a mathematically perfect Skip Graph
    This acts as an external observer and does not interfere with the algorithm logic
    """
    # Sort all nodes to evaluate the baseline level zero ring
    sorted_nodes = sorted(nodes_list, key=lambda n: n.id)
    l0_correct = 0
    for i, node in enumerate(sorted_nodes):
        expected_l0 = set()
        # Each node must be connected to its immediate numerical predecessor and successor
        if i > 0: expected_l0.add(sorted_nodes[i - 1])
        if i < len(sorted_nodes) - 1: expected_l0.add(sorted_nodes[i + 1])
        if expected_l0.issubset(node.neighbors): l0_correct += 1

    # Backup the current local ranges to prevent giving nodes global knowledge
    old_ranges = {u: u.ranges.copy() for u in nodes_list}

    total_expected, found_expected = 0, 0
    for u in nodes_list:
        # Pass the entire network to the range calculation
        u.calculate_ranges(nodes_list)
        for v in nodes_list:
            if u == v: continue
            c = u.shared_prefix_length(v)
            is_range_neighbor = any(u.ranges.get(i, (float('-inf'), float('inf')))[0] <= v.id <=
                                    u.ranges.get(i, (float('-inf'), float('inf')))[1] for i in range(c + 1))
            if is_range_neighbor:
                total_expected += 1
                # Check if the expected ideal edge actually exists in the current network state
                if v in u.neighbors: found_expected += 1

        # Restore the local state so the algorithm continues running strictly on local knowledge
        u.ranges = old_ranges[u]

    # Calculate correctness
    ratio_l0 = l0_correct / len(sorted_nodes)
    ratio_skip = found_expected / total_expected if total_expected > 0 else 0
    return (ratio_l0 * 50.0) + (ratio_skip * 50.0)


def run_skip_round(nodes_list):
    # track the network overhead per round
    id_map = {n.id: n for n in nodes_list}
    messages_this_round = 0

    # Phase 1 Ping Mechanism
    # Nodes send pings to neighbors. Missing pings indicate a crashed node.
    for u in nodes_list: u.pings_received.clear()
    for u in nodes_list:
        for v in u.neighbors:
            if u in nodes_list: v.pings_received.add(u)

    # Phase 2 Local State Update
    for u in nodes_list:
        # Remove dead nodes by intersecting current neighbors with received pings
        u.neighbors.intersection_update(u.pings_received)
        u.neighbors.update(u.pings_received)
        u.inbox_neighbors = set(u.neighbors)

        u.calculate_ranges(u.inbox_neighbors)
        u.calculate_stability(u.inbox_neighbors)
        u.F.clear()

        for v in u.inbox_neighbors:
            if v in u.current_stable:
                u.F[v] = 1
                # Track two messages for a mutual insert request if the stable node is new
                if v not in u.neighbors:
                    u.insert_requests.add(v)
                    v.insert_requests.add(u)
                    messages_this_round += 2
            else:
                u.F[v] = 0

    # Phase 3 Core Rule Evaluation
    for u in nodes_list:

        # Rule 1 Range Reduction
        # Introduce nodes that fall within the valid range of a stable neighbor
        for v in u.current_stable:
            for w in u.inbox_neighbors:
                if w != v and v.ranges.get(v.shared_prefix_length(w), (float('-inf'), float('inf')))[0] <= w.id <= \
                        v.ranges.get(v.shared_prefix_length(w), (float('-inf'), float('inf')))[1]:
                    # Track mutual insert requests generated by the range reduction
                    if w not in v.neighbors:
                        v.insert_requests.add(w)
                        w.insert_requests.add(v)
                        messages_this_round += 2

        # Rule 2 Forward Edges
        # Discard temporary connections and forward them to the best matching stable peer
        for v in u.inbox_neighbors:
            if u.F[v] == 0:
                u.remove_requests.add(v)
                best_w = None
                max_match = -1
                for w in u.current_stable:
                    match = w.shared_prefix_length(v)
                    if match > max_match or (match == max_match and (best_w is None or w.id < best_w.id)):
                        max_match, best_w = match, w
                if best_w:
                    best_w.insert_requests.add(v)
                    # Track a single forward request sent to the best matching peer
                    messages_this_round += 1

        # Rule 3 Local Closure
        # If structural stability changes introduce all known neighbors
        if u.stable_levels != u.prev_stable_levels:
            for v1 in u.inbox_neighbors:
                for v2 in u.inbox_neighbors:
                    if v1 != v2 and v2 not in v1.neighbors:
                        v1.insert_requests.add(v2)
                        # Track introduction messages sent to close local network gaps
                        messages_this_round += 1
        u.prev_stable_levels = u.stable_levels.copy()

    # Rule 4 Execute LSN Protocol per Prefix Group
    # Extract personas group them by prefix and run the LSN algorithm on them
    active_prefixes = set(
        u.vector[:i] for u in nodes_list for v in u.current_stable for i in range(1, u.shared_prefix_length(v) + 1))
    lsn_global_proposals = set()

    for pfx in active_prefixes:
        pfx_nodes = [n for n in nodes_list if n.vector[:len(pfx)] == pfx]
        if len(pfx_nodes) <= 1: continue

        lsn_nodes = [n.get_lsn_persona(pfx) for n in pfx_nodes]
        lsn_map = {n.id: n for n in lsn_nodes}

        current_lsn_edges = set(
            frozenset({lsn_map[u.id], lsn_map[v.id]}) for u in pfx_nodes for v in u.current_stable if
            v in pfx_nodes and v.vector[:len(pfx)] == pfx and u.id < v.id)

        for edge in run_lsn_round(lsn_nodes, current_lsn_edges):
            u_lsn, v_lsn = tuple(edge)
            lsn_global_proposals.add(frozenset({u_lsn.id, v_lsn.id}))

    # Map the resulting LSN edges back to the main Skip+ nodes
    for edge in lsn_global_proposals:
        u_id, v_id = tuple(edge)
        if u_id in id_map and v_id in id_map:
            if id_map[v_id] not in id_map[u_id].neighbors:
                id_map[u_id].insert_requests.add(id_map[v_id])
                id_map[v_id].insert_requests.add(id_map[u_id])
                # Track mutual edge proposals returned from the isolated LSN persona executions
                messages_this_round += 2

    # Final Phase Apply all queued requests simultaneously
    # This acts as the discrete time step and prevents concurrent modification errors
    for u in nodes_list:
        for v in u.remove_requests: u.neighbors.discard(v)
        for v in u.insert_requests:
            if v not in u.remove_requests: u.neighbors.add(v)
        u.insert_requests.clear()
        u.remove_requests.clear()

    return messages_this_round


#monte carlo evaluation
def run_single_overhead_simulation(N, seed, max_rounds=300):
    random.seed(seed)
    nodes_list = [SkipPlusNode(i) for i in range(N)]


    for i in range(N - 1):
        nodes_list[i].neighbors.add(nodes_list[i + 1])
        nodes_list[i + 1].neighbors.add(nodes_list[i])

    global_peak_degree = 0
    global_peak_messages = 0
    cleanup_timer = 0

    # Lemma 3.11
    required_cleanup_rounds = math.ceil(math.log2(N)) + 2

    for round_idx in range(1, max_rounds + 1):


        msgs_this_round = run_skip_round(nodes_list)

        if msgs_this_round > global_peak_messages:
            global_peak_messages = msgs_this_round


        current_max_degree = max(len(u.neighbors) for u in nodes_list)
        if current_max_degree > global_peak_degree:
            global_peak_degree = current_max_degree


        if calculate_global_correctness(nodes_list) >= 99.9:
            cleanup_timer += 1
            if cleanup_timer >= required_cleanup_rounds:
                final_stable_degree = max(len(u.neighbors) for u in nodes_list)
                return global_peak_degree, final_stable_degree, global_peak_messages
        else:
            cleanup_timer = 0

    final_stable_degree = max(len(u.neighbors) for u in nodes_list)
    return global_peak_degree, final_stable_degree, global_peak_messages


def evaluate_overhead():
    sizes = [16, 32, 64, 128]
    iterations = 5

    mean_peaks, std_peaks = [], []
    mean_finals, std_finals = [], []
    mean_messages, std_messages = [], []

    print("=" * 105)

    print("=" * 105)
    print(
        f"{'N':<5} | {'Peak Transient Deg. (O(N))':<28} | {'Final Stable Deg. (O(log n))':<30} | {'Peak Messages':<20}")
    print("-" * 105)

    for n in sizes:
        peaks, finals, messages = [], [], []
        for i in range(iterations):
            p, f, m = run_single_overhead_simulation(n, seed=42 + i)
            peaks.append(p)
            finals.append(f)
            messages.append(m)

        m_peak = statistics.mean(peaks)
        s_peak = statistics.stdev(peaks) if iterations > 1 else 0
        m_fin = statistics.mean(finals)
        s_fin = statistics.stdev(finals) if iterations > 1 else 0
        m_msg = statistics.mean(messages)
        s_msg = statistics.stdev(messages) if iterations > 1 else 0

        mean_peaks.append(m_peak)
        std_peaks.append(s_peak)
        mean_finals.append(m_fin)
        std_finals.append(s_fin)
        mean_messages.append(m_msg)
        std_messages.append(s_msg)

        print(
            f"{n:<5} | {m_peak:>6.1f} ± {s_peak:<19.1f} | {m_fin:>6.1f} ± {s_fin:<21.1f} | {m_msg:>8.1f} ± {s_msg:<10.1f}")

    print("=" * 105)
    plot_overhead(sizes, mean_peaks, std_peaks, mean_finals, std_finals, mean_messages, std_messages)


def plot_overhead(sizes, mean_peaks, std_peaks, mean_finals, std_finals, mean_messages, std_messages):
    # ===============================
    # PLOT 1: Peak vs Final Node Degree
    # ===============================
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax1.errorbar(sizes, mean_peaks, yerr=std_peaks, marker='o', color='#d62728',
                 linewidth=2.5, markersize=8, capsize=5, label='Peak Transient Degree (Healing Phase)')

    ax1.errorbar(sizes, mean_finals, yerr=std_finals, marker='s', color='#2ca02c',
                 linewidth=2.5, markersize=8, capsize=5, label='Final Stable Degree (Stabilized Network)')

    c_peak = mean_peaks[-1] / sizes[-1]
    ax1.plot(sizes, [c_peak * n for n in sizes], color='black', linestyle='-.', linewidth=2,
             label=r'Linear Overhead $\mathcal{O}(N)$')

    c_fin = mean_finals[-1] / math.log2(sizes[-1])
    ax1.plot(sizes, [c_fin * math.log2(n) for n in sizes], color='gray', linestyle=':', linewidth=2,
             label=r'Lemma 2.4 Bound $\mathcal{O}(\log n)$')

    ax1.set_title('SKIP+ Memory Overhead: Dynamic vs. Static Degree', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Number of Nodes (N) [Linear Scale]', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Maximum Node Degree', fontweight='bold', fontsize=12)
    ax1.set_xticks(sizes)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_facecolor('#fafafa')
    ax1.legend(fontsize=11, loc='upper left')
    fig1.tight_layout()

    # ===============================
    # PLOT 2: Message Overhead (Insert Requests)
    # ===============================
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.errorbar(sizes, mean_messages, yerr=std_messages, marker='^', color='#1f77b4',
                 linewidth=2.5, markersize=8, capsize=5, label='Peak Message Burst')

    ax2.set_title('SKIP+ Maximum Network Overhead per Round', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Number of Nodes (N) [Linear Scale]', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Total Insert Requests', fontweight='bold', fontsize=12)
    ax2.set_xticks(sizes)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_facecolor('#fafafa')
    ax2.legend(fontsize=11)
    fig2.tight_layout()

    plt.show()


if __name__ == '__main__':
    evaluate_overhead()