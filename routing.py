import math
import random
import matplotlib

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


def run_skip_round(nodes_list):
    id_map = {n.id: n for n in nodes_list}

    # Phase 1: Ping Mechanism
    # Nodes send pings to neighbors. Missing pings indicate a crashed node.
    for u in nodes_list: u.pings_received.clear()
    for u in nodes_list:
        for v in u.neighbors:
            if u in nodes_list: v.pings_received.add(u)

    # Phase 2: Local State Update
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
                u.insert_requests.add(v)
                v.insert_requests.add(u)
            else:
                u.F[v] = 0

    # Phase 3: Core Rule Evaluation
    for u in nodes_list:

        # Rule 1: Range Reduction
        # Introduce nodes that fall within the valid range of a stable neighbor.
        for v in u.current_stable:
            for w in u.inbox_neighbors:
                if w != v and v.ranges.get(v.shared_prefix_length(w), (float('-inf'), float('inf')))[0] <= w.id <= \
                        v.ranges.get(v.shared_prefix_length(w), (float('-inf'), float('inf')))[1]:
                    v.insert_requests.add(w)
                    w.insert_requests.add(v)

        # Rule 2: Forward Edges
        # Discard temporary connections and forward them to the best matching stable peer.
        for v in u.inbox_neighbors:
            if u.F[v] == 0:
                u.remove_requests.add(v)
                best_w = None
                max_match = -1
                for w in u.current_stable:
                    match = w.shared_prefix_length(v)
                    if match > max_match or (match == max_match and (best_w is None or w.id < best_w.id)):
                        max_match, best_w = match, w
                if best_w: best_w.insert_requests.add(v)

        # Rule 3: Local Closure
        # If structural stability changes, introduce all known neighbors
        if u.stable_levels != u.prev_stable_levels:
            for v1 in u.inbox_neighbors:
                for v2 in u.inbox_neighbors:
                    if v1 != v2: v1.insert_requests.add(v2)
        u.prev_stable_levels = u.stable_levels.copy()

    # Rule 4: Execute LSN Protocol per Prefix Group
    # Extract personas, group them by prefix, and run the LSN algorithm on them.
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
            id_map[u_id].insert_requests.add(id_map[v_id])
            id_map[v_id].insert_requests.add(id_map[u_id])

    # Final Phase: Apply all queued requests simultaneously
    # This acts as the discrete time step and prevents concurrent modification errors.
    for u in nodes_list:
        for v in u.remove_requests: u.neighbors.discard(v)
        for v in u.insert_requests:
            if v not in u.remove_requests: u.neighbors.add(v)
        u.insert_requests.clear()
        u.remove_requests.clear()

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
    id_map = {n.id: n for n in nodes_list}

    # ==========================================
    # PHASE 1: PREPROCESSING
    # ==========================================
    for u in nodes_list:
        u.pings_received.clear()

    for u in nodes_list:
        for v in u.neighbors:
            v.pings_received.add(u)

    for u in nodes_list:
        u.neighbors.intersection_update(u.pings_received)
        u.neighbors.update(u.pings_received)
        u.inbox_neighbors = set(u.neighbors)

    for u in nodes_list:
        u.calculate_ranges(u.inbox_neighbors)
        u.calculate_stability(u.inbox_neighbors)

    for u in nodes_list:
        u.F.clear()
        for v in u.inbox_neighbors:
            if v in u.current_stable:
                u.F[v] = 1
                u.insert_requests.add(v)
                v.insert_requests.add(u)
            else:
                u.F[v] = 0

                # ==========================================
    # PHASE 2: SKIP+ REGELN 1-4
    # ==========================================
    for u in nodes_list:
        # RULE 1: Range Reduction
        for v in u.current_stable:
            for w in u.inbox_neighbors:
                if w != v:
                    c_vw = v.shared_prefix_length(w)
                    for i in range(c_vw + 1):
                        low_v, high_v = v.ranges.get(i, (float('-inf'), float('inf')))
                        if low_v <= w.id <= high_v:
                            v.insert_requests.add(w)
                            w.insert_requests.add(v)

        # RULE 2: Forward Edges
        for v in u.inbox_neighbors:
            if u.F[v] == 0:
                u.remove_requests.add(v)

                best_w = None
                max_match = -1
                for w in u.current_stable:
                    match = w.shared_prefix_length(v)
                    if match > max_match or (match == max_match and (best_w is None or w.id < best_w.id)):
                        max_match = match
                        best_w = w

                if best_w:
                    best_w.insert_requests.add(v)

        # RULE 3: Local Closure
        if u.stable_levels != u.prev_stable_levels:
            for v1 in u.inbox_neighbors:
                for v2 in u.inbox_neighbors:
                    if v1 != v2:
                        v1.insert_requests.add(v2)

        u.prev_stable_levels = u.stable_levels.copy()

    # ==========================================
    # RULE 4: LINEARIZE
    # ==========================================
    active_prefixes = set()
    for u in nodes_list:
        for v in u.current_stable:
            c = u.shared_prefix_length(v)
            for i in range(1, c + 1):
                active_prefixes.add(u.vector[:i])

    lsn_global_proposals = set()

    for pfx in active_prefixes:
        pfx_nodes = [n for n in nodes_list if n.vector[:len(pfx)] == pfx]
        if len(pfx_nodes) <= 1: continue

        lsn_nodes = [n.get_lsn_persona(pfx) for n in pfx_nodes]
        lsn_map = {n.id: n for n in lsn_nodes}

        current_lsn_edges = set()
        for u in pfx_nodes:
            for v in u.current_stable:
                if v in pfx_nodes and v.vector[:len(pfx)] == pfx and u.id < v.id:
                    current_lsn_edges.add(frozenset({lsn_map[u.id], lsn_map[v.id]}))

        next_lsn_edges = run_lsn_round(lsn_nodes, current_lsn_edges)
        for edge in next_lsn_edges:
            u_lsn, v_lsn = tuple(edge)
            lsn_global_proposals.add(frozenset({u_lsn.id, v_lsn.id}))

    for edge in lsn_global_proposals:
        u_id, v_id = tuple(edge)
        u, v = id_map[u_id], id_map[v_id]
        u.insert_requests.add(v)
        v.insert_requests.add(u)

    # ==========================================
    # PHASE 3: APPLY CHANGES
    # ==========================================
    for u in nodes_list:
        for v in u.remove_requests:
            u.neighbors.discard(v)

        for v in u.insert_requests:
            if v not in u.remove_requests:
                u.neighbors.add(v)

        u.insert_requests.clear()
        u.remove_requests.clear()

def greedy_route(start_node, target_id):
    """
    Simulate hypercubic routing by forwarding the message to the neighbor numerically closest to the target identifier
    """
    current = start_node
    hops = 0
    # Track visited nodes
    visited = set()

    # Continue forwarding the message until the exact target identifier is reached
    while current.id != target_id:
        visited.add(current.id)
        best_neighbor = current
        best_dist = abs(current.id - target_id)

        # Evaluate all active connections to find the peer closest to the destination
        for neighbor in current.neighbors:
            dist = abs(neighbor.id - target_id)
            # Strictly route towards the target by picking the unvisited neighbor with the smallest absolute distance
            if dist < best_dist and neighbor.id not in visited:
                best_dist = dist
                best_neighbor = neighbor

        if best_neighbor == current:
            return float('inf')

        # Move to the selected neighbor and increment the hop count
        current = best_neighbor
        hops += 1

    return hops




def evaluate_routing_efficiency():
    sizes = [32, 64, 128, 256, 512]
    avg_hops_history = []
    max_hops_history = []

    print("=" * 60)
    print("START ROUTING EFFICIENCY EVALUATION")
    print("=" * 60)

    for n in sizes:
        required_cleanup_rounds = math.ceil(math.log2(n)) + 2

        nodes_list = [SkipPlusNode(i) for i in range(n)]


        shuffled = nodes_list.copy()
        random.seed(42)
        random.shuffle(shuffled)
        for i in range(n):
            shuffled[i].neighbors.add(shuffled[(i + 1) % n])
            shuffled[(i + 1) % n].neighbors.add(shuffled[i])


        while calculate_global_correctness(nodes_list) < 99.9:
            run_skip_round(nodes_list)

        for _ in range(required_cleanup_rounds):
            run_skip_round(nodes_list)


        total_hops = 0
        local_max = 0
        tests = 0

        # Each node routes to every other node
        for source in nodes_list:
            for target in nodes_list:
                if source.id != target.id:
                    hops = greedy_route(source, target.id)
                    total_hops += hops
                    local_max = max(local_max, hops)
                    tests += 1

        avg_hops = total_hops / tests
        avg_hops_history.append(avg_hops)
        max_hops_history.append(local_max)

        print(f"N={n:<4} | Avg Hops: {avg_hops:.2f} | Max Hops: {local_max} | log2(N): {math.log2(n):.2f}")

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(sizes, avg_hops_history, marker='o', color='#1f77b4', linewidth=2.5, markersize=8,
            label='Average Path Length (Avg Hops)')
    ax.plot(sizes, max_hops_history, marker='^', color='#d62728', linewidth=2, markersize=7, linestyle='--',
            label='Maximum Path Length (Worst-Case)')

    # Theoretical Reference
    theo_log = [math.log2(x) for x in sizes]
    ax.plot(sizes, theo_log, color='black', linestyle=':', linewidth=2, label=rf'Theoretical Bound: $\log_2(n)$')

    ax.set_title('SKIP+ Routing Efficiency: Path Lengths in Stable Graph', fontweight='bold', fontsize=14)
    ax.set_xlabel('Number of Nodes (N)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Hops', fontweight='bold', fontsize=12)

    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.grid(True, which="both", linestyle='--', alpha=0.6)
    ax.set_facecolor('#fafafa')
    ax.legend(fontsize=11)

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate_routing_efficiency()