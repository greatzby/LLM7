"""
Alpha-Mixing Experiment: Dual-Track Analysis (Single-Pool, Fixed)

Track A: Fixed marginal ratios r; alpha controls the share drawn via dependent sampling (f_nat)
         vs independent per-class sampling; then top up to exactly match r (largest remainder).
Track B: Marginal ratios drift from f_nat to r as alpha increases.

Key fixes:
- Correct reachability direction (use source in reachability[target]).
- Unify node ids as str.
- Compute f_nat from reachable pairs (nx.has_path), not from generated paths.
- Use largest remainder allocation to reduce rounding bias.
"""

import os
import random
import pickle
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import networkx as nx
from itertools import product


def largest_remainder_alloc(ratio: Dict[str, float], total: int) -> Dict[str, int]:
    keys = list(ratio.keys())
    targets = np.array([ratio[k] * total for k in keys], dtype=float)
    floors = np.floor(targets).astype(int)
    remainder = targets - floors
    missing = total - int(floors.sum())
    if missing > 0:
        order = np.argsort(-remainder)  # descending
        floors[order[:missing]] += 1
    return {k: int(v) for k, v in zip(keys, floors)}


class ALPINEPathGenerator:
    """Guided random walk with correct reachability semantics, node ids are str."""
    def __init__(self, G: nx.DiGraph):
        self.G = G
        self.reach_pred = {}  # reach_pred[t] = set of nodes that can reach t
        self._compute_reachability()

    def _compute_reachability(self):
        print("Computing transitive closure (predecessors)...")
        TC = nx.transitive_closure(self.G)
        for node in self.G.nodes():
            self.reach_pred[node] = set(TC.predecessors(node))

    def can_reach(self, s: str, t: str) -> bool:
        if s == t:
            return True
        return s in self.reach_pred.get(t, set())

    def random_walk(self, source: str, target: str, max_restarts: int = 20) -> List[str] | None:
        source = str(source)
        target = str(target)
        if not self.can_reach(source, target):
            return None

        for _ in range(max_restarts):
            path = [source]
            current = source
            visited = {current}
            max_steps = len(self.G) * 2

            for _ in range(max_steps):
                if current == target:
                    return path
                neighbors = list(self.G.successors(current))
                valid_next = [n for n in neighbors if (n == target) or (n in self.reach_pred.get(target, set()))]
                # 避免原地绕圈（DAG 中影响不大，但更稳）
                valid_next = [n for n in valid_next if n not in visited]
                if not valid_next:
                    break  # restart
                nxt = random.choice(valid_next)
                path.append(nxt)
                visited.add(nxt)
                current = nxt
            # restart
        return None


class AlphaMixingGenerator:
    def __init__(self, graph_dir='standardized_alpine_90_seed42', paths_per_pair=20, seed=42):
        self.graph_dir = graph_dir
        self.paths_per_pair = paths_per_pair
        random.seed(seed)
        np.random.seed(seed)
        self._load_graph_and_stages()
        self.pathgen = ALPINEPathGenerator(self.G)
        self._enumerate_reachable_pairs()  # for f_nat
        self._build_path_pool()            # single pool of paths

    def _load_graph_and_stages(self):
        graph_file = os.path.join(self.graph_dir, 'composition_graph.graphml')
        self.G = nx.read_graphml(graph_file)  # node ids as str
        with open(os.path.join(self.graph_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        self.S1 = [str(x) for x in stage_info['stages'][0]]
        self.S2 = [str(x) for x in stage_info['stages'][1]]
        self.S3 = [str(x) for x in stage_info['stages'][2]]
        print(f"Loaded graph: |V|={self.G.number_of_nodes()}, |E|={self.G.number_of_edges()}")
        print(f"S1={len(self.S1)}, S2={len(self.S2)}, S3={len(self.S3)}")

    def _enumerate_reachable_pairs(self):
        self.pairs = {'12': [], '23': [], '13': []}
        # S1->S2
        for s1, s2 in product(self.S1, self.S2):
            if self.pathgen.can_reach(s1, s2):
                self.pairs['12'].append((s1, s2))
        # S2->S3
        for s2, s3 in product(self.S2, self.S3):
            if self.pathgen.can_reach(s2, s3):
                self.pairs['23'].append((s2, s3))
        # S1->S3 (reachable at graph level; S2-pass-through enforced when generating paths)
        for s1, s3 in product(self.S1, self.S3):
            if self.pathgen.can_reach(s1, s3):
                self.pairs['13'].append((s1, s3))

        self.N_12, self.N_23, self.N_13 = len(self.pairs['12']), len(self.pairs['23']), len(self.pairs['13'])
        self.N_total = self.N_12 + self.N_23 + self.N_13
        if self.N_total == 0:
            raise ValueError("No reachable pairs in this graph.")
        self.f_nat = {'12': self.N_12 / self.N_total,
                      '23': self.N_23 / self.N_total,
                      '13': self.N_13 / self.N_total}
        print(f"Reachable pairs: N_12={self.N_12}, N_23={self.N_23}, N_13={self.N_13}, N_total={self.N_total}")
        print(f"Natural fractions (f_nat): 12={self.f_nat['12']:.1%}, 23={self.f_nat['23']:.1%}, 13={self.f_nat['13']:.1%}")

    def _build_path_pool(self):
        """Build a single central pool of paths: list of dicts with src,tgt,path,type."""
        print("\n" + "="*60)
        print("Building central path pool...")
        print("="*60)
        self.pool_by_type: Dict[str, List[dict]] = {'12': [], '23': [], '13': []}

        def gen_for_pairs(pair_list: List[Tuple[str, str]], ptype: str, require_s2=False):
            added = 0
            for src, tgt in pair_list:
                trials = 0
                collected = 0
                while collected < self.paths_per_pair and trials < self.paths_per_pair * 5:
                    trials += 1
                    p = self.pathgen.random_walk(src, tgt)
                    if p is None:
                        continue
                    if require_s2:
                        # require at least one S2 node strictly between src and tgt
                        if not any(n in set(self.S2) for n in p[1:-1]):
                            continue
                    self.pool_by_type[ptype].append({'src': src, 'tgt': tgt, 'path': p, 'type': ptype})
                    collected += 1
                    added += 1
            return added

        a12 = gen_for_pairs(self.pairs['12'], '12', require_s2=False)
        a23 = gen_for_pairs(self.pairs['23'], '23', require_s2=False)
        a13 = gen_for_pairs(self.pairs['13'], '13', require_s2=True)

        print(f"Pool sizes: 12={len(self.pool_by_type['12'])}, 23={len(self.pool_by_type['23'])}, 13={len(self.pool_by_type['13'])}")

    def _sample_dependent(self, n_by_type: Dict[str, int]) -> List[dict]:
        """Sample by f_nat shares; sampling with replacement from each type pool."""
        out = []
        for t in ['12', '23', '13']:
            n = n_by_type.get(t, 0)
            if n <= 0 or len(self.pool_by_type[t]) == 0:
                continue
            out += random.choices(self.pool_by_type[t], k=n)
        return out

    def _count_by_type(self, records: List[dict]) -> Dict[str, int]:
        c = {'12': 0, '23': 0, '13': 0}
        for r in records:
            c[r['type']] += 1
        return c

    def _to_lines(self, records: List[dict]) -> List[List[str]]:
        lines = []
        for r in records:
            path = r['path']
            if not (path[0] == r['src'] and path[-1] == r['tgt']):
                continue
            # line: src tgt path_len node0 ... nodeL
            lines.append([r['src'], r['tgt'], str(len(path))] + path)
        return lines

    def generate_track_A_dataset(self, alpha: float, dataset_size: int, r: Dict[str, float]) -> List[List[str]]:
        """
        Track A: final marginal ratios ≈ r for all alpha.
        - First draw (1-alpha)*N using f_nat (dependent).
        - Then top up per class independently to hit r (independent).
        """
        # part 1: dependent
        n_dep_total = int(round((1 - alpha) * dataset_size))
        n_dep = largest_remainder_alloc(self.f_nat, n_dep_total)
        dep_records = self._sample_dependent(n_dep)

        # part 2: top up to r
        target_counts = largest_remainder_alloc(r, dataset_size)
        cur_counts = self._count_by_type(dep_records)
        add_records = []
        for t in ['12', '23', '13']:
            need = max(0, target_counts[t] - cur_counts.get(t, 0))
            if need > 0 and len(self.pool_by_type[t]) > 0:
                add_records += random.choices(self.pool_by_type[t], k=need)

        records = dep_records + add_records
        random.shuffle(records)
        return self._to_lines(records)

    def generate_track_B_dataset(self, alpha: float, dataset_size: int, r: Dict[str, float]) -> List[List[str]]:
        """
        Track B: drift from f_nat to r as alpha increases.
        - Draw (1-alpha)*N by f_nat (dependent)
        - Draw alpha*N by r (independent)
        """
        n_dep_total = int(round((1 - alpha) * dataset_size))
        n_ind_total = dataset_size - n_dep_total
        n_dep = largest_remainder_alloc(self.f_nat, n_dep_total)
        n_ind = largest_remainder_alloc(r, n_ind_total)

        dep_records = self._sample_dependent(n_dep)
        ind_records = self._sample_dependent(n_ind)  # independent per-class quotas
        records = dep_records + ind_records
        random.shuffle(records)
        return self._to_lines(records)

    def compute_stats(self, lines: List[List[str]]) -> Dict:
        stats = {'total': len(lines), 'S1->S2': 0, 'S2->S3': 0, 'S1->S3': 0}
        S1, S2, S3 = set(self.S1), set(self.S2), set(self.S3)
        for line in lines:
            src, tgt = line[0], line[1]
            if src in S1 and tgt in S2:
                stats['S1->S2'] += 1
            elif src in S2 and tgt in S3:
                stats['S2->S3'] += 1
            elif src in S1 and tgt in S3:
                stats['S1->S3'] += 1
        tot = max(1, stats['total'])
        stats['fractions'] = {
            'S1->S2': stats['S1->S2'] / tot,
            'S2->S3': stats['S2->S3'] / tot,
            'S1->S3': stats['S1->S3'] / tot,
        }
        return stats


def save_txt(lines: List[List[str]], path: str):
    with open(path, 'w') as f:
        for ln in lines:
            f.write(' '.join(ln) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Alpha-Mixing Dual-Track Experiment')
    parser.add_argument('--graph_dir', default='standardized_alpine_90_seed42')
    parser.add_argument('--paths_per_pair', type=int, default=20)
    parser.add_argument('--dataset_size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='alpha_mixing_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("="*60)
    print("Initializing Alpha-Mixing Dual-Track Experiment")
    print("="*60)
    gen = AlphaMixingGenerator(args.graph_dir, args.paths_per_pair, args.seed)

    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    r = {'12': 0.4, '23': 0.4, '13': 0.2}

    summary = {
        'alpha_values': alpha_values,
        'natural_fractions': gen.f_nat,
        'fixed_ratio': r,
        'N_pairs': {'N_12': gen.N_12, 'N_23': gen.N_23, 'N_13': gen.N_13, 'N_total': gen.N_total},
        'track_A': [],
        'track_B': [],
    }

    print("\n" + "="*60)
    print("Generating datasets across alpha")
    print("="*60)
    for a in alpha_values:
        print(f"\n--- alpha = {a:.1f} ---")

        # Track A
        lines_A = gen.generate_track_A_dataset(a, args.dataset_size, r)
        stats_A = gen.compute_stats(lines_A)
        save_txt(lines_A, os.path.join(args.output_dir, f"track_A_alpha_{a:.1f}.txt"))
        summary['track_A'].append(stats_A)
        print(f"Track A fractions: S1->S2={stats_A['fractions']['S1->S2']:.1%}, "
              f"S2->S3={stats_A['fractions']['S2->S3']:.1%}, "
              f"S1->S3={stats_A['fractions']['S1->S3']:.1%} "
              f"(n={stats_A['total']})")

        # Track B
        lines_B = gen.generate_track_B_dataset(a, args.dataset_size, r)
        stats_B = gen.compute_stats(lines_B)
        save_txt(lines_B, os.path.join(args.output_dir, f"track_B_alpha_{a:.1f}.txt"))
        summary['track_B'].append(stats_B)
        print(f"Track B fractions: S1->S2={stats_B['fractions']['S1->S2']:.1%}, "
              f"S2->S3={stats_B['fractions']['S2->S3']:.1%}, "
              f"S1->S3={stats_B['fractions']['S1->S3']:.1%} "
              f"(n={stats_B['total']})")

    with open(os.path.join(args.output_dir, 'experiment_summary.pkl'), 'wb') as f:
        pickle.dump(summary, f)

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"f_nat: 12={gen.f_nat['12']:.1%}, 23={gen.f_nat['23']:.1%}, 13={gen.f_nat['13']:.1%}")
    print(f"fixed r: 12={r['12']:.1%}, 23={r['23']:.1%}, 13={r['13']:.1%}")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()