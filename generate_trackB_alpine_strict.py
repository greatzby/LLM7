#!/usr/bin/env python3
"""
generate_trackB_alpine_strict.py
严格遵循 ALPINE 论文风格生成数据集：
- 不移除自环，按原图构造可达对与路径
- 直接边 (u->v) 100% 放入训练集
- 对每个训练集中的直接边注入一条直达样本 [u, v, u, v]
- 其他可达对按 train_split_ratio 随机划分 train/test
- 训练对各生成 N 条随机有效路径；测试对各生成 1 条随机路径
- 输出：train.txt、test.txt、composition_graph.graphml、stage_info.pkl
- 同时生成：train.bin、val.bin、meta.pkl（含 stoi/itos），可直接用于训练脚本
"""

import argparse
import os
import random
import pickle
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm


# -----------------------------
# 路径生成与可达性
# -----------------------------
def precompute_reachability(G: nx.DiGraph) -> Dict[str, set]:
    """
    预计算可达性：对每个 target，缓存其所有 ancestors，并把自身加入集合。
    """
    cache: Dict[str, set] = {}
    for node in tqdm(G.nodes(), desc="Caching reachability"):
        anc = nx.ancestors(G, node)
        anc.add(node)
        cache[node] = anc
    return cache


def generate_random_path(
    G: nx.DiGraph,
    source: str,
    target: str,
    reachability_cache: Dict[str, set],
    max_hops_factor: int = 2,
) -> List[int]:
    """
    生成一条从 source 到 target 的随机有效路径。
    严格按你给的 ALPINE 风格：
    - 每步只在当前节点的后继中挑选
    - 下一步必须属于 reachability_cache[target]
    - 不禁止走自环（如果图里有自环边，则可能停留在当前点）
    - 若长度超过阈值，则重试
    """
    path = [source]
    cur = source
    max_len = G.number_of_nodes() * max_hops_factor

    while cur != target:
        if len(path) > max_len:
            # 超长重试
            return generate_random_path(G, source, target, reachability_cache, max_hops_factor)

        succ = list(G.successors(cur))
        valid = [v for v in succ if v in reachability_cache[target]]

        if not valid:
            # 理论不应发生（因从可达对开始），但仍做健壮性处理
            return generate_random_path(G, source, target, reachability_cache, max_hops_factor)

        nxt = random.choice(valid)
        path.append(nxt)
        cur = nxt

    return [int(x) for x in path]


# -----------------------------
# 严格 ALPINE 的对划分与样本生成
# -----------------------------
def find_all_reachable_pairs(G: nx.DiGraph, reachability_cache: Dict[str, set]) -> List[Tuple[str, str]]:
    pairs = []
    nodes = list(G.nodes())
    for s in nodes:
        for t in nodes:
            if s != t and s in reachability_cache[t]:
                pairs.append((s, t))
    return pairs


def split_pairs_alpine_strict(
    G: nx.DiGraph,
    all_pairs: List[Tuple[str, str]],
    train_split_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], int]:
    """
    严格 ALPINE：
    - 直接边 (u->v) 全部放入训练集
    - 其他可达对按比例随机分配
    返回：train_pairs, test_pairs, 直接边数量
    """
    rng = random.Random(seed)
    train_pairs, test_pairs = [], []
    direct_count = 0

    for (u, v) in all_pairs:
        if G.has_edge(u, v):
            train_pairs.append((u, v))
            direct_count += 1
        else:
            if rng.random() < train_split_ratio:
                train_pairs.append((u, v))
            else:
                test_pairs.append((u, v))

    return train_pairs, test_pairs, direct_count


def create_alpine_style_dataset(
    G: nx.DiGraph,
    num_train_paths_per_pair: int,
    train_split_ratio: float,
    stage_info: dict,
    seed: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    print("\n" + "=" * 70)
    print("ALPINE STRICT MODE - Following paper exactly")
    print("=" * 70)

    # Step 1: reachability
    print("\nStep 1: Pre-computing reachability for all nodes...")
    reachability_cache = precompute_reachability(G)

    # Step 2: all reachable (s,t)
    print("\nStep 2: Finding all reachable (source, target) pairs...")
    all_pairs = find_all_reachable_pairs(G, reachability_cache)
    print(f"Found {len(all_pairs)} total reachable pairs.")

    # Step 3: split with 'direct edges -> train'
    print(f"\nStep 3: Splitting pairs (ALPINE rules: direct edges→train, others→{train_split_ratio:.0%})...")
    train_pairs, test_pairs, direct_edges = split_pairs_alpine_strict(
        G, all_pairs, train_split_ratio, seed
    )
    print(f"  Direct edges (forced to training): {direct_edges}")
    print(f"  Final split - Training pairs: {len(train_pairs)}, Testing pairs: {len(test_pairs)}")

    # Pair-level stats by stage
    if stage_info and "stages" in stage_info:
        S1, S2, S3 = stage_info["stages"]
        def count_pairs(pairs):
            a = b = c = 0
            for u, v in pairs:
                iu, iv = int(u), int(v)
                if iu in S1 and iv in S2: a += 1
                elif iu in S2 and iv in S3: b += 1
                elif iu in S1 and iv in S3: c += 1
            return a, b, c

        tr_a, tr_b, tr_c = count_pairs(train_pairs)
        te_a, te_b, te_c = count_pairs(test_pairs)
        print("\n" + "-" * 50)
        print("PAIR DISTRIBUTION (before path generation):")
        print("-" * 50)
        print(f"  S1→S2: Train={tr_a}, Test={te_a}")
        print(f"  S2→S3: Train={tr_b}, Test={te_b}")
        print(f"  S1→S3: Train={tr_c}, Test={te_c}")
        total_s1s3 = tr_c + te_c
        if total_s1s3 > 0:
            print(f"\n  🔍 KEY METRIC: S1→S3 in training = {tr_c}/{total_s1s3} = {tr_c/total_s1s3:.1%}")
        print("-" * 50)

    # Step 4: build samples
    print(f"\nStep 4: Generating {num_train_paths_per_pair} random paths for each training pair...")

    train_set: List[List[int]] = []
    test_set: List[List[int]] = []

    # Inject direct paths for direct edges in training pairs
    direct_paths_added = 0
    for (u, v) in tqdm(train_pairs, desc="Generating training data"):
        if G.has_edge(u, v):
            train_set.append([int(u), int(v), int(u), int(v)])
            direct_paths_added += 1
        # N random paths per pair
        for _ in range(num_train_paths_per_pair):
            path = generate_random_path(G, u, v, reachability_cache)
            if path:
                train_set.append([int(u), int(v)] + path)
    print(f"  Added {direct_paths_added} direct paths for direct edges")

    print("\nStep 5: Generating 1 random path for each testing pair...")
    for (u, v) in tqdm(test_pairs, desc="Generating testing data"):
        path = generate_random_path(G, u, v, reachability_cache)
        if path:
            test_set.append([int(u), int(v)] + path)

    random.shuffle(train_set)
    random.shuffle(test_set)

    # Sample-level stats
    if stage_info and "stages" in stage_info:
        S1, S2, S3 = stage_info["stages"]
        def count_samples(samples):
            a = b = c = 0
            for s in samples:
                uu, vv = s[0], s[1]
                if uu in S1 and vv in S2: a += 1
                elif uu in S2 and vv in S3: b += 1
                elif uu in S1 and vv in S3: c += 1
            return a, b, c

        tr_a, tr_b, tr_c = count_samples(train_set)
        te_a, te_b, te_c = count_samples(test_set)

        print("\n" + "=" * 70)
        print("FINAL DATASET STATISTICS")
        print("=" * 70)
        print(f"Training samples: {len(train_set)}")
        print(f"  S1→S2: {tr_a}")
        print(f"  S2→S3: {tr_b}")
        print(f"  S1→S3: {tr_c}")
        print(f"\nTest samples: {len(test_set)}")
        print(f"  S1→S2: {te_a}")
        print(f"  S2→S3: {te_b}")
        print(f"  S1→S3: {te_c}")

    return train_set, test_set


# -----------------------------
# 文本与二进制写出（含 stoi/itos）
# -----------------------------
def write_txt(samples: List[List[int]], file_path: str) -> None:
    with open(file_path, "w") as f:
        for row in samples:
            f.write(" ".join(str(x) for x in row) + "\n")


def convert_to_binary_format(train_txt_path: str, test_txt_path: str, output_dir: str, max_node_id: int):
    """
    将 train.txt / test.txt 转为训练脚本需要的二进制：
    - vocab：节点 0..max_node_id 映射到 2..(N+1)，[PAD]=0，'\n'=1
    - block_size：向上取整到 32 的倍数
    - 数据按 (block_size+1) 的步长铺平存储（含行末 '\n' 与右侧 PAD）
    """
    print("\n" + "=" * 60)
    print("Converting to binary format...")
    print("=" * 60)

    with open(train_txt_path, "r") as f:
        train_data = f.read()
    with open(test_txt_path, "r") as f:
        val_data = f.read()

    total_nodes = max_node_id + 1
    vocab_size = total_nodes + 2  # + [PAD], + '\n'

    # stoi / itos
    stoi = {str(i): i + 2 for i in range(total_nodes)}
    itos = {i + 2: str(i) for i in range(total_nodes)}
    stoi["[PAD]"] = 0; itos[0] = "[PAD]"
    stoi["\n"] = 1;    itos[1] = "\n"

    def encode_string(s: str, stonum: dict):
        tokens = s.split(" ")
        out = []
        for t in tokens:
            if t == "":
                continue
            if t not in stonum:
                raise ValueError(f"Unknown token '{t}' not in vocabulary 0..{total_nodes-1}.")
            out.append(stonum[t])
        return out

    def get_block_size(s: str):
        bs = 0
        for st in s.split("\n"):
            if st != "":
                enc = encode_string(st, stoi) + [1]  # 追加换行
                bs = max(bs, len(enc))
        return bs

    def process_reasoning(s: str, block_size: int):
        ret = []
        for st in s.split("\n"):
            if st != "":
                enc = encode_string(st, stoi) + [1]
                ret += enc + [0] * (block_size + 1 - len(enc))
        return ret

    max_len = max(get_block_size(train_data), get_block_size(val_data))
    block_size = ((max_len + 31) // 32) * 32  # 向上取 32 的倍数

    print(f"  Max sequence length: {max_len}")
    print(f"  Using block_size:    {block_size}")
    print(f"  Vocab size:          {vocab_size}")

    train_ids = np.array(process_reasoning(train_data, block_size), dtype=np.uint16)
    val_ids   = np.array(process_reasoning(val_data,   block_size), dtype=np.uint16)

    print(f"  Train samples: {len(train_ids) // (block_size + 1)}")
    print(f"  Val samples:   {len(val_ids)   // (block_size + 1)}")

    train_ids.tofile(os.path.join(output_dir, "train.bin"))
    val_ids.tofile(os.path.join(output_dir, "val.bin"))

    meta = {
        "unreachable": False,
        "simple_format": True,
        "block_size": block_size,
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("  ✅ Created: train.bin, val.bin, meta.pkl")
    return block_size, vocab_size


# -----------------------------
# 主流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate dataset using ALPINE's strict rules (Track B graph).")
    parser.add_argument("--input_graph", type=str, required=True, help="Path to composition_graph.graphml")
    parser.add_argument("--stage_info", type=str, required=True, help="Path to stage_info.pkl")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--train_paths_per_pair", type=int, default=20, help="Number of random paths per training pair")
    parser.add_argument("--train_split_ratio", type=float, default=0.5, help="Training ratio for non-direct reachable pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading graph from: {args.input_graph}")
    G = nx.read_graphml(args.input_graph)

    print(f"Loading stage info from: {args.stage_info}")
    with open(args.stage_info, "rb") as f:
        stage_info = pickle.load(f)

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 生成数据（严格 ALPINE）
    train_set, test_set = create_alpine_style_dataset(
        G=G,
        num_train_paths_per_pair=args.train_paths_per_pair,
        train_split_ratio=args.train_split_ratio,
        stage_info=stage_info,
        seed=args.seed,
    )

    # 输出目录与文本写出
    os.makedirs(args.output_dir, exist_ok=True)
    train_txt = os.path.join(args.output_dir, "train.txt")
    test_txt  = os.path.join(args.output_dir, "test.txt")
    write_txt(train_set, train_txt)
    write_txt(test_set,  test_txt)
    print(f"\nText datasets written to:\n  {train_txt}\n  {test_txt}")

    # 保存原始图与 stage_info（不移除自环）
    out_graph = os.path.join(args.output_dir, "composition_graph.graphml")
    out_stage = os.path.join(args.output_dir, "stage_info.pkl")
    nx.write_graphml(G, out_graph)
    with open(out_stage, "wb") as f:
        pickle.dump(stage_info, f)
    print(f"Also saved:\n  {out_graph}\n  {out_stage}")

    # 转为二进制
    max_node_id = max(int(n) for n in G.nodes())
    print("\nStep 6: Converting text to binary (.bin + meta.pkl)...")
    convert_to_binary_format(train_txt, test_txt, args.output_dir, max_node_id)

    print("\n" + "=" * 80)
    print("✅ DATASET GENERATION COMPLETE (ALPINE STRICT)!")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}/")
    print("Files created:")
    print("  - train.txt, test.txt (text)")
    print("  - train.bin, val.bin (binary)")
    print("  - meta.pkl (with stoi/itos)")
    print("  - composition_graph.graphml (original, with self-loops if any)")
    print("  - stage_info.pkl")


if __name__ == "__main__":
    main()