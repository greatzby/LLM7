# generate_alpine_no_direct_no_selfloop.py
# 目标：
# - 使用给定图与 stage_info 生成数据集
# - 去除自环（self-loop）：生成路径时禁止 node->node，且用无自环版图计算可达性
# - 不强制将“直接边”放入训练集；不额外添加任何“直达样本”
# - 为每个训练对生成 N 条随机有效路径；为每个测试对生成 1 条路径
# - 输出 train.txt、test.txt，并直接生成 train.bin、val.bin、meta.pkl 方便训练

import argparse
import os
import random
import pickle
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm


def remove_self_loops(G: nx.DiGraph) -> nx.DiGraph:
    H = G.copy()
    loops = list(nx.selfloop_edges(H))
    if loops:
        H.remove_edges_from(loops)
    return H


def precompute_reachability(G: nx.DiGraph) -> Dict[str, set]:
    cache = {}
    for node in G.nodes():
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
    从 source 到 target 生成一条随机有效路径：
    - 每一步只能走到当前点的后继；
    - 且该后继必须“最终可达 target”（在 ancestors(target) 内）；
    - 禁止自环步（下一步 != 当前节点）。
    """
    max_len = G.number_of_nodes() * max_hops_factor

    path_nodes = [source]
    cur = source
    hops = 0

    while cur != target:
        hops += 1
        if hops > max_len:
            # 极少发生（图为 DAG 时几乎不会），重试一条
            return generate_random_path(G, source, target, reachability_cache, max_hops_factor)

        succ = list(G.successors(cur))
        # 有效下一步：既在可达集合中，又不是自环
        valid = [v for v in succ if v != cur and v in reachability_cache[target]]

        if not valid:
            # 理论不应出现；做一次重试
            return generate_random_path(G, source, target, reachability_cache, max_hops_factor)

        nxt = random.choice(valid)
        path_nodes.append(nxt)
        cur = nxt

    return [int(x) for x in path_nodes]


def split_pairs_without_forcing_direct_edges(
    G: nx.DiGraph,
    reachability_cache: Dict[str, set],
    train_split_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], int]:
    """
    - 列出所有可达对 (u, v), u != v
    - 不再“强制将直接边加入训练”，而是统一按 train_split_ratio 随机划分
    - 返回：训练对列表、测试对列表、直接边数量（仅统计，用于打印）
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    train_pairs, test_pairs = [], []
    direct_edges = 0

    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            # 可达性：u 是 v 的祖先
            if u in reachability_cache[v]:
                if G.has_edge(u, v):
                    direct_edges += 1
                if rng.random() < train_split_ratio:
                    train_pairs.append((u, v))
                else:
                    test_pairs.append((u, v))

    return train_pairs, test_pairs, direct_edges


def count_stage_triplets(pairs: List[Tuple[str, str]], stages) -> Tuple[int, int, int]:
    S1, S2, S3 = stages
    a = b = c = 0
    for u, v in pairs:
        iu, iv = int(u), int(v)
        if iu in S1 and iv in S2:
            a += 1
        elif iu in S2 and iv in S3:
            b += 1
        elif iu in S1 and iv in S3:
            c += 1
    return a, b, c


def count_stage_triplets_from_samples(samples: List[List[int]], stages) -> Tuple[int, int, int]:
    S1, S2, S3 = stages
    a = b = c = 0
    for seq in samples:
        u, v = seq[0], seq[1]
        if u in S1 and v in S2:
            a += 1
        elif u in S2 and v in S3:
            b += 1
        elif u in S1 and v in S3:
            c += 1
    return a, b, c


def write_lines_txt(dataset: List[List[int]], file_path: str) -> None:
    with open(file_path, "w") as f:
        for row in dataset:
            f.write(" ".join(str(x) for x in row) + "\n")


def build_bins(
    train_samples: List[List[int]],
    val_samples: List[List[int]],
    out_dir: str,
    block_size: int,
) -> None:
    """
    生成 train.bin、val.bin 和 meta.pkl
    约定：
    - 将每条样本（变长）右侧用 0 进行 padding 到 block_size；
    - dtype 用 int32；
    - 在 meta.pkl 中记录 vocab_size（= 数据中出现的最大 token + 1）与 block_size、shape。
    注意：训练脚本需能处理 padding=0（通常借助 loss mask 或仅在指定位置计算损失）。
    """
    os.makedirs(out_dir, exist_ok=True)

    max_token = 0
    max_len = 0
    for seq in train_samples + val_samples:
        if not seq:
            continue
        max_token = max(max_token, max(seq))
        max_len = max(max_len, len(seq))

    if max_len > block_size:
        raise ValueError(f"Max sequence length {max_len} exceeds block_size {block_size}. "
                         f"Increase --block_size or reduce path length.")

    def pad_and_stack(samples: List[List[int]]) -> np.ndarray:
        if not samples:
            return np.zeros((0, block_size), dtype=np.int32)
        arr = np.zeros((len(samples), block_size), dtype=np.int32)
        for i, seq in enumerate(samples):
            L = min(len(seq), block_size)
            arr[i, :L] = np.array(seq[:L], dtype=np.int32)
        return arr

    train_arr = pad_and_stack(train_samples)
    val_arr = pad_and_stack(val_samples)

    # 保存二进制
    train_bin = os.path.join(out_dir, "train.bin")
    val_bin = os.path.join(out_dir, "val.bin")
    train_arr.tofile(train_bin)
    val_arr.tofile(val_bin)

    meta = {
        "vocab_size": int(max_token + 1),
        "block_size": int(block_size),
        "train_shape": tuple(train_arr.shape),
        "val_shape": tuple(val_arr.shape),
        "dtype": "int32",
        "pad_token": 0,
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("\n============================================================")
    print("Binary files written")
    print("============================================================")
    print(f"  Max sequence length: {max_len}")
    print(f"  Using block size:    {block_size}")
    print(f"  Vocab size:          {meta['vocab_size']}")
    print(f"  Train shape:         {train_arr.shape}")
    print(f"  Val shape:           {val_arr.shape}")
    print(f"  Saved:               {train_bin}, {val_bin}, meta.pkl")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset without self-loops and without adding direct-edge samples.")
    parser.add_argument("--input_graph", type=str, required=True, help="Path to input composition_graph.graphml")
    parser.add_argument("--stage_info", type=str, required=True, help="Path to stage_info.pkl")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--train_paths_per_pair", type=int, default=20, help="Number of random paths per training pair")
    parser.add_argument("--train_split_ratio", type=float, default=0.5, help="Train split ratio for reachable pairs")
    parser.add_argument("--block_size", type=int, default=32, help="Block size for .bin files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("============================================================")
    print("Generating ALPINE-style dataset (NO self-loops, NO direct-edge additions)")
    print("============================================================")
    print(f"Loading graph:      {args.input_graph}")
    print(f"Loading stage info: {args.stage_info}")

    G_orig = nx.read_graphml(args.input_graph)
    with open(args.stage_info, "rb") as f:
        stage_info = pickle.load(f)

    print(f"Original graph: {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges")
    G = remove_self_loops(G_orig)
    removed = G_orig.number_of_edges() - G.number_of_edges()
    if removed > 0:
        print(f"Removed self-loops: {removed}")
    else:
        print("No self-loops found in the input graph.")

    print("\nStep 1: Pre-computing reachability (using graph WITHOUT self-loops)...")
    reachability_cache = precompute_reachability(G)

    print("\nStep 2: Splitting reachable pairs (no forcing of direct edges into train)...")
    train_pairs, test_pairs, direct_edges_total = split_pairs_without_forcing_direct_edges(
        G, reachability_cache, args.train_split_ratio, args.seed
    )
    print(f"  Total reachable pairs: {len(train_pairs) + len(test_pairs)}")
    print(f"  Direct edges in graph: {direct_edges_total}")
    print(f"  Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    if "stages" in stage_info:
        S1, S2, S3 = stage_info["stages"]
        tr_s1s2, tr_s2s3, tr_s1s3 = count_stage_triplets(train_pairs, (S1, S2, S3))
        te_s1s2, te_s2s3, te_s1s3 = count_stage_triplets(test_pairs, (S1, S2, S3))
        print("\nPair distribution by stage (before path generation):")
        print(f"  Train - S1→S2: {tr_s1s2}, S2→S3: {tr_s2s3}, S1→S3: {tr_s1s3}")
        print(f"  Test  - S1→S2: {te_s1s2}, S2→S3: {te_s2s3}, S1→S3: {te_s1s3}")

    print(f"\nStep 3: Generating {args.train_paths_per_pair} random paths for each training pair...")
    train_set: List[List[int]] = []
    for (u, v) in tqdm(train_pairs, desc="Train pairs"):
        for _ in range(args.train_paths_per_pair):
            path = generate_random_path(G, u, v, reachability_cache)
            if path:
                # 样本格式：[src, tgt] + path_nodes
                train_set.append([int(u), int(v)] + path)

    print("\nStep 4: Generating 1 random path for each testing pair...")
    test_set: List[List[int]] = []
    for (u, v) in tqdm(test_pairs, desc="Test pairs"):
        path = generate_random_path(G, u, v, reachability_cache)
        if path:
            test_set.append([int(u), int(v)] + path)

    random.shuffle(train_set)
    random.shuffle(test_set)

    # 统计样本分布
    if "stages" in stage_info:
        S1, S2, S3 = stage_info["stages"]
        tr_s1s2_s, tr_s2s3_s, tr_s1s3_s = count_stage_triplets_from_samples(train_set, (S1, S2, S3))
        te_s1s2_s, te_s2s3_s, te_s1s3_s = count_stage_triplets_from_samples(test_set, (S1, S2, S3))

        print("\n============================================================")
        print("FINAL DATASET STATISTICS")
        print("============================================================")
        print(f"Training samples: {len(train_set)}")
        print(f"  S1→S2: {tr_s1s2_s}")
        print(f"  S2→S3: {tr_s2s3_s}")
        print(f"  S1→S3: {tr_s1s3_s}")
        print(f"Test samples: {len(test_set)}")
        print(f"  S1→S2: {te_s1s2_s}")
        print(f"  S2→S3: {te_s2s3_s}")
        print(f"  S1→S3: {te_s1s3_s}")

    # 写文本文件
    os.makedirs(args.output_dir, exist_ok=True)
    train_txt = os.path.join(args.output_dir, "train.txt")
    test_txt = os.path.join(args.output_dir, "test.txt")
    write_lines_txt(train_set, train_txt)
    write_lines_txt(test_set, test_txt)
    print(f"\nText datasets written to:\n  {train_txt}\n  {test_txt}")

    # 保存“无自环”的图与原 stage_info
    out_graph = os.path.join(args.output_dir, "composition_graph.graphml")
    out_stage = os.path.join(args.output_dir, "stage_info.pkl")
    nx.write_graphml(G, out_graph)  # 保存的是无自环版本，确保训练/评估一致
    with open(out_stage, "wb") as f:
        pickle.dump(stage_info, f)
    print(f"Also saved:\n  {out_graph}\n  {out_stage}")

    # 直接生成 .bin 与 meta.pkl，避免额外步骤
    print("\nStep 5: Building binary files (train.bin, val.bin, meta.pkl)...")
    build_bins(train_set, test_set, args.output_dir, args.block_size)

    print("\n✅ DONE. You can train now.")


if __name__ == "__main__":
    main()