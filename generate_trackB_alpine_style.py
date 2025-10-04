#!/usr/bin/env python3
"""
generate_trackB_alpine_style.py
生成Track B风格的数据集，兼容训练脚本的格式要求
"""

import argparse
import os
import random
import pickle
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm


def remove_self_loops(G: nx.DiGraph) -> nx.DiGraph:
    """移除自环"""
    H = G.copy()
    loops = list(nx.selfloop_edges(H))
    if loops:
        H.remove_edges_from(loops)
        print(f"  Removed {len(loops)} self-loops")
    return H


def precompute_reachability(G: nx.DiGraph) -> Dict[str, set]:
    """预计算可达性"""
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
    """生成随机路径（无自环）"""
    max_len = G.number_of_nodes() * max_hops_factor
    
    path_nodes = [source]
    cur = source
    hops = 0
    
    while cur != target:
        hops += 1
        if hops > max_len:
            return generate_random_path(G, source, target, reachability_cache, max_hops_factor)
        
        succ = list(G.successors(cur))
        # 有效下一步：既在可达集合中，又不是自环
        valid = [v for v in succ if v != cur and v in reachability_cache[target]]
        
        if not valid:
            return generate_random_path(G, source, target, reachability_cache, max_hops_factor)
        
        nxt = random.choice(valid)
        path_nodes.append(nxt)
        cur = nxt
    
    return [int(x) for x in path_nodes]


def split_pairs_normal(
    G: nx.DiGraph,
    reachability_cache: Dict[str, set],
    train_split_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """正常分割（不强制直接边）"""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    train_pairs, test_pairs = [], []
    
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            if u in reachability_cache[v]:
                if rng.random() < train_split_ratio:
                    train_pairs.append((u, v))
                else:
                    test_pairs.append((u, v))
    
    return train_pairs, test_pairs


def count_stage_samples(samples: List[List[int]], stages) -> Dict[str, int]:
    """统计样本的阶段分布"""
    S1, S2, S3 = stages
    counts = {'S1->S2': 0, 'S2->S3': 0, 'S1->S3': 0, 'other': 0}
    
    for seq in samples:
        if len(seq) < 2:
            continue
        u, v = seq[0], seq[1]
        if u in S1 and v in S2:
            counts['S1->S2'] += 1
        elif u in S2 and v in S3:
            counts['S2->S3'] += 1
        elif u in S1 and v in S3:
            counts['S1->S3'] += 1
        else:
            counts['other'] += 1
    
    return counts


def write_txt(samples: List[List[int]], file_path: str):
    """写入txt文件"""
    with open(file_path, 'w') as f:
        for row in samples:
            # 格式：src tgt path_nodes
            f.write(' '.join(str(x) for x in row) + '\n')


def convert_to_binary_format(train_txt_path: str, test_txt_path: str, output_dir: str):
    """
    转换txt到训练脚本需要的二进制格式
    这个函数完全复制了您原来的转换逻辑
    """
    print("\n" + "="*60)
    print("Converting to binary format...")
    print("="*60)
    
    # 读取数据
    with open(train_txt_path, 'r') as f:
        train_data = f.read()
    with open(test_txt_path, 'r') as f:
        val_data = f.read()
    
    # 设置词汇表（90个节点 + PAD + 换行符）
    total_nodes = 90
    vocab_size = total_nodes + 2
    
    # 构建stoi和itos映射
    stoi = {str(i): i + 2 for i in range(total_nodes)}
    itos = {i + 2: str(i) for i in range(total_nodes)}
    stoi['[PAD]'] = 0
    itos[0] = '[PAD]'
    stoi['\n'] = 1
    itos[1] = '\n'
    
    def encode_string(s, stonum):
        """编码字符串"""
        ss = s.split(" ")
        return [stonum[ch] for ch in ss if ch in stonum]
    
    def get_block_size(s):
        """计算需要的block size"""
        split_text = s.split('\n')
        bs = 0
        for st in split_text:
            if st != "":
                enc_str = encode_string(st, stoi) + [1]  # 加上换行符
                bs = max(bs, len(enc_str))
        return bs
    
    def process_reasoning(s, block_size):
        """处理数据"""
        split_text = s.split('\n')
        ret = []
        for st in split_text:
            if st != "":
                enc_str = encode_string(st, stoi) + [1]  # 加上换行符
                # padding到block_size + 1
                ret += enc_str + [0] * (block_size + 1 - len(enc_str))
        return ret
    
    # 计算block size（对齐到32的倍数）
    max_len = max(get_block_size(train_data), get_block_size(val_data))
    block_size = ((max_len // 32) + 1) * 32
    
    print(f"  Max sequence length: {max_len}")
    print(f"  Using block_size: {block_size}")
    print(f"  Vocab size: {vocab_size}")
    
    # 编码数据
    train_ids = np.array(process_reasoning(train_data, block_size), dtype=np.uint16)
    val_ids = np.array(process_reasoning(val_data, block_size), dtype=np.uint16)
    
    print(f"  Train samples: {len(train_ids) // (block_size + 1)}")
    print(f"  Val samples: {len(val_ids) // (block_size + 1)}")
    
    # 保存bin文件
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))
    
    # 保存meta信息（训练脚本需要的格式）
    meta = {
        'unreachable': False,
        'simple_format': True,
        'block_size': block_size,
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"  ✅ Created: train.bin, val.bin, meta.pkl")
    
    return block_size, vocab_size


def main():
    parser = argparse.ArgumentParser(description="Generate Track B style dataset")
    parser.add_argument("--input_graph", type=str, required=True, help="Path to composition_graph.graphml")
    parser.add_argument("--stage_info", type=str, required=True, help="Path to stage_info.pkl")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--train_paths_per_pair", type=int, default=20, help="Paths per training pair")
    parser.add_argument("--train_split_ratio", type=float, default=0.5, help="Train split ratio")
    parser.add_argument("--with_direct_edges", action='store_true', help="Add direct edge shortest paths (ALPINE style)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*80)
    if args.with_direct_edges:
        print("Generating dataset WITH direct edge additions (ALPINE style)")
    else:
        print("Generating dataset WITHOUT direct edge additions (Track B style)")
    print("="*80)
    
    # 加载图和stage信息
    print(f"\nLoading graph: {args.input_graph}")
    G_orig = nx.read_graphml(args.input_graph)
    
    print(f"Loading stage info: {args.stage_info}")
    with open(args.stage_info, "rb") as f:
        stage_info = pickle.load(f)
    
    print(f"Original graph: {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges")
    
    # 移除自环
    G = remove_self_loops(G_orig)
    print(f"Graph after removing self-loops: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 预计算可达性
    print("\nStep 1: Pre-computing reachability...")
    reachability_cache = precompute_reachability(G)
    
    # 分割训练/测试对
    print("\nStep 2: Splitting pairs...")
    train_pairs, test_pairs = split_pairs_normal(G, reachability_cache, args.train_split_ratio, args.seed)
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")
    
    # 生成训练集路径
    print(f"\nStep 3: Generating {args.train_paths_per_pair} paths per training pair...")
    train_set = []
    
    # 如果要添加直接边（ALPINE风格）
    if args.with_direct_edges:
        print("  Adding direct edge shortest paths first...")
        direct_count = 0
        for u, v in train_pairs:
            if G.has_edge(u, v):
                # 添加最短路径 [src, tgt, src, tgt]
                train_set.append([int(u), int(v), int(u), int(v)])
                direct_count += 1
        print(f"  Added {direct_count} direct edge shortest paths")
    
    # 生成随机路径
    for u, v in tqdm(train_pairs, desc="Generating training paths"):
        for _ in range(args.train_paths_per_pair):
            path = generate_random_path(G, u, v, reachability_cache)
            if path:
                train_set.append([int(u), int(v)] + path)
    
    # 生成测试集路径
    print("\nStep 4: Generating 1 path per test pair...")
    test_set = []
    for u, v in tqdm(test_pairs, desc="Generating test paths"):
        path = generate_random_path(G, u, v, reachability_cache)
        if path:
            test_set.append([int(u), int(v)] + path)
    
    # 打乱顺序
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    # 统计
    if "stages" in stage_info:
        S1, S2, S3 = stage_info["stages"]
        train_counts = count_stage_samples(train_set, (S1, S2, S3))
        test_counts = count_stage_samples(test_set, (S1, S2, S3))
        
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Training samples: {len(train_set)}")
        for k, v in train_counts.items():
            if v > 0:
                print(f"  {k}: {v} ({v/len(train_set)*100:.1f}%)")
        
        print(f"\nTest samples: {len(test_set)}")
        for k, v in test_counts.items():
            if v > 0:
                print(f"  {k}: {v} ({v/len(test_set)*100:.1f}%)")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 写入txt文件
    train_txt = os.path.join(args.output_dir, "train.txt")
    test_txt = os.path.join(args.output_dir, "test.txt")
    write_txt(train_set, train_txt)
    write_txt(test_set, test_txt)
    print(f"\nText files saved to {args.output_dir}/")
    
    # 保存图和stage信息
    out_graph = os.path.join(args.output_dir, "composition_graph.graphml")
    out_stage = os.path.join(args.output_dir, "stage_info.pkl")
    nx.write_graphml(G, out_graph)
    with open(out_stage, "wb") as f:
        pickle.dump(stage_info, f)
    
    # 转换为二进制格式
    block_size, vocab_size = convert_to_binary_format(train_txt, test_txt, args.output_dir)
    
    print("\n" + "="*80)
    print("✅ DATASET GENERATION COMPLETE!")
    print("="*80)
    print(f"Output directory: {args.output_dir}/")
    print(f"Files created:")
    print(f"  - train.txt, test.txt (text format)")
    print(f"  - train.bin, val.bin (binary format)")
    print(f"  - meta.pkl (metadata)")
    print(f"  - composition_graph.graphml")
    print(f"  - stage_info.pkl")
    print("\nYou can now train with:")
    print(f"python train_compositiontrack_final_fixed.py \\")
    print(f"    --data_dir {args.output_dir} \\")
    print(f"    --train_file train.bin \\")
    print(f"    --n_embd 92 --batch_size 128")


if __name__ == "__main__":
    main()