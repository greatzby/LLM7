#!/usr/bin/env python3
"""
add_alpine_direct_edges.py
为Track B α=0数据集添加ALPINE风格的直接边处理
"""

import os
import pickle
import shutil
import networkx as nx
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def load_graph_and_stages(data_dir):
    """加载图和stage信息"""
    # 加载图
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    
    # 加载stage信息
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    
    S1, S2, S3 = stage_info['stages']
    return G, S1, S2, S3

def identify_direct_edges(G):
    """识别所有直接边"""
    direct_edges = set()
    for edge in G.edges():
        # 转换为整数对
        if isinstance(edge[0], str):
            source, target = int(edge[0]), int(edge[1])
        else:
            source, target = edge[0], edge[1]
        direct_edges.add((source, target))
    return direct_edges

def process_trackB_dataset(input_file, output_file, direct_edges, S1, S2, S3):
    """处理Track B数据集，添加ALPINE风格的直接边"""
    
    print("\n" + "="*60)
    print("Processing Track B α=0 dataset...")
    print("="*60)
    
    # 读取原始数据
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Original dataset: {len(lines)} paths")
    
    # 统计原始数据中的(source, target)对
    pair_counts = defaultdict(int)
    original_paths = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            source, target = int(parts[0]), int(parts[1])
            pair_counts[(source, target)] += 1
            original_paths.append(parts)
    
    # 统计直接边在原数据中的出现情况
    direct_edge_coverage = {}
    s1s2_direct = 0
    s2s3_direct = 0
    s1s1_direct = 0
    s2s2_direct = 0
    s3s3_direct = 0
    
    for (source, target) in direct_edges:
        if source in S1 and target in S2:
            s1s2_direct += 1
        elif source in S2 and target in S3:
            s2s3_direct += 1
        elif source in S1 and target in S1:
            s1s1_direct += 1
        elif source in S2 and target in S2:
            s2s2_direct += 1
        elif source in S3 and target in S3:
            s3s3_direct += 1
        
        direct_edge_coverage[(source, target)] = pair_counts.get((source, target), 0)
    
    print(f"\nDirect edges in graph:")
    print(f"  S1→S2: {s1s2_direct} edges")
    print(f"  S2→S3: {s2s3_direct} edges")
    print(f"  S1→S1: {s1s1_direct} edges")
    print(f"  S2→S2: {s2s2_direct} edges")
    print(f"  S3→S3: {s3s3_direct} edges")
    
    # 找出原数据中缺失或需要强化的直接边
    edges_to_add = []
    already_covered = 0
    
    for (source, target) in direct_edges:
        # 跳过自边（已经被删除）
        if (source in S1 and target in S1) or \
           (source in S2 and target in S2) or \
           (source in S3 and target in S3):
            continue
        
        if (source, target) in pair_counts:
            # 即使已经存在，也添加最短路径（ALPINE特色）
            edges_to_add.append((source, target))
            already_covered += 1
        else:
            # 完全缺失的直接边
            edges_to_add.append((source, target))
    
    print(f"\nDirect edges to reinforce with shortest paths:")
    print(f"  Already in dataset: {already_covered}")
    print(f"  Missing from dataset: {len(edges_to_add) - already_covered}")
    print(f"  Total to add: {len(edges_to_add)}")
    
    # 创建新数据集：原始数据 + 直接边最短路径
    new_lines = []
    
    # 首先添加所有直接边的最短路径（ALPINE特色）
    for source, target in edges_to_add:
        # ALPINE格式：source target source target
        shortest_path = f"{source} {target} {source} {target}\n"
        new_lines.append(shortest_path)
    
    # 然后添加原始数据
    for parts in original_paths:
        new_lines.append(' '.join(parts) + '\n')
    
    # 写入新文件
    with open(output_file, 'w') as f:
        for line in new_lines:
            f.write(line)
    
    print(f"\n✅ New dataset created: {len(new_lines)} paths")
    print(f"   Added {len(edges_to_add)} direct edge shortest paths")
    
    # 统计新数据集
    print("\n" + "-"*50)
    print("New dataset statistics:")
    print("-"*50)
    
    s1s2_count = 0
    s2s3_count = 0
    s1s3_count = 0
    
    for line in new_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            source, target = int(parts[0]), int(parts[1])
            if source in S1 and target in S2:
                s1s2_count += 1
            elif source in S2 and target in S3:
                s2s3_count += 1
            elif source in S1 and target in S3:
                s1s3_count += 1
    
    total = len(new_lines)
    print(f"  S1→S2: {s1s2_count} ({s1s2_count/total*100:.1f}%)")
    print(f"  S2→S3: {s2s3_count} ({s2s3_count/total*100:.1f}%)")
    print(f"  S1→S3: {s1s3_count} ({s1s3_count/total*100:.1f}%)")
    
    return len(edges_to_add)

def prepare_binary_files(data_dir):
    """准备二进制文件用于训练"""
    print("\n" + "="*60)
    print("Preparing binary files...")
    print("="*60)
    
    # 读取数据
    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        train_data = f.read()
    
    with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
        val_data = f.read()
    
    # 设置词汇表
    total_nodes = 90
    vocab_size = total_nodes + 2
    stoi = {str(i): i + 2 for i in range(total_nodes)}
    itos = {i + 2: str(i) for i in range(total_nodes)}
    stoi['[PAD]'] = 0
    itos[0] = '[PAD]'
    stoi['\n'] = 1
    itos[1] = '\n'
    
    def encode(s):
        ss = s.split(" ")
        return [stoi[ch] for ch in ss if ch in stoi]
    
    def get_max_len(s):
        return max(len(line.split(' ')) for line in s.strip().split('\n') if line)
    
    # 计算block size
    max_len_train = get_max_len(train_data)
    max_len_val = get_max_len(val_data)
    block_size = max(max_len_train, max_len_val)
    print(f"Max sequence length: {block_size}")
    
    # 对齐到32的倍数
    block_size = (block_size // 32 + 1) * 32
    print(f"Using block size: {block_size}")
    
    def process_data(s, block_size):
        lines = s.strip().split('\n')
        ids = []
        for line in lines:
            if line:
                encoded_line = encode(line) + [stoi['\n']]
                padding = [stoi['[PAD]']] * (block_size - len(encoded_line))
                ids.extend(encoded_line + padding)
        return ids
    
    # 编码数据
    train_ids = process_data(train_data, block_size)
    val_ids = process_data(val_data, block_size)
    
    print(f"Train has {len(train_ids) // block_size} sequences")
    print(f"Val has {len(val_ids) // block_size} sequences")
    
    # 转换为numpy数组
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    # 保存bin文件
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))
    
    # 保存元信息
    meta = {
        'unreachable': False,
        'simple_format': True,
        'block_size': block_size,
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("✅ Binary files created: train.bin, val.bin, meta.pkl")

def main():
    # 源目录和目标目录
    source_dir = 'data/simple_graph/alpha_track_B_0.0'
    target_dir = 'data/simple_graph/trackB_0.0_alpine_style'
    
    print("\n" + "="*80)
    print("🔧 ADDING ALPINE DIRECT EDGES TO TRACK B α=0")
    print("="*80)
    print(f"\nSource: {source_dir}")
    print(f"Target: {target_dir}")
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 加载图和stage信息
    G, S1, S2, S3 = load_graph_and_stages(source_dir)
    
    # 识别所有直接边
    direct_edges = identify_direct_edges(G)
    print(f"\nTotal direct edges in graph: {len(direct_edges)}")
    
    # 处理训练数据
    input_train = os.path.join(source_dir, 'train.txt')
    output_train = os.path.join(target_dir, 'train.txt')
    
    num_added = process_trackB_dataset(input_train, output_train, direct_edges, S1, S2, S3)
    
    # 复制其他必要文件
    print("\n" + "="*60)
    print("Copying other necessary files...")
    print("="*60)
    
    files_to_copy = [
        'test.txt',
        'composition_graph.graphml',
        'stage_info.pkl'
    ]
    
    for filename in files_to_copy:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(target_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {filename}")
    
    # 准备二进制文件
    prepare_binary_files(target_dir)
    
    # 创建README
    readme_content = f"""# Track B α=0 with ALPINE Direct Edges

This dataset is Track B α=0 enhanced with ALPINE-style direct edge processing.

## Modifications:
- Added {num_added} shortest paths for direct edges
- Each direct edge now has a path: [source, target, source, target]
- This matches ALPINE's treatment of direct edges

## Purpose:
To isolate the effect of direct edge handling by making Track B α=0 
comparable to ALPINE strict method.

## Original source:
{source_dir}

## Processing date:
{os.popen('date').read().strip()}
"""
    
    with open(os.path.join(target_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print("\n" + "="*80)
    print("✅ PROCESSING COMPLETE!")
    print("="*80)
    print(f"\n📁 New dataset saved to: {target_dir}")
    print("\n🚀 You can now train with:")
    print(f"python train.py \\")
    print(f"    --dataset_path {target_dir} \\")
    print(f"    --out_dir out/trackB_0.0_alpine_style_d92_seed42 \\")
    print(f"    --n_layer 1 --n_head 1 --n_embd 92 \\")
    print(f"    --max_iters 50000 --eval_interval 5000 \\")
    print(f"    --batch_size 64 --learning_rate 1e-3 --seed 42")

if __name__ == "__main__":
    main()