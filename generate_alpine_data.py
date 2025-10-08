#!/usr/bin/env python3
"""
generate_alpine_data.py
为修复后的图生成纯ALPINE数据（Tier 3策略）
"""

import os
import random
import pickle
import numpy as np
import networkx as nx
import argparse
from typing import List

class PureALPINEGenerator:
    def __init__(self, graph_dir, seed=42):
        self.graph_dir = graph_dir
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        self.load_graph_and_stages()
        self.compute_reachability()
    
    def load_graph_and_stages(self):
        """加载图和分组信息"""
        # 加载图
        graph_path = os.path.join(self.graph_dir, 'composition_graph.graphml')
        self.G = nx.read_graphml(graph_path)
        
        # 加载分组
        with open(os.path.join(self.graph_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        
        stages = stage_info['stages']
        self.S1 = [str(x) for x in stages[0]]
        self.S2 = [str(x) for x in stages[1]]
        self.S3 = [str(x) for x in stages[2]]
        
        print(f"  Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        print(f"  Stages: S1={len(self.S1)}, S2={len(self.S2)}, S3={len(self.S3)}")
    
    def compute_reachability(self):
        """计算可达性"""
        print("  Computing reachability...")
        TC = nx.transitive_closure(self.G)
        self.reachability = {}
        for node in self.G.nodes():
            self.reachability[node] = set(TC.predecessors(node))
    
    def random_walk(self, source, target, max_attempts=50):
        """纯ALPINE随机游走"""
        source_str = str(source)
        target_str = str(target)
        
        if not nx.has_path(self.G, source_str, target_str):
            return None
        
        for _ in range(max_attempts):
            path = [int(source_str)]
            current = source_str
            visited = {current}
            max_steps = len(self.G.nodes()) * 2
            
            for _ in range(max_steps):
                if current == target_str:
                    return path
                
                # 获取所有邻居
                neighbors = list(self.G.successors(current))
                
                # 过滤：只保留能到达目标的邻居
                valid_next = [
                    n for n in neighbors 
                    if (n == target_str or n in self.reachability.get(target_str, set()))
                    and n not in visited
                ]
                
                if not valid_next:
                    break  # 重试
                
                # 随机选择下一步
                next_node = random.choice(valid_next)
                path.append(int(next_node))
                visited.add(next_node)
                current = next_node
            
            # 如果没成功，重试
        
        return None
    
    def generate_pure_alpine_dataset(self, num_samples=10000, test_samples=500):
        """生成纯ALPINE数据集（所有路径通过随机游走生成）"""
        print("\n  Generating pure ALPINE dataset...")
        
        train_paths = []
        test_paths = []
        
        # 计算所有可达对
        all_pairs = []
        
        # S1->S2
        for s1 in self.S1:
            for s2 in self.S2:
                if nx.has_path(self.G, s1, s2):
                    all_pairs.append((int(s1), int(s2), 'S1->S2'))
        
        # S2->S3
        for s2 in self.S2:
            for s3 in self.S3:
                if nx.has_path(self.G, s2, s3):
                    all_pairs.append((int(s2), int(s3), 'S2->S3'))
        
        # S1->S3
        for s1 in self.S1:
            for s3 in self.S3:
                if nx.has_path(self.G, s1, s3):
                    all_pairs.append((int(s1), int(s3), 'S1->S3'))
        
        print(f"    Total reachable pairs: {len(all_pairs)}")
        
        if len(all_pairs) == 0:
            print("    WARNING: No reachable pairs!")
            return [], []
        
        # 生成训练集
        print(f"    Generating {num_samples} training samples...")
        for i in range(num_samples):
            # 随机选择一对
            source, target, pair_type = random.choice(all_pairs)
            
            # 生成路径
            path = self.random_walk(source, target)
            
            if path and len(path) >= 2:
                # 格式：source target path
                train_paths.append([source, target] + path)
            
            if (i + 1) % 2000 == 0:
                print(f"      Generated {i + 1}/{num_samples} samples")
        
        # 生成测试集
        print(f"    Generating {test_samples} test samples...")
        for i in range(test_samples):
            source, target, pair_type = random.choice(all_pairs)
            path = self.random_walk(source, target)
            
            if path and len(path) >= 2:
                test_paths.append([source, target] + path)
        
        print(f"    Generated: {len(train_paths)} train, {len(test_paths)} test")
        
        return train_paths, test_paths
    
    def save_dataset(self, train_paths, test_paths):
        """保存数据集"""
        # 保存训练集
        train_file = os.path.join(self.graph_dir, 'train_alpine.txt')
        with open(train_file, 'w') as f:
            for path in train_paths:
                f.write(' '.join(map(str, path)) + '\n')
        
        # 保存测试集
        test_file = os.path.join(self.graph_dir, 'test.txt')
        with open(test_file, 'w') as f:
            for path in test_paths:
                f.write(' '.join(map(str, path)) + '\n')
        
        print(f"  ✓ Saved to: {self.graph_dir}/")

def process_all_graphs(base_dir='data/graph_repair_experiment'):
    """处理所有7个图"""
    graph_names = ['G_base', 'G_A_low', 'G_A_high', 'G_B_low', 'G_B_high', 'G_C_low', 'G_C_high']
    
    for name in graph_names:
        graph_dir = os.path.join(base_dir, name)
        
        if not os.path.exists(graph_dir):
            print(f"⚠️ Skipping {name} - directory not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")
        
        generator = PureALPINEGenerator(graph_dir)
        train_paths, test_paths = generator.generate_pure_alpine_dataset()
        generator.save_dataset(train_paths, test_paths)

def main():
    parser = argparse.ArgumentParser(description='Generate pure ALPINE data for graphs')
    parser.add_argument('--graph_dir', type=str, default=None,
                       help='Single graph directory to process')
    parser.add_argument('--all', action='store_true',
                       help='Process all 7 graphs')
    parser.add_argument('--base_dir', type=str, default='data/graph_repair_experiment',
                       help='Base directory containing all graphs')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of training samples')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.all:
        print("\n" + "="*80)
        print("GENERATING PURE ALPINE DATA FOR ALL GRAPHS")
        print("="*80)
        process_all_graphs(args.base_dir)
        print("\n✅ All done!")
    elif args.graph_dir:
        print(f"\nProcessing single graph: {args.graph_dir}")
        generator = PureALPINEGenerator(args.graph_dir, args.seed)
        train_paths, test_paths = generator.generate_pure_alpine_dataset(args.num_samples)
        generator.save_dataset(train_paths, test_paths)
    else:
        print("Please specify --all or --graph_dir")

if __name__ == "__main__":
    main()