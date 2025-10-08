#!/usr/bin/env python3
"""
graph_repair_experiment.py
图修复实验 - 生成7个不同强度的修复图
"""

import os
import networkx as nx
import pickle
import random
import numpy as np
from typing import Set, List, Tuple
import argparse

class GraphRepairExperiment:
    def __init__(self, base_graph_dir='data/simple_graph/standardized_alpine_90_seed42', 
                 output_base_dir='data/graph_repair_experiment', seed=42):
        self.base_graph_dir = base_graph_dir
        self.output_base_dir = output_base_dir
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 加载原始GraphB
        self.load_base_graph()
        
        # 创建输出目录
        os.makedirs(output_base_dir, exist_ok=True)
    
    def load_base_graph(self):
        """加载原始GraphB和分组信息"""
        print("Loading base GraphB...")
        
        # 加载图
        graph_path = os.path.join(self.base_graph_dir, 'composition_graph.graphml')
        self.G_base = nx.read_graphml(graph_path)
        
        # 加载分组
        with open(os.path.join(self.base_graph_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        
        stages = stage_info['stages']
        self.S1 = set([str(x) for x in stages[0]])
        self.S2 = set([str(x) for x in stages[1]])
        self.S3 = set([str(x) for x in stages[2]])
        
        print(f"  Loaded graph: {self.G_base.number_of_nodes()} nodes, {self.G_base.number_of_edges()} edges")
        print(f"  S1={len(self.S1)}, S2={len(self.S2)}, S3={len(self.S3)}")
        
        # 统计原始边
        self.count_edges(self.G_base, "Base GraphB")
    
    def count_edges(self, G, name="Graph"):
        """统计图的边类型"""
        s1_s2, s2_s3, s1_s3 = 0, 0, 0
        
        for u, v in G.edges():
            if u in self.S1 and v in self.S2:
                s1_s2 += 1
            elif u in self.S2 and v in self.S3:
                s2_s3 += 1
            elif u in self.S1 and v in self.S3:
                s1_s3 += 1
        
        print(f"  [{name}] S1→S2: {s1_s2}, S2→S3: {s2_s3}, S1→S3: {s1_s3}, Total: {G.number_of_edges()}")
        
        return {'S1->S2': s1_s2, 'S2->S3': s2_s3, 'S1->S3': s1_s3}
    
    def add_uniform_edges(self, G, source_set: Set, target_set: Set, k_per_node: int) -> nx.DiGraph:
        """均匀补边：为source_set中每个节点添加k条到target_set的边"""
        G_new = G.copy()
        added_count = 0
        
        for source in source_set:
            # 找出当前未连接的目标节点
            current_neighbors = set(G_new.successors(source)) & target_set
            available_targets = list(target_set - current_neighbors)
            
            if len(available_targets) == 0:
                continue
            
            # 随机选择k个目标（如果不足k个，则全选）
            num_to_add = min(k_per_node, len(available_targets))
            new_targets = random.sample(available_targets, num_to_add)
            
            for target in new_targets:
                G_new.add_edge(source, target)
                added_count += 1
        
        print(f"    Added {added_count} edges ({k_per_node} per node × {len(source_set)} nodes)")
        return G_new
    
    def generate_repair_graphs(self):
        """生成所有7个实验图"""
        experiments = []
        
        # 1. G_base - 原始GraphB
        print("\n" + "="*60)
        print("1. G_base - Original GraphB (no repair)")
        print("="*60)
        G = self.G_base.copy()
        stats = self.count_edges(G, "G_base")
        self.save_graph(G, "G_base", stats)
        experiments.append(("G_base", G, stats))
        
        # 2. G_A_low - 只修入口，低强度
        print("\n" + "="*60)
        print("2. G_A_low - Entry repair only (k_in=2)")
        print("="*60)
        G = self.G_base.copy()
        G = self.add_uniform_edges(G, self.S1, self.S2, k_per_node=2)
        stats = self.count_edges(G, "G_A_low")
        self.save_graph(G, "G_A_low", stats)
        experiments.append(("G_A_low", G, stats))
        
        # 3. G_A_high - 只修入口，高强度
        print("\n" + "="*60)
        print("3. G_A_high - Entry repair only (k_in=5)")
        print("="*60)
        G = self.G_base.copy()
        G = self.add_uniform_edges(G, self.S1, self.S2, k_per_node=5)
        stats = self.count_edges(G, "G_A_high")
        self.save_graph(G, "G_A_high", stats)
        experiments.append(("G_A_high", G, stats))
        
        # 4. G_B_low - 只修出口，低强度
        print("\n" + "="*60)
        print("4. G_B_low - Exit repair only (k_out=1)")
        print("="*60)
        G = self.G_base.copy()
        G = self.add_uniform_edges(G, self.S2, self.S3, k_per_node=1)
        stats = self.count_edges(G, "G_B_low")
        self.save_graph(G, "G_B_low", stats)
        experiments.append(("G_B_low", G, stats))
        
        # 5. G_B_high - 只修出口，高强度
        print("\n" + "="*60)
        print("5. G_B_high - Exit repair only (k_out=3)")
        print("="*60)
        G = self.G_base.copy()
        G = self.add_uniform_edges(G, self.S2, self.S3, k_per_node=3)
        stats = self.count_edges(G, "G_B_high")
        self.save_graph(G, "G_B_high", stats)
        experiments.append(("G_B_high", G, stats))
        
        # 6. G_C_low - 协同修复，低强度
        print("\n" + "="*60)
        print("6. G_C_low - Collaborative repair (k_in=1, k_out=1)")
        print("="*60)
        G = self.G_base.copy()
        G = self.add_uniform_edges(G, self.S1, self.S2, k_per_node=1)
        G = self.add_uniform_edges(G, self.S2, self.S3, k_per_node=1)
        stats = self.count_edges(G, "G_C_low")
        self.save_graph(G, "G_C_low", stats)
        experiments.append(("G_C_low", G, stats))
        
        # 7. G_C_high - 协同修复，高强度
        print("\n" + "="*60)
        print("7. G_C_high - Collaborative repair (k_in=2, k_out=2)")
        print("="*60)
        G = self.G_base.copy()
        G = self.add_uniform_edges(G, self.S1, self.S2, k_per_node=2)
        G = self.add_uniform_edges(G, self.S2, self.S3, k_per_node=2)
        stats = self.count_edges(G, "G_C_high")
        self.save_graph(G, "G_C_high", stats)
        experiments.append(("G_C_high", G, stats))
        
        return experiments
    
    def save_graph(self, G, name, stats):
        """保存图和相关信息"""
        output_dir = os.path.join(self.output_base_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图
        nx.write_graphml(G, os.path.join(output_dir, 'composition_graph.graphml'))
        
        # 保存分组信息（与原始一致）
        stages = [
            sorted([int(x) for x in self.S1]),
            sorted([int(x) for x in self.S2]),
            sorted([int(x) for x in self.S3])
        ]
        
        stage_info = {
            'stages': stages,
            'nodes_per_stage': 30
        }
        
        with open(os.path.join(output_dir, 'stage_info.pkl'), 'wb') as f:
            pickle.dump(stage_info, f)
        
        # 保存修复统计
        repair_info = {
            'name': name,
            'edge_stats': stats,
            'total_edges': G.number_of_edges(),
            'seed': self.seed
        }
        
        with open(os.path.join(output_dir, 'repair_info.pkl'), 'wb') as f:
            pickle.dump(repair_info, f)
        
        print(f"  ✓ Saved to: {output_dir}/")
    
    def generate_summary_report(self, experiments):
        """生成汇总报告"""
        report_path = os.path.join(self.output_base_dir, 'repair_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GRAPH REPAIR EXPERIMENT SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("Experiment Design:\n")
            f.write("-"*40 + "\n")
            f.write("Base: Original GraphB (pathological)\n")
            f.write("A Group: Entry repair only (S1→S2)\n")
            f.write("B Group: Exit repair only (S2→S3)\n")
            f.write("C Group: Collaborative repair (both)\n\n")
            
            f.write("Edge Statistics:\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Graph':<12} {'S1→S2':>8} {'S2→S3':>8} {'S1→S3':>8} {'Total':>8}\n")
            f.write("-"*40 + "\n")
            
            for name, G, stats in experiments:
                f.write(f"{name:<12} {stats['S1->S2']:>8} {stats['S2->S3']:>8} "
                       f"{stats.get('S1->S3', 0):>8} {G.number_of_edges():>8}\n")
        
        print(f"\n✅ Summary report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Graph Repair Experiment')
    parser.add_argument('--base_graph_dir', type=str, 
                       default='data/simple_graph/standardized_alpine_90_seed42',
                       help='Directory containing the base GraphB')
    parser.add_argument('--output_dir', type=str,
                       default='data/graph_repair_experiment',
                       help='Output directory for repaired graphs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GRAPH REPAIR EXPERIMENT")
    print("="*80)
    
    experiment = GraphRepairExperiment(args.base_graph_dir, args.output_dir, args.seed)
    experiments = experiment.generate_repair_graphs()
    experiment.generate_summary_report(experiments)
    
    print("\n" + "="*80)
    print("✅ GRAPH GENERATION COMPLETE!")
    print(f"📁 Results saved to: {args.output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()