#!/usr/bin/env python3
"""
analyze_repair_weight_gap.py
分析图修复实验的weight gap - 适配不同的图结构
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import networkx as nx
from tqdm import tqdm
from datetime import datetime
import argparse

try:
    from model import GPTConfig, GPT
except ImportError:
    print("❌ Error: Cannot import 'model.py'")
    exit()

class RepairModelConfig:
    """修复实验的模型配置"""
    def __init__(self, checkpoint_dir, graph_name):
        self.checkpoint_dir = checkpoint_dir
        self.graph_name = graph_name
        self.device = torch.device('cpu')
        
        # 模型参数
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 92
        self.vocab_size = 92
        
        # 动态确定图路径
        self.graph_base_dir = f'data/graph_repair_experiment/{graph_name}'
        
        # 加载对应图的信息
        self.load_stage_info()
        self.load_graph_structure()
    
    def load_stage_info(self):
        """加载节点分组信息"""
        stage_info_path = os.path.join(self.graph_base_dir, 'stage_info.pkl')
        
        if not os.path.exists(stage_info_path):
            print(f"❌ ERROR: stage_info.pkl not found at {stage_info_path}")
            raise FileNotFoundError(f"Required file not found: {stage_info_path}")
        
        with open(stage_info_path, 'rb') as f:
            stage_info = pickle.load(f)
        
        self.S1, self.S2, self.S3 = stage_info['stages']
        
        # 转换为集合
        self.S1_set = set(self.S1)
        self.S2_set = set(self.S2)
        self.S3_set = set(self.S3)
        
        # 创建节点到token的映射
        self.node_to_token = {node: node + 2 for node in range(90)}
        self.token_to_node = {token: node for node, token in self.node_to_token.items()}
        
        # S1, S2, S3的token索引
        self.S1_tokens = [self.node_to_token[n] for n in self.S1]
        self.S2_tokens = [self.node_to_token[n] for n in self.S2]
        self.S3_tokens = [self.node_to_token[n] for n in self.S3]
        
        print(f"  ✓ Loaded stage info from {self.graph_name}: S1={len(self.S1)}, S2={len(self.S2)}, S3={len(self.S3)}")
    
    def load_graph_structure(self):
        """加载图结构"""
        graph_path = os.path.join(self.graph_base_dir, 'composition_graph.graphml')
        
        if not os.path.exists(graph_path):
            print(f"❌ ERROR: composition_graph.graphml not found at {graph_path}")
            raise FileNotFoundError(f"Required file not found: {graph_path}")
        
        G = nx.read_graphml(graph_path)
        
        # 确保节点是整数
        if isinstance(list(G.nodes())[0], str):
            self.G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
        else:
            self.G = G
        
        print(f"  ✓ Graph {self.graph_name}: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # 创建邻接矩阵
        self.A_true = np.zeros((self.vocab_size, self.vocab_size))
        
        for edge in self.G.edges():
            source_token = self.node_to_token[edge[0]]
            target_token = self.node_to_token[edge[1]]
            self.A_true[source_token, target_token] = 1
        
        # 统计各类型边
        s1_s2_edges = 0
        s2_s3_edges = 0
        s1_s3_edges = 0
        
        for edge in self.G.edges():
            source, target = edge[0], edge[1]
            if source in self.S1_set and target in self.S2_set:
                s1_s2_edges += 1
            elif source in self.S2_set and target in self.S3_set:
                s2_s3_edges += 1
            elif source in self.S1_set and target in self.S3_set:
                s1_s3_edges += 1
        
        print(f"  Edge statistics: S1→S2={s1_s2_edges}, S2→S3={s2_s3_edges}, S1→S3={s1_s3_edges}")
        
        self.edge_stats = {
            'S1->S2': s1_s2_edges,
            'S2->S3': s2_s3_edges,
            'S1->S3': s1_s3_edges
        }

def extract_W_M_prime(checkpoint_path, config):
    """提取W'_M矩阵"""
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    
    model_args = checkpoint.get('model_args', {})
    if not model_args:
        model_args = {
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'vocab_size': config.vocab_size,
            'block_size': 512,
            'dropout': 0.0,
            'bias': False
        }
    
    model_args['vocab_size'] = config.vocab_size
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(config.device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    
    W_M_prime = []
    with torch.no_grad():
        for i in range(config.vocab_size):
            token_emb = model.transformer.wte(torch.tensor([i], device=config.device))
            ffn_out = model.transformer.h[0].mlp(token_emb)
            combined = token_emb + ffn_out
            logits = model.lm_head(combined)
            W_M_prime.append(logits.squeeze().cpu().numpy()[:config.vocab_size])
    
    return np.array(W_M_prime)

def calculate_weight_gap(W_M_prime, config, path_type):
    """计算weight gap"""
    if path_type == 'S1->S2':
        source_tokens = config.S1_tokens
        target_tokens = config.S2_tokens
    elif path_type == 'S2->S3':
        source_tokens = config.S2_tokens
        target_tokens = config.S3_tokens
    elif path_type == 'S1->S3':
        source_tokens = config.S1_tokens
        target_tokens = config.S3_tokens
    
    W_sub = W_M_prime[np.ix_(source_tokens, target_tokens)]
    A_sub = config.A_true[np.ix_(source_tokens, target_tokens)]
    
    edge_mask = (A_sub == 1)
    non_edge_mask = (A_sub == 0)
    
    stats = {}
    
    if np.sum(edge_mask) > 0:
        stats['edge'] = np.mean(W_sub[edge_mask])
    else:
        stats['edge'] = 0
    
    if np.sum(non_edge_mask) > 0:
        stats['non_edge'] = np.mean(W_sub[non_edge_mask])
    else:
        stats['non_edge'] = 0
    
    stats['gap'] = stats['edge'] - stats['non_edge']
    return stats

def analyze_single_model(checkpoint_dir, graph_name, iterations=None):
    """分析单个模型"""
    
    print("\n" + "="*80)
    print(f"🔬 Analyzing: {graph_name}")
    print("="*80)
    
    if iterations is None:
        iterations = list(range(5000, 51000, 5000))
    
    # 加载对应的图配置
    config = RepairModelConfig(checkpoint_dir, graph_name)
    
    # 收集结果
    results = {
        'S1->S2': {'edge': [], 'non_edge': [], 'gap': []},
        'S2->S3': {'edge': [], 'non_edge': [], 'gap': []},
        'S1->S3': {'edge': [], 'non_edge': [], 'gap': []}
    }
    
    available = []
    for iteration in iterations:
        ckpt_path = os.path.join(checkpoint_dir, f'ckpt_{iteration}.pt')
        
        if not os.path.exists(ckpt_path):
            for path_type in results:
                results[path_type]['edge'].append(np.nan)
                results[path_type]['non_edge'].append(np.nan)
                results[path_type]['gap'].append(np.nan)
            continue
        
        available.append(iteration)
        
        try:
            W_M_prime = extract_W_M_prime(ckpt_path, config)
            
            for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                stats = calculate_weight_gap(W_M_prime, config, path_type)
                results[path_type]['edge'].append(stats['edge'])
                results[path_type]['non_edge'].append(stats['non_edge'])
                results[path_type]['gap'].append(stats['gap'])
                
        except Exception as e:
            print(f"  ⚠️ Error at iteration {iteration}: {e}")
            for path_type in results:
                results[path_type]['edge'].append(np.nan)
                results[path_type]['non_edge'].append(np.nan)
                results[path_type]['gap'].append(np.nan)
    
    print(f"  Analyzed {len(available)} checkpoints: {available}")
    
    # 打印最终结果
    for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
        gaps = [g for g in results[path_type]['gap'] if not np.isnan(g)]
        if gaps:
            print(f"  {path_type}: Final gap = {gaps[-1]:.4f}")
    
    return results, iterations, config

def analyze_all_repair_models():
    """分析所有7个修复模型"""
    
    graph_names = ['G_base', 'G_A_low', 'G_A_high', 'G_B_low', 'G_B_high', 'G_C_low', 'G_C_high']
    
    # 收集所有结果
    all_results = {}
    
    for graph_name in graph_names:
        # 查找对应的checkpoint目录
        pattern = f'out/{graph_name}_d92_seed42_*'
        dirs = glob.glob(pattern)
        
        if not dirs:
            print(f"⚠️ No checkpoint directory found for {graph_name}")
            continue
        
        # 使用最新的目录
        checkpoint_dir = sorted(dirs)[-1]
        print(f"\n📁 Found checkpoint for {graph_name}: {checkpoint_dir}")
        
        try:
            results, iterations, config = analyze_single_model(checkpoint_dir, graph_name)
            all_results[graph_name] = {
                'results': results,
                'iterations': iterations,
                'edge_stats': config.edge_stats
            }
        except Exception as e:
            print(f"❌ Failed to analyze {graph_name}: {e}")
    
    # 生成对比图
    if all_results:
        plot_comparison(all_results)
    
    return all_results

def plot_comparison(all_results):
    """生成对比图"""
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Graph Repair Experiment: Weight Gap Evolution', fontsize=16, fontweight='bold')
    
    # 定义颜色
    colors = {
        'G_base': 'black',
        'G_A_low': '#ff7f0e',
        'G_A_high': '#ff7f0e',
        'G_B_low': '#2ca02c',
        'G_B_high': '#2ca02c',
        'G_C_low': '#d62728',
        'G_C_high': '#d62728'
    }
    
    linestyles = {
        'G_base': '-',
        'G_A_low': '--',
        'G_A_high': '-',
        'G_B_low': '--',
        'G_B_high': '-',
        'G_C_low': '--',
        'G_C_high': '-'
    }
    
    path_types = ['S1->S2', 'S2->S3', 'S1->S3']
    
    for idx, path_type in enumerate(path_types):
        # 第一行：Weight Gap演化
        ax = axes[0, idx]
        
        for graph_name, data in all_results.items():
            gaps = data['results'][path_type]['gap']
            iterations = data['iterations']
            
            valid_idx = ~np.isnan(gaps)
            if np.any(valid_idx):
                ax.plot(np.array(iterations)[valid_idx], 
                       np.array(gaps)[valid_idx],
                       label=graph_name,
                       color=colors.get(graph_name, 'gray'),
                       linestyle=linestyles.get(graph_name, '-'),
                       linewidth=2,
                       marker='o' if 'high' in graph_name else 's',
                       markersize=4,
                       alpha=0.8)
        
        ax.set_title(f'{path_type} Weight Gap Evolution', fontsize=12)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Weight Gap')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        if idx == 2:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 第二行：最终值对比
        ax = axes[1, idx]
        
        final_gaps = []
        labels = []
        colors_list = []
        
        for graph_name in ['G_base', 'G_A_low', 'G_A_high', 'G_B_low', 'G_B_high', 'G_C_low', 'G_C_high']:
            if graph_name in all_results:
                gaps = all_results[graph_name]['results'][path_type]['gap']
                valid_gaps = [g for g in gaps if not np.isnan(g)]
                
                if valid_gaps:
                    final_gaps.append(valid_gaps[-1])
                    labels.append(graph_name.replace('_', '\n'))
                    colors_list.append(colors.get(graph_name, 'gray'))
        
        if final_gaps:
            bars = ax.bar(range(len(final_gaps)), final_gaps, color=colors_list, alpha=0.7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=0, fontsize=9)
            ax.set_title(f'{path_type} Final Gap Comparison', fontsize=12)
            ax.set_ylabel('Final Weight Gap')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 标注数值
            for bar, val in zip(bars, final_gaps):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom' if val > 0 else 'top',
                       fontsize=8)
    
    plt.tight_layout()
    
    # 保存
    save_dir = 'repair_weight_gap_analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'comparison_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to: {save_path}")
    plt.show()
    
    # 打印总结
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY (S2->S3 Weight Gap)")
    print("="*80)
    
    for graph_name in ['G_base', 'G_A_low', 'G_A_high', 'G_B_low', 'G_B_high', 'G_C_low', 'G_C_high']:
        if graph_name in all_results:
            gaps = all_results[graph_name]['results']['S2->S3']['gap']
            valid_gaps = [g for g in gaps if not np.isnan(g)]
            
            if valid_gaps:
                final = valid_gaps[-1]
                min_gap = min(valid_gaps)
                
                if min_gap > 0:
                    status = "✅ SUCCESS (always positive)"
                elif final > 0:
                    status = "⚠️ RECOVERED (but went negative)"
                else:
                    status = "❌ FAILED (negative at end)"
                
                print(f"{graph_name:<12}: Final={final:>7.3f}, Min={min_gap:>7.3f}  {status}")

def main():
    parser = argparse.ArgumentParser(description='Analyze repair experiment weight gaps')
    parser.add_argument('--graph', type=str, default=None, 
                       help='Specific graph to analyze (e.g., G_base, G_B_high)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Specific checkpoint directory')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all 7 graphs')
    
    args = parser.parse_args()
    
    if args.all or (not args.graph and not args.checkpoint_dir):
        # 分析所有模型
        analyze_all_repair_models()
    elif args.checkpoint_dir and args.graph:
        # 分析指定的模型
        results, iterations, config = analyze_single_model(args.checkpoint_dir, args.graph)
        
        # 生成单独的图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Weight Gap Analysis: {args.graph}', fontsize=14, fontweight='bold')
        
        path_types = ['S1->S2', 'S2->S3', 'S1->S3']
        colors = ['#2E86AB', '#F18F01', '#A23B72']
        
        for idx, (path_type, color) in enumerate(zip(path_types, colors)):
            ax = axes[idx]
            
            gaps = results[path_type]['gap']
            valid_idx = ~np.isnan(gaps)
            
            if np.any(valid_idx):
                ax.plot(np.array(iterations)[valid_idx],
                       np.array(gaps)[valid_idx],
                       marker='o', color=color, linewidth=2)
                ax.fill_between(np.array(iterations)[valid_idx],
                               0, np.array(gaps)[valid_idx],
                               alpha=0.3, color=color)
            
            ax.set_title(path_type)
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Weight Gap')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    else:
        print("Please specify --all or both --graph and --checkpoint_dir")

if __name__ == "__main__":
    main()