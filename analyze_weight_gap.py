#!/usr/bin/env python3
"""
analyze_weight_gap.py
通用的weight gap分析脚本 - 可以分析任何checkpoint目录
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

# ==================== 配置类 ====================

class ModelConfig:
    """模型配置类"""
    def __init__(self, checkpoint_dir, model_name="Model"):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.device = torch.device('cpu')
        
        # 模型参数 - 1层1头
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 92
        self.vocab_size = 92
        
        # 使用标准图路径
        self.graph_base_dir = 'data/simple_graph/standardized_alpine_90_seed42'
        
        # 加载节点分组和图结构
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
        
        print(f"  ✓ Loaded stage info: S1={len(self.S1)}, S2={len(self.S2)}, S3={len(self.S3)} nodes")
    
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
        
        print(f"  ✓ Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
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

# ==================== 核心分析函数 ====================

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
    stats['num_edges'] = np.sum(edge_mask)
    stats['num_non_edges'] = np.sum(non_edge_mask)
    
    return stats

def analyze_checkpoint_dir(checkpoint_dir, iterations=None, output_name=None):
    """分析checkpoint目录"""
    
    # 如果没有指定输出名称，从路径提取
    if output_name is None:
        output_name = os.path.basename(checkpoint_dir)
    
    print("\n" + "="*80)
    print(f"🔬 WEIGHT GAP ANALYSIS: {output_name}")
    print("="*80)
    
    # 默认iterations
    if iterations is None:
        iterations = list(range(5000, 51000, 5000))
    
    save_dir = f'weight_gap_analysis_{output_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📋 Configuration:")
    print(f"  • Checkpoint directory: {checkpoint_dir}")
    print(f"  • Iterations: {iterations}")
    print(f"  • Output: {save_dir}/")
    
    # 验证checkpoint目录
    if not os.path.exists(checkpoint_dir):
        print(f"\n❌ ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # 检查可用的checkpoints
    available = []
    for it in iterations:
        if os.path.exists(os.path.join(checkpoint_dir, f'ckpt_{it}.pt')):
            available.append(it)
    
    print(f"  • Found {len(available)}/{len(iterations)} checkpoints: {available}")
    
    if not available:
        print("❌ No checkpoints found!")
        return
    
    # 初始化配置
    print("\n" + "="*60)
    print("Loading graph structure...")
    print("="*60)
    
    config = ModelConfig(checkpoint_dir, output_name)
    
    # 收集数据
    print("\n" + "="*60)
    print("Analyzing checkpoints...")
    print("="*60)
    
    results = {
        'S1->S2': {'edge': [], 'non_edge': [], 'gap': []},
        'S2->S3': {'edge': [], 'non_edge': [], 'gap': []},
        'S1->S3': {'edge': [], 'non_edge': [], 'gap': []}
    }
    
    for iteration in tqdm(iterations, desc="Processing"):
        ckpt_path = os.path.join(checkpoint_dir, f'ckpt_{iteration}.pt')
        
        if not os.path.exists(ckpt_path):
            for path_type in results:
                results[path_type]['edge'].append(np.nan)
                results[path_type]['non_edge'].append(np.nan)
                results[path_type]['gap'].append(np.nan)
            continue
        
        try:
            W_M_prime = extract_W_M_prime(ckpt_path, config)
            
            for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                stats = calculate_weight_gap(W_M_prime, config, path_type)
                results[path_type]['edge'].append(stats['edge'])
                results[path_type]['non_edge'].append(stats['non_edge'])
                results[path_type]['gap'].append(stats['gap'])
                
        except Exception as e:
            print(f"\n  ⚠️ Error at iteration {iteration}: {e}")
            for path_type in results:
                results[path_type]['edge'].append(np.nan)
                results[path_type]['non_edge'].append(np.nan)
                results[path_type]['gap'].append(np.nan)
    
    # 生成详细图表
    print("\n" + "="*60)
    print("Generating detailed plots...")
    print("="*60)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f'Weight Gap Analysis: {output_name}', fontsize=16, fontweight='bold')
    
    path_types = ['S1->S2', 'S2->S3', 'S1->S3']
    colors = {'edge': '#2E86AB', 'non_edge': '#A23B72', 'gap': '#F18F01'}
    
    for i, path_type in enumerate(path_types):
        # 第1列：Edge权重
        ax = axes[i, 0]
        edge_weights = results[path_type]['edge']
        valid_iters = [it for it, w in zip(iterations, edge_weights) if not np.isnan(w)]
        valid_weights = [w for w in edge_weights if not np.isnan(w)]
        
        if valid_weights:
            ax.plot(valid_iters, valid_weights, marker='o', color=colors['edge'],
                   linewidth=2, markersize=6, alpha=0.8)
            ax.annotate(f'{valid_weights[-1]:.3f}', 
                       xy=(valid_iters[-1], valid_weights[-1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=colors['edge'])
        
        ax.set_title('Average Edge Weight' if i == 0 else '', fontsize=12)
        ax.set_ylabel(f'{path_type}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 第2列：Non-edge权重
        ax = axes[i, 1]
        non_edge_weights = results[path_type]['non_edge']
        valid_iters = [it for it, w in zip(iterations, non_edge_weights) if not np.isnan(w)]
        valid_weights = [w for w in non_edge_weights if not np.isnan(w)]
        
        if valid_weights:
            ax.plot(valid_iters, valid_weights, marker='s', color=colors['non_edge'],
                   linewidth=2, markersize=6, alpha=0.8, linestyle='--')
            ax.annotate(f'{valid_weights[-1]:.3f}', 
                       xy=(valid_iters[-1], valid_weights[-1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=colors['non_edge'])
        
        ax.set_title('Average Non-Edge Weight' if i == 0 else '', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 第3列：Weight Gap
        ax = axes[i, 2]
        gaps = results[path_type]['gap']
        valid_iters = [it for it, g in zip(iterations, gaps) if not np.isnan(g)]
        valid_gaps = [g for g in gaps if not np.isnan(g)]
        
        if valid_gaps:
            ax.plot(valid_iters, valid_gaps, marker='^', color=colors['gap'],
                   linewidth=2.5, markersize=7)
            
            # 标注最终值
            final_gap = valid_gaps[-1]
            ax.annotate(f'{final_gap:.3f}', 
                      xy=(valid_iters[-1], final_gap),
                      xytext=(5, 5), textcoords='offset points',
                      fontsize=10, color=colors['gap'], fontweight='bold')
            
            # 填充区域
            ax.fill_between(valid_iters, 0, valid_gaps, 
                           where=np.array(valid_gaps) > 0, 
                           alpha=0.2, color='green', interpolate=True)
            ax.fill_between(valid_iters, 0, valid_gaps, 
                           where=np.array(valid_gaps) < 0, 
                           alpha=0.2, color='red', interpolate=True)
            
            # 检查S2->S3的gap
            if path_type == 'S2->S3':
                if min(valid_gaps) > 0:
                    ax.text(0.95, 0.05, '✅ Always positive', 
                           transform=ax.transAxes, ha='right', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                else:
                    ax.text(0.95, 0.05, '⚠️ Goes negative', 
                           transform=ax.transAxes, ha='right', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        ax.set_title('Weight Gap (Edge - Non-Edge)' if i == 0 else '', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 设置x轴
    for i in range(3):
        for j in range(3):
            axes[i, j].set_xlabel('Training Iterations' if i == 2 else '', fontsize=11)
            axes[i, j].tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存详细图
    detail_path = os.path.join(save_dir, 'weight_gap_detailed.png')
    plt.savefig(detail_path, dpi=150, bbox_inches='tight')
    print(f"✅ Detailed plot saved to: {detail_path}")
    plt.show()
    
    # 生成简化图（只显示gap）
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle(f'Weight Gap Evolution: {output_name}', fontsize=14, fontweight='bold')
    
    colors2 = ['#2E86AB', '#F18F01', '#A23B72']
    
    for idx, (path_type, color) in enumerate(zip(path_types, colors2)):
        ax = axes2[idx]
        
        gaps = results[path_type]['gap']
        valid_iters = [it for it, g in zip(iterations, gaps) if not np.isnan(g)]
        valid_gaps = [g for g in gaps if not np.isnan(g)]
        
        if valid_gaps:
            ax.plot(valid_iters, valid_gaps, marker='o', color=color, 
                   linewidth=2, markersize=6)
            ax.fill_between(valid_iters, 0, valid_gaps, alpha=0.3, color=color)
            
            # 标注最终值
            ax.annotate(f'{valid_gaps[-1]:.3f}', 
                       xy=(valid_iters[-1], valid_gaps[-1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_title(f'{path_type}', fontsize=12)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Weight Gap')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # x轴标签
        ax.set_xticks(valid_iters[::2] if len(valid_iters) > 5 else valid_iters)
        ax.set_xticklabels([f'{k//1000}k' for k in (valid_iters[::2] if len(valid_iters) > 5 else valid_iters)])
    
    plt.tight_layout()
    
    # 保存简化图
    simple_path = os.path.join(save_dir, 'weight_gap_simple.png')
    plt.savefig(simple_path, dpi=150, bbox_inches='tight')
    print(f"✅ Simple plot saved to: {simple_path}")
    plt.show()
    
    # 打印最终统计
    print("\n" + "="*60)
    print("📊 FINAL STATISTICS")
    print("="*60)
    
    for path_type in path_types:
        gaps = [g for g in results[path_type]['gap'] if not np.isnan(g)]
        if gaps:
            print(f"\n{path_type}:")
            print(f"  Final gap: {gaps[-1]:.4f}")
            print(f"  Min gap:   {min(gaps):.4f}")
            print(f"  Max gap:   {max(gaps):.4f}")
            
            if path_type == 'S2->S3':
                if min(gaps) > 0:
                    print(f"  ✅ Always positive - Strong compositionality")
                else:
                    print(f"  ⚠️ Goes negative - Weak compositionality")
    
    # 保存数据
    with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump({
            'results': results,
            'iterations': iterations,
            'checkpoint_dir': checkpoint_dir,
            'edge_stats': config.edge_stats
        }, f)
    
    # 保存文本报告
    with open(os.path.join(save_dir, 'report.txt'), 'w') as f:
        f.write(f"Weight Gap Analysis Report\n")
        f.write(f"="*60 + "\n")
        f.write(f"Model: {output_name}\n")
        f.write(f"Checkpoint dir: {checkpoint_dir}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for path_type in path_types:
            gaps = [g for g in results[path_type]['gap'] if not np.isnan(g)]
            if gaps:
                f.write(f"\n{path_type}:\n")
                f.write(f"  Final gap: {gaps[-1]:.4f}\n")
                f.write(f"  Min gap:   {min(gaps):.4f}\n")
                f.write(f"  Max gap:   {max(gaps):.4f}\n")
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {save_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Analyze weight gap from checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--name', type=str, default=None, help='Output name (optional)')
    parser.add_argument('--iterations', type=str, default=None, help='Iterations to analyze (e.g., "5000,10000,15000")')
    
    args = parser.parse_args()
    
    # 解析iterations
    if args.iterations:
        iterations = [int(x) for x in args.iterations.split(',')]
    else:
        iterations = None
    
    analyze_checkpoint_dir(args.checkpoint_dir, iterations, args.name)

if __name__ == "__main__":
    # 如果没有命令行参数，直接分析指定的路径
    import sys
    if len(sys.argv) == 1:
        # 分析您提供的两个路径
        print("\n" + "="*80)
        print("ANALYZING TWO CHECKPOINT DIRECTORIES")
        print("="*80)
        
        # 第一个路径
        analyze_checkpoint_dir(
            'out/unknown_unknown_d92_seed42_20251005_070420',
            output_name='model_070420'
        )
        
        # 第二个路径
        analyze_checkpoint_dir(
            'out/unknown_unknown_d92_seed42_20251005_071502',
            output_name='model_071502'
        )
    else:
        main()