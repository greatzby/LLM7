#!/usr/bin/env python3
"""
analyze_alpha_weight_gaps.py
分析Track A所有alpha值的weight gap演化
基于1层1头模型的原始weight gap计算方法
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import networkx as nx
from tqdm import tqdm
import json
from datetime import datetime
import seaborn as sns

try:
    from model import GPTConfig, GPT
except ImportError:
    print("❌ Error: Cannot import 'model.py'")
    exit()

# ==================== 1. 配置与辅助函数 ====================

class AlphaModelConfig:
    """Alpha模型配置类 - 1层1头版本"""
    def __init__(self, alpha_value, track='A', d_model=92):
        self.alpha = alpha_value
        self.track = track
        self.d_model = d_model
        self.device = torch.device('cpu')
        
        # 模型参数 - 注意是1层1头！
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = d_model
        self.vocab_size = 92
        
        # 数据目录
        self.data_dir = f'data/simple_graph/alpha_track_{track}_{alpha_value:.1f}'
        
        # 加载节点分组和图结构
        self.load_stage_info()
        self.load_graph_structure()
    
    def load_stage_info(self):
        """加载节点分组信息"""
        stage_info_path = os.path.join(self.data_dir, 'stage_info.pkl')
        
        if not os.path.exists(stage_info_path):
            # 如果当前目录没有，使用原始目录的
            stage_info_path = 'data/simple_graph/standardized_alpine_90_seed42/stage_info.pkl'
        
        with open(stage_info_path, 'rb') as f:
            stage_info = pickle.load(f)
        
        self.S1, self.S2, self.S3 = stage_info['stages']
        
        # 转换为集合
        self.S1_set = set(self.S1)
        self.S2_set = set(self.S2)
        self.S3_set = set(self.S3)
        
        # 创建节点到token的映射（+2是因为token 0,1是特殊token）
        self.node_to_token = {node: node + 2 for node in range(90)}
        self.token_to_node = {token: node for node, token in self.node_to_token.items()}
        
        # S1, S2, S3的token索引
        self.S1_tokens = [self.node_to_token[n] for n in self.S1]
        self.S2_tokens = [self.node_to_token[n] for n in self.S2]
        self.S3_tokens = [self.node_to_token[n] for n in self.S3]
    
    def load_graph_structure(self):
        """加载图结构"""
        graph_path = os.path.join(self.data_dir, 'composition_graph.graphml')
        
        if not os.path.exists(graph_path):
            graph_path = 'data/simple_graph/standardized_alpine_90_seed42/composition_graph.graphml'
        
        G = nx.read_graphml(graph_path)
        
        # 确保节点是整数
        if isinstance(list(G.nodes())[0], str):
            self.G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
        else:
            self.G = G
        
        # 创建邻接矩阵
        self.A_true = np.zeros((self.vocab_size, self.vocab_size))
        
        for edge in self.G.edges():
            source_token = self.node_to_token[edge[0]]
            target_token = self.node_to_token[edge[1]]
            self.A_true[source_token, target_token] = 1

def find_alpha_checkpoint(alpha_value, iteration, base_dir='out'):
    """查找特定alpha值的checkpoint"""
    alpha_int = int(alpha_value * 100)
    pattern = f"{base_dir}/trackA_alpha{alpha_int}_d92_seed42_*/ckpt_{iteration}.pt"
    
    files = glob.glob(pattern)
    if files:
        return files[0]
    return None

def extract_W_M_prime(checkpoint_path, config):
    """提取W'_M矩阵 - 使用您原始的1层1头版本"""
    try:
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
        
    except Exception as e:
        print(f"    Error extracting W_M_prime: {e}")
        raise

def calculate_path_statistics(W_M_prime, config, path_type):
    """计算路径统计 - 使用正确的节点分组"""
    if path_type == 'S1->S2':
        source_tokens = config.S1_tokens
        target_tokens = config.S2_tokens
    elif path_type == 'S2->S3':
        source_tokens = config.S2_tokens
        target_tokens = config.S3_tokens
    elif path_type == 'S1->S3':
        source_tokens = config.S1_tokens
        target_tokens = config.S3_tokens
    else:
        raise ValueError(f"Invalid path_type: {path_type}")
    
    # 提取相关的子矩阵
    W_sub = W_M_prime[np.ix_(source_tokens, target_tokens)]
    A_sub = config.A_true[np.ix_(source_tokens, target_tokens)]
    
    # 创建掩码
    edge_mask = (A_sub == 1)
    non_edge_mask = (A_sub == 0)
    
    stats = {
        'num_edges': np.sum(edge_mask),
        'num_non_edges': np.sum(non_edge_mask)
    }
    
    if stats['num_edges'] > 0:
        stats['avg_edge_weight'] = np.mean(W_sub[edge_mask])
        stats['std_edge_weight'] = np.std(W_sub[edge_mask])
    else:
        stats['avg_edge_weight'] = np.nan
        stats['std_edge_weight'] = np.nan
    
    if stats['num_non_edges'] > 0:
        stats['avg_non_edge_weight'] = np.mean(W_sub[non_edge_mask])
        stats['std_non_edge_weight'] = np.std(W_sub[non_edge_mask])
    else:
        stats['avg_non_edge_weight'] = 0
        stats['std_non_edge_weight'] = 0
    
    if stats['num_edges'] > 0:
        stats['gap'] = stats['avg_edge_weight'] - stats['avg_non_edge_weight']
    else:
        stats['gap'] = np.nan
    
    return stats

# ==================== 2. 数据收集函数 ====================

def collect_alpha_evolution_data(alpha_values, iterations):
    """收集所有alpha值的演化数据"""
    all_data = {}
    
    for alpha in tqdm(alpha_values, desc="Processing alpha values"):
        print(f"\n📊 Processing α={alpha:.1f} model...")
        
        config = AlphaModelConfig(alpha_value=alpha, track='A')
        
        model_data = {
            'S1->S2': {'edge': [], 'non_edge': [], 'gap': []},
            'S2->S3': {'edge': [], 'non_edge': [], 'gap': []},
            'S1->S3': {'edge': [], 'non_edge': [], 'gap': []}
        }
        
        found_checkpoints = 0
        for iteration in iterations:
            checkpoint_path = find_alpha_checkpoint(alpha, iteration)
            
            if checkpoint_path is None:
                for path_type in model_data.keys():
                    model_data[path_type]['edge'].append(np.nan)
                    model_data[path_type]['non_edge'].append(np.nan)
                    model_data[path_type]['gap'].append(np.nan)
                continue
            
            try:
                W_M_prime = extract_W_M_prime(checkpoint_path, config)
                found_checkpoints += 1
                
                for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                    stats = calculate_path_statistics(W_M_prime, config, path_type)
                    model_data[path_type]['edge'].append(stats['avg_edge_weight'])
                    model_data[path_type]['non_edge'].append(stats['avg_non_edge_weight'])
                    model_data[path_type]['gap'].append(stats['gap'])
                    
            except Exception as e:
                print(f"  ⚠️ Error at iteration {iteration}: {e}")
                for path_type in model_data.keys():
                    model_data[path_type]['edge'].append(np.nan)
                    model_data[path_type]['non_edge'].append(np.nan)
                    model_data[path_type]['gap'].append(np.nan)
        
        print(f"  ✓ Found {found_checkpoints}/{len(iterations)} checkpoints")
        all_data[alpha] = model_data
    
    return all_data

# ==================== 3. 可视化函数 ====================

def plot_alpha_weight_gaps(all_data, alpha_values, iterations, save_dir):
    """生成alpha weight gap分析图"""
    
    # 1. Weight Gap演化图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Weight Gap Evolution Across Alpha Values (Track A, 1L-1H Model)', fontsize=16, fontweight='bold')
    
    path_types = ['S1->S2', 'S2->S3', 'S1->S3']
    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_values)))
    
    for i, path_type in enumerate(path_types):
        ax = axes[i]
        
        for j, alpha in enumerate(alpha_values):
            gaps = all_data[alpha][path_type]['gap']
            valid_iters = [it for it, g in zip(iterations, gaps) if not np.isnan(g)]
            valid_gaps = [g for g in gaps if not np.isnan(g)]
            
            if valid_gaps:
                ax.plot(valid_iters, valid_gaps, marker='o', label=f'α={alpha:.1f}',
                       color=colors[j], linewidth=1.5, markersize=3, alpha=0.7)
        
        ax.set_title(f'{path_type} Weight Gap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Iterations', fontsize=12)
        ax.set_ylabel('Weight Gap', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        if i == 2:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'alpha_weight_gap_evolution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Evolution plot saved to: {save_path}")
    plt.show()
    
    # 2. 最终Weight Gap vs Alpha图
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Final Weight Gap vs Alpha (1L-1H Model)', fontsize=16, fontweight='bold')
    
    final_gaps = {path_type: [] for path_type in path_types}
    
    for alpha in alpha_values:
        for path_type in path_types:
            gaps = all_data[alpha][path_type]['gap']
            valid_gaps = [g for g in gaps if not np.isnan(g)]
            if valid_gaps:
                final_gaps[path_type].append(valid_gaps[-1])
            else:
                final_gaps[path_type].append(np.nan)
    
    # 绘制曲线
    for path_type in path_types:
        valid_alphas = [a for a, g in zip(alpha_values, final_gaps[path_type]) if not np.isnan(g)]
        valid_gaps = [g for g in final_gaps[path_type] if not np.isnan(g)]
        
        if valid_gaps:
            ax.plot(valid_alphas, valid_gaps, marker='o', label=path_type, 
                   linewidth=2, markersize=8)
    
    ax.set_xlabel('Alpha (Interpolation Parameter)', fontsize=14)
    ax.set_ylabel('Final Weight Gap', fontsize=14)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    for alpha in [0.0, 0.5, 1.0]:
        ax.axvline(x=alpha, color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'final_weight_gap_vs_alpha.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Final gap plot saved to: {save_path}")
    plt.show()
    
    # 3. 热图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Weight Gap Heatmaps (1L-1H Model)', fontsize=16, fontweight='bold')
    
    for i, path_type in enumerate(path_types):
        ax = axes[i]
        
        gap_matrix = []
        for alpha in alpha_values:
            gaps = all_data[alpha][path_type]['gap']
            gap_matrix.append(gaps)
        
        gap_matrix = np.array(gap_matrix)
        
        im = ax.imshow(gap_matrix, aspect='auto', cmap='RdBu_r', 
                      vmin=-0.5, vmax=0.5, interpolation='nearest')
        
        ax.set_title(f'{path_type}', fontsize=14)
        ax.set_xlabel('Training Iterations', fontsize=12)
        ax.set_ylabel('Alpha', fontsize=12)
        
        ax.set_xticks(range(0, len(iterations), 2))
        ax.set_xticklabels([f'{it//1000}k' for it in iterations[::2]])
        ax.set_yticks(range(len(alpha_values)))
        ax.set_yticklabels([f'{a:.1f}' for a in alpha_values])
        
        plt.colorbar(im, ax=ax, label='Weight Gap')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'weight_gap_heatmaps.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Heatmap saved to: {save_path}")
    plt.show()

def print_summary_statistics(all_data, alpha_values):
    """打印汇总统计"""
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS (1L-1H Model)")
    print("="*80)
    
    path_types = ['S1->S2', 'S2->S3', 'S1->S3']
    
    final_gaps = {path_type: {} for path_type in path_types}
    
    for alpha in alpha_values:
        for path_type in path_types:
            gaps = all_data[alpha][path_type]['gap']
            valid_gaps = [g for g in gaps if not np.isnan(g)]
            if valid_gaps:
                final_gaps[path_type][alpha] = valid_gaps[-1]
    
    print("\nFinal Weight Gaps:")
    print("-" * 60)
    print(f"{'Alpha':<10} {'S1->S2':<15} {'S2->S3':<15} {'S1->S3':<15}")
    print("-" * 60)
    
    for alpha in alpha_values:
        row = f"{alpha:<10.1f}"
        for path_type in path_types:
            if alpha in final_gaps[path_type]:
                value = final_gaps[path_type][alpha]
                row += f"{value:<15.4f}"
                if path_type == 'S2->S3' and value < 0:
                    row = row[:-5] + "⚠️" + row[-4:]
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("-" * 60)
    
    print("\n🔍 Key Observations:")
    
    s23_gaps = final_gaps['S2->S3']
    negative_alphas = [a for a, g in s23_gaps.items() if g < 0]
    if negative_alphas:
        print(f"  • S2->S3 gap becomes negative at α ≥ {min(negative_alphas):.1f}")
    else:
        print(f"  • S2->S3 gap remains positive for all α values ✅")
    
    s13_gaps = final_gaps['S1->S3']
    if s13_gaps:
        max_alpha = max(s13_gaps.items(), key=lambda x: x[1])[0]
        print(f"  • S1->S3 gap is maximum at α = {max_alpha:.1f}")

# ==================== 4. 主函数 ====================

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 ALPHA MIXING WEIGHT GAP ANALYSIS (1-Layer 1-Head Model)")
    print("="*80)
    
    # 配置
    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    iterations = list(range(5000, 51000, 5000))
    save_dir = 'alpha_weight_gap_analysis_1L1H'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📋 Configuration:")
    print(f"  • Model: 1 Layer, 1 Head, 92D")
    print(f"  • Alpha values: {alpha_values}")
    print(f"  • Iterations: {iterations}")
    print(f"  • Output directory: {save_dir}")
    
    # 收集数据
    print("\n" + "="*60)
    print("Collecting data from checkpoints...")
    print("="*60)
    
    all_data = collect_alpha_evolution_data(alpha_values, iterations)
    
    # 保存原始数据
    with open(os.path.join(save_dir, 'alpha_weight_gap_data.pkl'), 'wb') as f:
        pickle.dump(all_data, f)
    print(f"✅ Raw data saved to: {save_dir}/alpha_weight_gap_data.pkl")
    
    # 生成可视化
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    plot_alpha_weight_gaps(all_data, alpha_values, iterations, save_dir)
    
    # 打印统计
    print_summary_statistics(all_data, alpha_values)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print(f"📁 All results saved to: {save_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()