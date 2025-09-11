#!/usr/bin/env python3
"""
weight_gap_comparison_correct.py
修正版 - 使用正确的节点分组
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

try:
    from model import GPTConfig, GPT
except ImportError:
    print("❌ Error: Cannot import 'model.py'")
    exit()

# ==================== 1. 配置与辅助函数 ====================

class ModelConfig:
    """模型配置类 - 使用正确的节点分组"""
    def __init__(self, mix_ratio, d_model=92):
        self.mix_ratio = mix_ratio
        self.d_model = d_model
        self.device = torch.device('cpu')
        
        # 模型参数
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = d_model
        self.vocab_size = 92
        
        # 数据目录
        self.data_dir = 'data/simple_graph/standardized_alpine_90_seed42'
        
        # 加载正确的节点分组和图结构
        self.load_stage_info()
        self.load_graph_structure()
    
    def load_stage_info(self):
        """加载正确的节点分组信息"""
        stage_info_path = os.path.join(self.data_dir, 'stage_info.pkl')
        
        if not os.path.exists(stage_info_path):
            raise FileNotFoundError(f"stage_info.pkl not found at {stage_info_path}")
        
        with open(stage_info_path, 'rb') as f:
            stage_info = pickle.load(f)
        
        self.S1, self.S2, self.S3 = stage_info['stages']
        
        # 转换为集合以便快速查找
        self.S1_set = set(self.S1)
        self.S2_set = set(self.S2)
        self.S3_set = set(self.S3)
        
        # 创建节点到token的映射（+2是因为token 0,1是特殊token）
        self.node_to_token = {node: node + 2 for node in range(90)}
        self.token_to_node = {token: node for node, token in self.node_to_token.items()}
        
        # 创建S1, S2, S3的token索引
        self.S1_tokens = [self.node_to_token[n] for n in self.S1]
        self.S2_tokens = [self.node_to_token[n] for n in self.S2]
        self.S3_tokens = [self.node_to_token[n] for n in self.S3]
        
        print(f"  ✓ Loaded stage info:")
        print(f"    S1: {len(self.S1)} nodes - {sorted(self.S1)[:10]}...")
        print(f"    S2: {len(self.S2)} nodes - {sorted(self.S2)[:10]}...")
        print(f"    S3: {len(self.S3)} nodes - {sorted(self.S3)[:10]}...")
    
    def load_graph_structure(self):
        """加载图结构文件"""
        graph_path = os.path.join(self.data_dir, 'composition_graph.graphml')
        
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found at {graph_path}")
        
        print(f"  Loading graph from: {graph_path}")
        G = nx.read_graphml(graph_path)
        self.G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
        
        print(f"  ✓ Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # 创建92x92的邻接矩阵（包含特殊token）
        self.A_true = np.zeros((self.vocab_size, self.vocab_size))
        
        # 填充邻接矩阵（考虑token偏移）
        for edge in self.G.edges():
            source_token = self.node_to_token[edge[0]]
            target_token = self.node_to_token[edge[1]]
            self.A_true[source_token, target_token] = 1
        
        # 统计各类型边的数量
        self.count_edges()
    
    def count_edges(self):
        """统计各类型边的数量"""
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
        
        print(f"  Edge statistics:")
        print(f"    S1->S2 edges: {s1_s2_edges}")
        print(f"    S2->S3 edges: {s2_s3_edges}")
        print(f"    S1->S3 edges: {s1_s3_edges}")
        
        self.edge_stats = {
            'S1->S2': s1_s2_edges,
            'S2->S3': s2_s3_edges,
            'S1->S3': s1_s3_edges
        }

def find_checkpoint(mix_ratio, d_model, iteration):
    """查找checkpoint文件"""
    if mix_ratio == 0:
        patterns = [
            f"out/composition_d{d_model}_0mix_*/ckpt_{iteration}.pt",
            f"out_d{d_model}/composition_mix0_*/ckpt_*{iteration}.pt",
        ]
    else:
        patterns = [
            f"out/composition_d{d_model}_{mix_ratio}mix_*/ckpt_{iteration}.pt",
            f"out_d{d_model}/composition_mix{mix_ratio}_*/ckpt_*{iteration}.pt",
        ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    return None

def extract_W_M_prime(checkpoint_path, config):
    """提取W'_M矩阵"""
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

def calculate_path_statistics_correct(W_M_prime, config, path_type):
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

def collect_evolution_data(config, iterations):
    """收集单个模型的演化数据"""
    model_name = f"{config.mix_ratio}% mix"
    print(f"\n📊 Processing {model_name} model...")
    
    model_data = {
        'S1->S2': {'edge': [], 'non_edge': [], 'gap': []},
        'S2->S3': {'edge': [], 'non_edge': [], 'gap': []},
        'S1->S3': {'edge': [], 'non_edge': [], 'gap': []}
    }
    
    found_checkpoints = 0
    for iteration in tqdm(iterations, desc=f"Loading checkpoints"):
        checkpoint_path = find_checkpoint(config.mix_ratio, config.d_model, iteration)
        
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
                stats = calculate_path_statistics_correct(W_M_prime, config, path_type)
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
    return model_data

# ==================== 3. 可视化和主函数 ====================

def plot_weight_gap_evolution(evolution_data, iterations, save_dir, model_name):
    """绘制weight gap演化图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Weight Gap Evolution - {model_name}', fontsize=16, fontweight='bold')
    
    path_types = ['S1->S2', 'S2->S3']
    
    for i, path_type in enumerate(path_types):
        # Edge weights
        ax = axes[i, 0]
        edge_weights = evolution_data[path_type]['edge']
        valid_iters = [it for it, w in zip(iterations, edge_weights) if not np.isnan(w)]
        valid_weights = [w for w in edge_weights if not np.isnan(w)]
        
        if valid_weights:
            ax.plot(valid_iters, valid_weights, marker='o', linewidth=2, markersize=5, color='blue')
        
        ax.set_ylabel(f'{path_type}', fontsize=12, fontweight='bold')
        ax.set_title('Average Edge Weight' if i == 0 else '', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Non-edge weights
        ax = axes[i, 1]
        non_edge_weights = evolution_data[path_type]['non_edge']
        valid_iters = [it for it, w in zip(iterations, non_edge_weights) if not np.isnan(w)]
        valid_weights = [w for w in non_edge_weights if not np.isnan(w)]
        
        if valid_weights:
            ax.plot(valid_iters, valid_weights, marker='s', linewidth=2, markersize=5, color='red')
        
        ax.set_title('Average Non-Edge Weight' if i == 0 else '', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Weight Gap
        ax = axes[i, 2]
        gaps = evolution_data[path_type]['gap']
        valid_iters = [it for it, g in zip(iterations, gaps) if not np.isnan(g)]
        valid_gaps = [g for g in gaps if not np.isnan(g)]
        
        if valid_gaps:
            ax.plot(valid_iters, valid_gaps, marker='^', linewidth=2.5, markersize=7, color='green')
            
            # 标注关键值
            final_gap = valid_gaps[-1]
            min_gap = min(valid_gaps)
            ax.annotate(f'Final: {final_gap:.3f}', 
                      xy=(valid_iters[-1], final_gap),
                      xytext=(5, 5), textcoords='offset points',
                      fontsize=8)
            
            if min_gap < 0:
                ax.fill_between(valid_iters, 0, [min(0, g) for g in valid_gaps], 
                               color='red', alpha=0.2, label='Negative gap')
        
        ax.set_title('Weight Gap (Edge - Non-Edge)' if i == 0 else '', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 添加S2->S3 gap的重要性标注
        if path_type == 'S2->S3' and valid_gaps:
            if min(valid_gaps) > 0:
                ax.text(0.95, 0.05, '✅ Always positive', 
                       transform=ax.transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            else:
                ax.text(0.95, 0.05, '⚠️ Goes negative', 
                       transform=ax.transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 设置x轴
    for i in range(2):
        for j in range(3):
            axes[i, j].set_xlabel('Training Iterations', fontsize=11)
            axes[i, j].set_xticks(iterations[::2])
            axes[i, j].set_xticklabels([f'{k//1000}k' for k in iterations[::2]], rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'weight_gap_{model_name.replace("% ", "_")}_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {save_path}")
    plt.show()

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 WEIGHT GAP ANALYSIS WITH CORRECT NODE GROUPING")
    print("="*80)
    
    d_model = 92
    iterations = list(range(5000, 51000, 5000))
    save_dir = 'weight_gap_analysis_corrected'
    os.makedirs(save_dir, exist_ok=True)
    
    # 分析两个模型
    for mix_ratio in [0, 20]:
        print(f"\n{'='*60}")
        print(f"Analyzing {mix_ratio}% mix model")
        print('='*60)
        
        config = ModelConfig(mix_ratio=mix_ratio, d_model=d_model)
        evolution_data = collect_evolution_data(config, iterations)
        
        # 打印统计
        print(f"\nStatistics for {mix_ratio}% mix model:")
        for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
            gaps = evolution_data[path_type]['gap']
            valid_gaps = [g for g in gaps if not np.isnan(g)]
            
            if valid_gaps:
                print(f"\n{path_type}:")
                print(f"  Initial gap: {valid_gaps[0]:.4f}")
                print(f"  Final gap: {valid_gaps[-1]:.4f}")
                print(f"  Min gap: {min(valid_gaps):.4f}")
                print(f"  Max gap: {max(valid_gaps):.4f}")
                
                if path_type == 'S2->S3' and min(valid_gaps) > 0:
                    print(f"  ✅ Always positive - S2 preserved!")
        
        # 生成图表
        plot_weight_gap_evolution(evolution_data, iterations, save_dir, f"{mix_ratio}% mix")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()