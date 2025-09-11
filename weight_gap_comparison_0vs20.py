#!/usr/bin/env python3
"""
weight_gap_comparison_0vs20.py
比较0%和20%模型的weight gap演化，基于ALPINE理论框架
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

# 确保模型定义文件存在
try:
    from model import GPTConfig, GPT
except ImportError:
    print("❌ Error: Cannot import 'model.py'. Please ensure it's in the same directory.")
    exit()

# ==================== 1. 配置与辅助函数 ====================

class ModelConfig:
    """模型配置类"""
    def __init__(self, mix_ratio, d_model=92):
        self.mix_ratio = mix_ratio
        self.d_model = d_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型参数
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = d_model
        self.vocab_size = 94  # 标准设置
        
        # 设置checkpoint目录模式
        if mix_ratio == 0:
            self.checkpoint_pattern = f"out/composition_d{d_model}_0mix_*/ckpt_*.pt"
        else:
            self.checkpoint_pattern = f"out/composition_d{d_model}_{mix_ratio}mix_*/ckpt_*.pt"
        
        # 加载图结构
        self.load_graph_structure()
    
    def load_graph_structure(self):
        """加载图结构并生成邻接矩阵"""
        # 尝试多个可能的路径
        graph_paths = [
            'data/simple_graph/composition_90/composition_graph.graphml',
            'data/simple_graph/composition_graph.graphml',
        ]
        
        graph_loaded = False
        for graph_path in graph_paths:
            if os.path.exists(graph_path):
                G = nx.read_graphml(graph_path)
                self.G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
                graph_loaded = True
                print(f"✓ Loaded graph from: {graph_path}")
                break
        
        if not graph_loaded:
            print("⚠️ Warning: Using synthetic graph structure")
            # 创建合成图：S1->S2->S3
            self.G = nx.DiGraph()
            # S1 nodes (0-29), S2 nodes (30-59), S3 nodes (60-89)
            for i in range(30):
                for j in range(30, 60):
                    if np.random.random() < 0.3:  # 30%连接概率
                        self.G.add_edge(i, j)
                for j in range(60, 90):
                    if np.random.random() < 0.3:
                        self.G.add_edge(j-30, j)  # S2->S3
        
        # 生成邻接矩阵
        nodelist = sorted(self.G.nodes())
        A_true_90 = nx.to_numpy_array(self.G, nodelist=nodelist)
        self.A_true = np.zeros((self.vocab_size, self.vocab_size))
        self.A_true[2:92, 2:92] = A_true_90
        
        # 计算可达矩阵（2步）
        self.R_true = self.A_true + np.linalg.matrix_power(self.A_true, 2)
        self.R_true[self.R_true > 0] = 1
        
        print(f"✓ Adjacency matrix created (shape: {self.A_true.shape})")
        print(f"  S1->S2 edges: {np.sum(self.A_true[2:32, 32:62])}")
        print(f"  S2->S3 edges: {np.sum(self.A_true[32:62, 62:92])}")

def find_checkpoint(mix_ratio, d_model, iteration):
    """查找特定迭代的checkpoint"""
    if mix_ratio == 0:
        pattern = f"out/composition_d{d_model}_0mix_*/ckpt_{iteration}.pt"
    else:
        pattern = f"out/composition_d{d_model}_{mix_ratio}mix_*/ckpt_{iteration}.pt"
    
    files = glob.glob(pattern)
    if files:
        return files[0]
    return None

def extract_W_M_prime(checkpoint_path, config):
    """提取W'_M矩阵（包含残差连接）"""
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
    
    # 获取模型配置
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
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(config.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 提取W'_M矩阵
    W_M_prime = []
    with torch.no_grad():
        for i in range(config.vocab_size):
            # Token embedding
            token_emb = model.transformer.wte(torch.tensor([i], device=config.device))
            # FFN output
            ffn_out = model.transformer.h[0].mlp(token_emb)
            # 残差连接
            combined = token_emb + ffn_out
            # 通过lm_head
            logits = model.lm_head(combined)
            W_M_prime.append(logits.squeeze().cpu().numpy())
    
    return np.array(W_M_prime)

def calculate_path_statistics(W_M_prime, A_true, path_type):
    """计算特定路径类型的统计信息"""
    if path_type == 'S1->S2':
        rows, cols = np.s_[2:32], np.s_[32:62]
    elif path_type == 'S2->S3':
        rows, cols = np.s_[32:62], np.s_[62:92]
    elif path_type == 'S1->S3':
        rows, cols = np.s_[2:32], np.s_[62:92]
    else:
        raise ValueError(f"Invalid path_type: {path_type}")
    
    # 提取子矩阵
    W_sub = W_M_prime[rows, cols]
    A_sub = A_true[rows, cols]
    
    # 创建掩码
    edge_mask = (A_sub == 1)
    non_edge_mask = (A_sub == 0)
    
    # 计算统计量
    stats = {
        'num_edges': np.sum(edge_mask),
        'num_non_edges': np.sum(non_edge_mask)
    }
    
    if stats['num_edges'] > 0:
        stats['avg_edge_weight'] = np.mean(W_sub[edge_mask])
        stats['std_edge_weight'] = np.std(W_sub[edge_mask])
        stats['max_edge_weight'] = np.max(W_sub[edge_mask])
        stats['min_edge_weight'] = np.min(W_sub[edge_mask])
    else:
        stats['avg_edge_weight'] = np.nan
        stats['std_edge_weight'] = np.nan
        stats['max_edge_weight'] = np.nan
        stats['min_edge_weight'] = np.nan
    
    if stats['num_non_edges'] > 0:
        stats['avg_non_edge_weight'] = np.mean(W_sub[non_edge_mask])
        stats['std_non_edge_weight'] = np.std(W_sub[non_edge_mask])
    else:
        stats['avg_non_edge_weight'] = 0
        stats['std_non_edge_weight'] = 0
    
    # 计算gap
    if stats['num_edges'] > 0:
        stats['gap'] = stats['avg_edge_weight'] - stats['avg_non_edge_weight']
        stats['gap_ratio'] = stats['avg_edge_weight'] / (stats['avg_non_edge_weight'] + 1e-8)
    else:
        stats['gap'] = np.nan
        stats['gap_ratio'] = np.nan
    
    return stats

# ==================== 2. 数据收集函数 ====================

def collect_evolution_data(configs, iterations):
    """收集两个模型的演化数据"""
    evolution_data = {}
    
    for model_name, config in configs.items():
        print(f"\n📊 Processing {model_name} model...")
        
        model_data = {
            'S1->S2': {'edge': [], 'non_edge': [], 'gap': [], 'gap_ratio': []},
            'S2->S3': {'edge': [], 'non_edge': [], 'gap': [], 'gap_ratio': []},
            'S1->S3': {'edge': [], 'non_edge': [], 'gap': [], 'gap_ratio': []}
        }
        
        found_checkpoints = 0
        for iteration in tqdm(iterations, desc=f"Loading {model_name} checkpoints"):
            checkpoint_path = find_checkpoint(config.mix_ratio, config.d_model, iteration)
            
            if checkpoint_path is None:
                # 填充NaN
                for path_type in model_data.keys():
                    model_data[path_type]['edge'].append(np.nan)
                    model_data[path_type]['non_edge'].append(np.nan)
                    model_data[path_type]['gap'].append(np.nan)
                    model_data[path_type]['gap_ratio'].append(np.nan)
                continue
            
            found_checkpoints += 1
            
            try:
                # 提取W'_M矩阵
                W_M_prime = extract_W_M_prime(checkpoint_path, config)
                
                # 计算各路径统计
                for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                    stats = calculate_path_statistics(W_M_prime, config.A_true, path_type)
                    model_data[path_type]['edge'].append(stats['avg_edge_weight'])
                    model_data[path_type]['non_edge'].append(stats['avg_non_edge_weight'])
                    model_data[path_type]['gap'].append(stats['gap'])
                    model_data[path_type]['gap_ratio'].append(stats['gap_ratio'])
                    
            except Exception as e:
                print(f"  ⚠️ Error at iteration {iteration}: {e}")
                for path_type in model_data.keys():
                    model_data[path_type]['edge'].append(np.nan)
                    model_data[path_type]['non_edge'].append(np.nan)
                    model_data[path_type]['gap'].append(np.nan)
                    model_data[path_type]['gap_ratio'].append(np.nan)
        
        print(f"  ✓ Found {found_checkpoints}/{len(iterations)} checkpoints")
        evolution_data[model_name] = model_data
    
    return evolution_data

# ==================== 3. 可视化函数 ====================

def plot_comprehensive_comparison(evolution_data, iterations, save_dir):
    """生成综合对比图（3x3布局）"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Weight Gap Evolution: 0% vs 20% Mix Models', fontsize=16, fontweight='bold')
    
    colors = {'0% mix': 'blue', '20% mix': 'red'}
    path_types = ['S1->S2', 'S2->S3', 'S1->S3']
    
    for i, path_type in enumerate(path_types):
        # 第1列：Edge权重对比
        ax = axes[i, 0]
        for model_name, model_data in evolution_data.items():
            edge_weights = model_data[path_type]['edge']
            valid_iters = [it for it, w in zip(iterations, edge_weights) if not np.isnan(w)]
            valid_weights = [w for w in edge_weights if not np.isnan(w)]
            
            if valid_weights and path_type != 'S1->S3':
                ax.plot(valid_iters, valid_weights, marker='o', label=model_name,
                       color=colors[model_name], linewidth=2, markersize=5, alpha=0.8)
        
        ax.set_ylabel(f'{path_type}', fontsize=12, fontweight='bold')
        if i == 0:
            ax.set_title('Average Edge Weight', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc='best')
        
        # 第2列：Non-edge权重对比
        ax = axes[i, 1]
        for model_name, model_data in evolution_data.items():
            non_edge_weights = model_data[path_type]['non_edge']
            valid_iters = [it for it, w in zip(iterations, non_edge_weights) if not np.isnan(w)]
            valid_weights = [w for w in non_edge_weights if not np.isnan(w)]
            
            if valid_weights:
                ax.plot(valid_iters, valid_weights, marker='s', label=model_name,
                       color=colors[model_name], linewidth=2, markersize=5, 
                       alpha=0.8, linestyle='--')
        
        if i == 0:
            ax.set_title('Average Non-Edge Weight', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc='best')
        
        # 第3列：Weight Gap对比
        ax = axes[i, 2]
        for model_name, model_data in evolution_data.items():
            gaps = model_data[path_type]['gap']
            valid_iters = [it for it, g in zip(iterations, gaps) if not np.isnan(g)]
            valid_gaps = [g for g in gaps if not np.isnan(g)]
            
            if valid_gaps and path_type != 'S1->S3':
                ax.plot(valid_iters, valid_gaps, marker='^', label=model_name,
                       color=colors[model_name], linewidth=2.5, markersize=7)
                
                # 标注关键点
                if valid_gaps:
                    final_gap = valid_gaps[-1]
                    ax.annotate(f'{final_gap:.3f}', 
                              xy=(valid_iters[-1], final_gap),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, color=colors[model_name])
        
        if i == 0:
            ax.set_title('Weight Gap (Edge - Non-Edge)', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc='best')
        
        # S1->S3特殊处理
        if path_type == 'S1->S3':
            axes[i, 0].text(0.5, 0.5, 'No direct edges\nin graph', 
                          transform=axes[i, 0].transAxes,
                          ha='center', va='center', fontsize=12, color='gray')
            axes[i, 2].text(0.5, 0.5, 'N/A', 
                          transform=axes[i, 2].transAxes,
                          ha='center', va='center', fontsize=12, color='gray')
    
    # 设置x轴标签
    for j in range(3):
        axes[2, j].set_xlabel('Training Iterations', fontsize=11)
        if iterations:
            step = max(1, len(iterations) // 5)
            axes[2, j].set_xticks(iterations[::step])
            axes[2, j].set_xticklabels([f'{k//1000}k' for k in iterations[::step]], rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'weight_gap_comparison_0vs20_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comprehensive plot saved to: {save_path}")
    plt.show()

def plot_focused_comparison(evolution_data, iterations, save_dir):
    """生成聚焦对比图（2x2布局，重点关注S2->S3）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Critical Path Analysis: 0% vs 20% Mix Models', fontsize=16, fontweight='bold')
    
    colors = {'0% mix': 'blue', '20% mix': 'red'}
    
    # 左上：S1->S2 gap
    ax = axes[0, 0]
    for model_name, model_data in evolution_data.items():
        gaps = model_data['S1->S2']['gap']
        valid_iters = [it for it, g in zip(iterations, gaps) if not np.isnan(g)]
        valid_gaps = [g for g in gaps if not np.isnan(g)]
        
        if valid_gaps:
            ax.plot(valid_iters, valid_gaps, marker='o', label=model_name,
                   color=colors[model_name], linewidth=2.5, markersize=6)
    
    ax.set_title('S1→S2 Weight Gap', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gap (Edge - Non-edge)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    
    # 右上：S2->S3 gap（最重要！）
    ax = axes[0, 1]
    for model_name, model_data in evolution_data.items():
        gaps = model_data['S2->S3']['gap']
        valid_iters = [it for it, g in zip(iterations, gaps) if not np.isnan(g)]
        valid_gaps = [g for g in gaps if not np.isnan(g)]
        
        if valid_gaps:
            ax.plot(valid_iters, valid_gaps, marker='s', label=model_name,
                   color=colors[model_name], linewidth=2.5, markersize=6)
            
            # 标注最小值
            min_gap = min(valid_gaps)
            min_idx = valid_gaps.index(min_gap)
            ax.annotate(f'Min: {min_gap:.3f}', 
                      xy=(valid_iters[min_idx], min_gap),
                      xytext=(-30, -20), textcoords='offset points',
                      arrowprops=dict(arrowstyle='->', color=colors[model_name], alpha=0.5),
                      fontsize=9, color=colors[model_name])
    
    ax.set_title('S2→S3 Weight Gap (Critical!)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gap (Edge - Non-edge)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    
    # 添加关键观察
    if min_gap > 0:
        ax.text(0.95, 0.05, '✅ Always positive', 
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    else:
        ax.text(0.95, 0.05, '⚠️ Goes negative', 
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 左下：Gap比较柱状图
    ax = axes[1, 0]
    final_idx = -1
    x_pos = np.arange(2)
    
    s12_gaps_final = [evolution_data[m]['S1->S2']['gap'][final_idx] 
                      for m in ['0% mix', '20% mix']]
    s23_gaps_final = [evolution_data[m]['S2->S3']['gap'][final_idx] 
                      for m in ['0% mix', '20% mix']]
    
    # 过滤NaN值
    s12_gaps_final = [g if not np.isnan(g) else 0 for g in s12_gaps_final]
    s23_gaps_final = [g if not np.isnan(g) else 0 for g in s23_gaps_final]
    
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, s12_gaps_final, width, label='S1→S2', color='skyblue')
    bars2 = ax.bar(x_pos + width/2, s23_gaps_final, width, label='S2→S3', color='lightcoral')
    
    ax.set_title(f'Final Weight Gaps (iter {iterations[-1]//1000}k)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gap Value', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['0% mix', '20% mix'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 右下：S1->S3 non-edge weights
    ax = axes[1, 1]
    for model_name, model_data in evolution_data.items():
        non_edge_weights = model_data['S1->S3']['non_edge']
        valid_iters = [it for it, w in zip(iterations, non_edge_weights) if not np.isnan(w)]
        valid_weights = [w for w in non_edge_weights if not np.isnan(w)]
        
        if valid_weights:
            ax.plot(valid_iters, valid_weights, marker='^', label=model_name,
                   color=colors[model_name], linewidth=2, markersize=5, alpha=0.8)
    
    ax.set_title('S1→S3 Average Weight (No direct edges)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Weight', fontsize=11)
    ax.set_xlabel('Training Iterations', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    
    # 设置x轴
    for ax in [axes[0, 0], axes[0, 1], axes[1, 1]]:
        if iterations:
            step = max(1, len(iterations) // 5)
            ax.set_xticks(iterations[::step])
            ax.set_xticklabels([f'{k//1000}k' for k in iterations[::step]], rotation=45)
        ax.set_xlabel('Training Iterations', fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'weight_gap_focused_0vs20_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Focused plot saved to: {save_path}")
    plt.show()

# ==================== 4. 统计分析函数 ====================

def print_detailed_statistics(evolution_data, iterations):
    """打印详细统计信息"""
    print("\n" + "="*80)
    print("📊 DETAILED STATISTICS COMPARISON")
    print("="*80)
    
    for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
        print(f"\n{'='*60}")
        print(f"{path_type} Path Analysis:")
        print(f"{'='*60}")
        
        for model_name in ['0% mix', '20% mix']:
            model_data = evolution_data[model_name][path_type]
            
            print(f"\n{model_name} Model:")
            print("-" * 40)
            
            if path_type != 'S1->S3':
                # Edge weights
                edge_weights = [w for w in model_data['edge'] if not np.isnan(w)]
                if edge_weights:
                    print(f"  Edge weights:")
                    print(f"    Initial: {edge_weights[0]:.4f}")
                    print(f"    Final:   {edge_weights[-1]:.4f}")
                    print(f"    Range:   [{min(edge_weights):.4f}, {max(edge_weights):.4f}]")
                
                # Gap analysis
                gaps = [g for g in model_data['gap'] if not np.isnan(g)]
                if gaps:
                    print(f"  Weight gap:")
                    print(f"    Initial: {gaps[0]:.4f}")
                    print(f"    Final:   {gaps[-1]:.4f}")
                    print(f"    Range:   [{min(gaps):.4f}, {max(gaps):.4f}]")
                    
                    if path_type == 'S2->S3':
                        if min(gaps) > 0:
                            print(f"    ✅ Always positive - S2 preserved!")
                        else:
                            negative_count = sum(1 for g in gaps if g < 0)
                            print(f"    ⚠️ Negative {negative_count}/{len(gaps)} times")
            else:
                # S1->S3 (only non-edges)
                non_edge_weights = [w for w in model_data['non_edge'] if not np.isnan(w)]
                if non_edge_weights:
                    print(f"  Average weight (no direct edges):")
                    print(f"    Initial: {non_edge_weights[0]:.4f}")
                    print(f"    Final:   {non_edge_weights[-1]:.4f}")
                    print(f"    Range:   [{min(non_edge_weights):.4f}, {max(non_edge_weights):.4f}]")

def generate_summary_report(evolution_data, iterations, save_dir):
    """生成总结报告"""
    report = {
        'experiment': '0% vs 20% Weight Gap Comparison',
        'timestamp': datetime.now().isoformat(),
        'iterations': iterations,
        'models': {}
    }
    
    for model_name in ['0% mix', '20% mix']:
        model_report = {}
        for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
            model_data = evolution_data[model_name][path_type]
            
            # 清理NaN值
            edge_weights = [w for w in model_data['edge'] if not np.isnan(w)]
            gaps = [g for g in model_data['gap'] if not np.isnan(g)]
            
            path_report = {}
            if edge_weights:
                path_report['edge_initial'] = edge_weights[0]
                path_report['edge_final'] = edge_weights[-1]
                path_report['edge_min'] = min(edge_weights)
                path_report['edge_max'] = max(edge_weights)
            
            if gaps:
                path_report['gap_initial'] = gaps[0]
                path_report['gap_final'] = gaps[-1]
                path_report['gap_min'] = min(gaps)
                path_report['gap_max'] = max(gaps)
                path_report['gap_always_positive'] = (min(gaps) > 0)
            
            model_report[path_type] = path_report
        
        report['models'][model_name] = model_report
    
    # 保存JSON报告
    json_path = os.path.join(save_dir, 'weight_gap_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Report saved to: {json_path}")
    
    return report

# ==================== 5. 主函数 ====================

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 WEIGHT GAP COMPARISON: 0% vs 20% MIX MODELS")
    print("="*80)
    
    # 配置参数
    d_model = 92
    iterations = list(range(5000, 51000, 5000))  # 每5k一个点
    save_dir = 'weight_gap_analysis_0vs20'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📋 Configuration:")
    print(f"  Model dimension: d={d_model}")
    print(f"  Iterations: {iterations[0]} to {iterations[-1]} (step {iterations[1]-iterations[0]})")
    print(f"  Output directory: {save_dir}")
    
    # 初始化配置
    configs = {
        '0% mix': ModelConfig(mix_ratio=0, d_model=d_model),
        '20% mix': ModelConfig(mix_ratio=20, d_model=d_model)
    }
    
    # 收集演化数据
    print("\n" + "="*80)
    print("📈 COLLECTING EVOLUTION DATA")
    print("="*80)
    
    evolution_data = collect_evolution_data(configs, iterations)
    
    # 打印详细统计
    print_detailed_statistics(evolution_data, iterations)
    
    # 生成可视化
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 综合对比图（3x3）
    plot_comprehensive_comparison(evolution_data, iterations, save_dir)
    
    # 聚焦对比图（2x2）
    plot_focused_comparison(evolution_data, iterations, save_dir)
    
    # 生成报告
    report = generate_summary_report(evolution_data, iterations, save_dir)
    
    # 关键发现总结
    print("\n" + "="*80)
    print("🎯 KEY FINDINGS")
    print("="*80)
    
    # 分析S2->S3 gap（最关键）
    for model_name in ['0% mix', '20% mix']:
        s23_gaps = evolution_data[model_name]['S2->S3']['gap']
        valid_gaps = [g for g in s23_gaps if not np.isnan(g)]
        
        if valid_gaps:
            print(f"\n{model_name} Model - S2→S3 Gap:")
            print(f"  Final value: {valid_gaps[-1]:.4f}")
            print(f"  Minimum: {min(valid_gaps):.4f}")
            
            if min(valid_gaps) > 0:
                print(f"  ✅ Always positive - S2 information preserved throughout training")
            else:
                print(f"  ⚠️ Goes negative - potential S2 suppression")
    
    # 比较两个模型
    print("\n" + "="*80)
    print("🔍 COMPARATIVE ANALYSIS")
    print("="*80)
    
    gap_0 = [g for g in evolution_data['0% mix']['S2->S3']['gap'] if not np.isnan(g)]
    gap_20 = [g for g in evolution_data['20% mix']['S2->S3']['gap'] if not np.isnan(g)]
    
    if gap_0 and gap_20:
        final_diff = gap_20[-1] - gap_0[-1]
        print(f"\nS2→S3 Gap Difference (20% - 0%):")
        print(f"  Final difference: {final_diff:.4f}")
        
        if final_diff < 0:
            print(f"  → 20% model has WEAKER S2→S3 connection")
            print(f"  → Memory task interferes with compositional structure")
        else:
            print(f"  → 20% model has STRONGER S2→S3 connection")
            print(f"  → Memory task may enhance path learning")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {save_dir}/")
    print("  • weight_gap_comparison_0vs20_*.png")
    print("  • weight_gap_focused_0vs20_*.png")
    print("  • weight_gap_report.json")

if __name__ == "__main__":
    main()