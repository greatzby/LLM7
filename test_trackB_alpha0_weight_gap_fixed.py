#!/usr/bin/env python3
"""
test_trackB_alpha0_weight_gap_fixed.py
测试Track B α=0的weight gap - 使用指定的checkpoint路径
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

try:
    from model import GPTConfig, GPT
except ImportError:
    print("❌ Error: Cannot import 'model.py'")
    exit()

# ==================== 1. 配置类 ====================

class TrackBModelConfig:
    """Track B模型配置类 - 1层1头版本"""
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device('cpu')
        
        # 模型参数 - 1层1头
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 92
        self.vocab_size = 92
        
        # 使用标准图路径（Track B和ALPINE用同一个图）
        self.graph_base_dir = 'data/simple_graph/standardized_alpine_90_seed42'
        
        # 加载节点分组和图结构
        self.load_stage_info()
        self.load_graph_structure()
    
    def load_stage_info(self):
        """加载节点分组信息"""
        stage_info_path = os.path.join(self.graph_base_dir, 'stage_info.pkl')
        
        print(f"  Loading stage_info from: {stage_info_path}")
        
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
        
        print(f"  Loading graph from: {graph_path}")
        
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

def find_checkpoint(iteration, config):
    """查找checkpoint - 使用指定路径"""
    checkpoint_path = os.path.join(config.checkpoint_dir, f'ckpt_{iteration}.pt')
    
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    return None

def extract_W_M_prime(checkpoint_path, config):
    """提取W'_M矩阵 - 1层1头版本"""
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

def calculate_path_statistics(W_M_prime, config, path_type):
    """计算路径统计"""
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
    
    # 处理有边的情况
    if stats['num_edges'] > 0:
        stats['avg_edge_weight'] = np.mean(W_sub[edge_mask])
        stats['std_edge_weight'] = np.std(W_sub[edge_mask])
    else:
        stats['avg_edge_weight'] = 0  # S1->S3没有边时设为0
        stats['std_edge_weight'] = 0
    
    # 处理无边的情况
    if stats['num_non_edges'] > 0:
        stats['avg_non_edge_weight'] = np.mean(W_sub[non_edge_mask])
        stats['std_non_edge_weight'] = np.std(W_sub[non_edge_mask])
    else:
        stats['avg_non_edge_weight'] = 0
        stats['std_non_edge_weight'] = 0
    
    # 计算gap
    stats['gap'] = stats['avg_edge_weight'] - stats['avg_non_edge_weight']
    
    return stats

# ==================== 2. 数据收集和可视化 ====================

def collect_evolution_data(config, iterations):
    """收集Track B α=0的演化数据"""
    print(f"\n📊 Processing checkpoints from: {config.checkpoint_dir}")
    
    model_data = {
        'S1->S2': {'edge': [], 'non_edge': [], 'gap': []},
        'S2->S3': {'edge': [], 'non_edge': [], 'gap': []},
        'S1->S3': {'edge': [], 'non_edge': [], 'gap': []}
    }
    
    found_checkpoints = 0
    for iteration in tqdm(iterations, desc="Loading checkpoints"):
        checkpoint_path = find_checkpoint(iteration, config)
        
        if checkpoint_path is None:
            print(f"  ⚠️ Checkpoint not found: ckpt_{iteration}.pt")
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
    return model_data

def plot_trackB_results(model_data, iterations, save_dir):
    """生成Track B α=0的结果图 - 改进版"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Track B α=0 Weight Gap Analysis (1L-1H Model)', fontsize=16, fontweight='bold')
    
    path_types = ['S1->S2', 'S2->S3', 'S1->S3']
    colors = {'edge': '#2E86AB', 'non_edge': '#A23B72', 'gap': '#F18F01'}
    
    for i, path_type in enumerate(path_types):
        # 第1列：Edge权重
        ax = axes[i, 0]
        
        edge_weights = model_data[path_type]['edge']
        valid_iters = [it for it, w in zip(iterations, edge_weights) if not np.isnan(w)]
        valid_weights = [w for w in edge_weights if not np.isnan(w)]
        
        if valid_weights:
            ax.plot(valid_iters, valid_weights, marker='o', color=colors['edge'],
                   linewidth=2, markersize=6, alpha=0.8)
            # 标注最终值
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
        
        non_edge_weights = model_data[path_type]['non_edge']
        valid_iters = [it for it, w in zip(iterations, non_edge_weights) if not np.isnan(w)]
        valid_weights = [w for w in non_edge_weights if not np.isnan(w)]
        
        if valid_weights:
            ax.plot(valid_iters, valid_weights, marker='s', color=colors['non_edge'],
                   linewidth=2, markersize=6, alpha=0.8, linestyle='--')
            # 标注最终值
            ax.annotate(f'{valid_weights[-1]:.3f}', 
                       xy=(valid_iters[-1], valid_weights[-1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=colors['non_edge'])
        
        ax.set_title('Average Non-Edge Weight' if i == 0 else '', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 第3列：Weight Gap
        ax = axes[i, 2]
        
        gaps = model_data[path_type]['gap']
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
            
            # 填充正负区域
            ax.fill_between(valid_iters, 0, valid_gaps, 
                           where=np.array(valid_gaps) > 0, 
                           alpha=0.2, color='green', interpolate=True)
            ax.fill_between(valid_iters, 0, valid_gaps, 
                           where=np.array(valid_gaps) < 0, 
                           alpha=0.2, color='red', interpolate=True)
            
            # 检查gap状态
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
            
            # 设置x轴刻度
            valid_gaps = model_data[path_types[0]]['gap']
            valid_iters = [it for it, g in zip(iterations, valid_gaps) if not np.isnan(g)]
            
            if len(valid_iters) > 10:
                tick_indices = list(range(0, len(valid_iters), 2))
                axes[i, j].set_xticks([valid_iters[idx] for idx in tick_indices if idx < len(valid_iters)])
                axes[i, j].set_xticklabels([f'{k//1000}k' for k in [valid_iters[idx] for idx in tick_indices if idx < len(valid_iters)]], rotation=45)
            else:
                axes[i, j].set_xticks(valid_iters)
                axes[i, j].set_xticklabels([f'{k//1000}k' for k in valid_iters], rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'trackB_alpha0_weight_gap_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {save_path}")
    plt.show()

def print_statistics(model_data, iterations, config):
    """打印统计信息"""
    print("\n" + "="*80)
    print("📊 TRACK B α=0 FINAL STATISTICS")
    print("="*80)
    
    path_types = ['S1->S2', 'S2->S3', 'S1->S3']
    
    for path_type in path_types:
        print(f"\n{path_type}:")
        print("-" * 40)
        
        # 图结构中的边数
        print(f"  True edges in graph: {config.edge_stats[path_type]}")
        
        gaps = model_data[path_type]['gap']
        valid_gaps = [g for g in gaps if not np.isnan(g)]
        
        edge_weights = model_data[path_type]['edge']
        valid_edge = [w for w in edge_weights if not np.isnan(w)]
        
        non_edge_weights = model_data[path_type]['non_edge']
        valid_non_edge = [w for w in non_edge_weights if not np.isnan(w)]
        
        if valid_gaps:
            print(f"  Gap evolution:")
            if len(valid_gaps) > 0:
                print(f"    Initial (5k):  {valid_gaps[0]:.4f}")
            if len(valid_gaps) >= 5:
                print(f"    Mid (25k):     {valid_gaps[4]:.4f}")
            if len(valid_gaps) > 0:
                print(f"    Final:         {valid_gaps[-1]:.4f}")
                print(f"    Min:           {min(valid_gaps):.4f}")
                print(f"    Max:           {max(valid_gaps):.4f}")
            
            # 特殊检查
            if path_type == 'S2->S3':
                if min(valid_gaps) > 0:
                    print(f"    ✅ Always positive - Strong compositionality!")
                elif valid_gaps[-1] > 0:
                    print(f"    ⚠️ Recovers to positive - Weak compositionality")
                else:
                    print(f"    ❌ Ends negative - No compositionality")
            
            if path_type == 'S1->S3' and config.edge_stats['S1->S3'] == 0:
                print(f"    Note: No direct S1->S3 edges (as expected)")
                if abs(valid_gaps[-1]) < 0.01:
                    print(f"    ✅ Near zero gap - Good S2 mediation")
                else:
                    print(f"    ⚠️ Non-zero gap ({valid_gaps[-1]:.4f})")
        
        if valid_edge:
            print(f"  Final edge weight:     {valid_edge[-1]:.4f}")
        
        if valid_non_edge:
            print(f"  Final non-edge weight: {valid_non_edge[-1]:.4f}")

def save_summary_report(model_data, iterations, config, save_dir):
    """保存文本报告"""
    report_path = os.path.join(save_dir, 'report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRACK B α=0 WEIGHT GAP ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint directory: {config.checkpoint_dir}\n")
        f.write(f"Model: 1 Layer, 1 Head, 92D\n")
        f.write(f"Iterations analyzed: {iterations}\n\n")
        
        for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
            f.write(f"\n{path_type} Analysis:\n")
            f.write("-"*40 + "\n")
            
            gaps = [g for g in model_data[path_type]['gap'] if not np.isnan(g)]
            if gaps:
                f.write(f"Initial gap (5k):  {gaps[0]:.4f}\n")
                if len(gaps) > 1:
                    f.write(f"Final gap:         {gaps[-1]:.4f}\n")
                f.write(f"Min gap:           {min(gaps):.4f}\n")
                f.write(f"Max gap:           {max(gaps):.4f}\n")
                
                if path_type == 'S2->S3':
                    f.write("\nCompositionality Status: ")
                    if min(gaps) > 0:
                        f.write("✅ STRONG (always positive)\n")
                    elif gaps[-1] > 0:
                        f.write("⚠️ WEAK (recovers to positive)\n")
                    else:
                        f.write("❌ BROKEN (ends negative)\n")
    
    print(f"✅ Report saved to: {report_path}")

# ==================== 3. 主函数 ====================

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 TRACK B α=0 WEIGHT GAP ANALYSIS")
    print("="*80)
    
    # 使用您提供的checkpoint路径
    checkpoint_dir = 'out/unknown_unknown_d92_seed42_20251005_070420'
    iterations = list(range(5000, 51000, 5000))
    save_dir = 'trackB_alpha0_analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📋 Configuration:")
    print(f"  • Checkpoint directory: {checkpoint_dir}")
    print(f"  • Model: 1 Layer, 1 Head, 92D")
    print(f"  • Iterations to analyze: {iterations}")
    print(f"  • Output directory: {save_dir}/")
    
    # 验证checkpoint目录存在
    if not os.path.exists(checkpoint_dir):
        print(f"\n❌ ERROR: Checkpoint directory not found: {checkpoint_dir}")
        print("Please check the path and try again.")
        return
    
    # 列出可用的checkpoints
    available_ckpts = glob.glob(os.path.join(checkpoint_dir, 'ckpt_*.pt'))
    if available_ckpts:
        print(f"  • Found {len(available_ckpts)} checkpoint files")
        available_iters = sorted([int(f.split('_')[-1].split('.')[0]) for f in available_ckpts])
        if len(available_iters) <= 10:
            print(f"  • Available iterations: {available_iters}")
        else:
            print(f"  • Available iterations: {available_iters[:5]}...{available_iters[-5:]}")
    else:
        print(f"\n❌ ERROR: No checkpoints found in {checkpoint_dir}")
        return
    
    # 初始化配置
    print("\n" + "="*60)
    print("Loading graph structure...")
    print("="*60)
    
    try:
        config = TrackBModelConfig(checkpoint_dir)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure the following files exist:")
        print("  • data/simple_graph/standardized_alpine_90_seed42/stage_info.pkl")
        print("  • data/simple_graph/standardized_alpine_90_seed42/composition_graph.graphml")
        return
    
    # 收集数据
    print("\n" + "="*60)
    print("Collecting data from checkpoints...")
    print("="*60)
    
    model_data = collect_evolution_data(config, iterations)
    
    # 保存原始数据
    data_path = os.path.join(save_dir, 'trackB_alpha0_data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump({
            'model_data': model_data,
            'iterations': iterations,
            'checkpoint_dir': checkpoint_dir,
            'edge_stats': config.edge_stats
        }, f)
    print(f"✅ Raw data saved to: {data_path}")
    
    # 生成可视化
    print("\n" + "="*60)
    print("Generating visualization...")
    print("="*60)
    
    plot_trackB_results(model_data, iterations, save_dir)
    
    # 打印统计
    print_statistics(model_data, iterations, config)
    
    # 保存报告
    save_summary_report(model_data, iterations, config, save_dir)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print(f"📁 All results saved to: {save_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()