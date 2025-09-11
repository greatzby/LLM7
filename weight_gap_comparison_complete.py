#!/usr/bin/env python3
"""
weight_gap_comparison_complete.py
完整版 - 包含S1->S3分析（即使没有真实边）
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

# [前面的ModelConfig, find_checkpoint, extract_W_M_prime等函数保持不变]
# ... [使用之前的代码]

def plot_complete_analysis(data_0mix, data_20mix, iterations, save_dir):
    """生成完整的对比分析图 - 包含所有三种路径"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Weight Gap Evolution: 0% vs 20% Mix Models', fontsize=16, fontweight='bold')
    
    path_types = ['S1->S2', 'S2->S3', 'S1->S3']
    colors = {'0% mix': 'blue', '20% mix': 'red'}
    
    for i, path_type in enumerate(path_types):
        # 第1列：Edge权重 (对S1->S3显示non-edge权重，因为没有真实边)
        ax = axes[i, 0]
        
        if path_type != 'S1->S3':
            # 正常的edge权重
            for model_name, data in [('0% mix', data_0mix), ('20% mix', data_20mix)]:
                edge_weights = data[path_type]['edge']
                valid_iters = [it for it, w in zip(iterations, edge_weights) if not np.isnan(w)]
                valid_weights = [w for w in edge_weights if not np.isnan(w)]
                
                if valid_weights:
                    ax.plot(valid_iters, valid_weights, marker='o', label=model_name,
                           color=colors[model_name], linewidth=2, markersize=5, alpha=0.8)
            
            ax.set_title('Average Edge Weight' if i == 0 else '', fontsize=13)
        else:
            # S1->S3: 显示所有权重的平均值（因为都是non-edge）
            for model_name, data in [('0% mix', data_0mix), ('20% mix', data_20mix)]:
                non_edge_weights = data[path_type]['non_edge']
                valid_iters = [it for it, w in zip(iterations, non_edge_weights) if not np.isnan(w)]
                valid_weights = [w for w in non_edge_weights if not np.isnan(w)]
                
                if valid_weights:
                    ax.plot(valid_iters, valid_weights, marker='o', label=model_name,
                           color=colors[model_name], linewidth=2, markersize=5, alpha=0.8)
            
            ax.set_title('Avg Weight (No true edges)' if i == 0 else '', fontsize=13)
            ax.text(0.5, 0.95, 'No direct S1→S3 edges in ground truth',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=10, color='gray', style='italic')
        
        ax.set_ylabel(f'{path_type}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc='best')
        
        # 第2列：Non-edge权重的标准差（衡量权重分布）
        ax = axes[i, 1]
        
        for model_name, data in [('0% mix', data_0mix), ('20% mix', data_20mix)]:
            # 计算每个时间点的标准差
            stds = []
            valid_iters_std = []
            
            for idx, iter_val in enumerate(iterations):
                if idx < len(data[path_type]['non_edge']):
                    non_edge_val = data[path_type]['non_edge'][idx]
                    if not np.isnan(non_edge_val):
                        # 这里简化处理，显示non-edge权重
                        stds.append(non_edge_val)
                        valid_iters_std.append(iter_val)
            
            if stds:
                ax.plot(valid_iters_std, stds, marker='s', label=model_name,
                       color=colors[model_name], linewidth=2, markersize=5, 
                       alpha=0.8, linestyle='--')
        
        ax.set_title('Average Non-Edge Weight' if i == 0 else '', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc='best')
        
        # 第3列：Weight Gap（对S1->S3特殊处理）
        ax = axes[i, 2]
        
        if path_type != 'S1->S3':
            # 正常的gap
            for model_name, data in [('0% mix', data_0mix), ('20% mix', data_20mix)]:
                gaps = data[path_type]['gap']
                valid_iters = [it for it, g in zip(iterations, gaps) if not np.isnan(g)]
                valid_gaps = [g for g in gaps if not np.isnan(g)]
                
                if valid_gaps:
                    ax.plot(valid_iters, valid_gaps, marker='^', label=model_name,
                           color=colors[model_name], linewidth=2.5, markersize=7)
                    
                    # 标注最终值
                    final_gap = valid_gaps[-1]
                    ax.annotate(f'{final_gap:.3f}', 
                              xy=(valid_iters[-1], final_gap),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, color=colors[model_name])
                    
                    # S2->S3特殊标注
                    if path_type == 'S2->S3' and min(valid_gaps) > 0:
                        ax.text(0.95, 0.05, '✅ Always positive', 
                               transform=ax.transAxes, ha='right', va='bottom',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        else:
            # S1->S3: 显示权重相对于0的偏离
            for model_name, data in [('0% mix', data_0mix), ('20% mix', data_20mix)]:
                non_edge_weights = data[path_type]['non_edge']
                valid_iters = [it for it, w in zip(iterations, non_edge_weights) if not np.isnan(w)]
                valid_weights = [w for w in non_edge_weights if not np.isnan(w)]
                
                if valid_weights:
                    ax.plot(valid_iters, valid_weights, marker='^', label=model_name,
                           color=colors[model_name], linewidth=2.5, markersize=7, alpha=0.7)
                    
                    # 标注最终值
                    final_weight = valid_weights[-1]
                    ax.annotate(f'{final_weight:.3f}', 
                              xy=(valid_iters[-1], final_weight),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, color=colors[model_name])
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
            ax.text(0.5, 0.95, 'Weight for non-existent edges\n(should be near 0)',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=10, color='gray', style='italic')
        
        ax.set_title('Weight Gap (Edge - Non-Edge)' if i == 0 else '', fontsize=13)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='best')
    
    # 设置x轴
    for i in range(3):
        for j in range(3):
            axes[i, j].set_xlabel('Training Iterations', fontsize=11)
            axes[i, j].set_xticks(iterations[::2])
            axes[i, j].set_xticklabels([f'{k//1000}k' for k in iterations[::2]], rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'weight_gap_complete_comparison_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Complete analysis plot saved to: {save_path}")
    plt.show()

def main():
    """主函数 - 完整分析"""
    print("\n" + "="*80)
    print("🔬 COMPLETE WEIGHT GAP ANALYSIS WITH ALL PATHS")
    print("="*80)
    
    d_model = 92
    iterations = list(range(5000, 51000, 5000))
    save_dir = 'weight_gap_analysis_complete'
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集两个模型的数据
    all_data = {}
    
    for mix_ratio in [0, 20]:
        print(f"\n{'='*60}")
        print(f"Analyzing {mix_ratio}% mix model")
        print('='*60)
        
        config = ModelConfig(mix_ratio=mix_ratio, d_model=d_model)
        evolution_data = collect_evolution_data(config, iterations)
        all_data[f'{mix_ratio}% mix'] = evolution_data
        
        # 打印详细统计
        print(f"\n📊 Detailed Statistics for {mix_ratio}% mix model:")
        print("-" * 50)
        
        for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
            print(f"\n{path_type}:")
            
            if path_type != 'S1->S3':
                # 正常路径分析
                gaps = evolution_data[path_type]['gap']
                valid_gaps = [g for g in gaps if not np.isnan(g)]
                
                if valid_gaps:
                    print(f"  Gap statistics:")
                    print(f"    Initial: {valid_gaps[0]:.4f}")
                    print(f"    Final:   {valid_gaps[-1]:.4f}")
                    print(f"    Min:     {min(valid_gaps):.4f}")
                    print(f"    Max:     {max(valid_gaps):.4f}")
                    
                    if path_type == 'S2->S3':
                        if min(valid_gaps) > 0:
                            print(f"    ✅ Always positive - S2 preserved!")
                        else:
                            print(f"    ⚠️ Goes negative - S2 may be suppressed")
            else:
                # S1->S3分析（无真实边）
                non_edge_weights = evolution_data[path_type]['non_edge']
                valid_weights = [w for w in non_edge_weights if not np.isnan(w)]
                
                if valid_weights:
                    print(f"  Weight statistics (no true edges):")
                    print(f"    Initial: {valid_weights[0]:.4f}")
                    print(f"    Final:   {valid_weights[-1]:.4f}")
                    print(f"    Min:     {min(valid_weights):.4f}")
                    print(f"    Max:     {max(valid_weights):.4f}")
                    
                    # 检查是否学习了虚假连接
                    threshold = 0.1  # 权重阈值
                    if max(np.abs(valid_weights)) > threshold:
                        print(f"    ⚠️ Model assigns significant weight to non-existent S1->S3 paths!")
                    else:
                        print(f"    ✅ Weights remain small (< {threshold})")
    
    # 生成完整对比图
    print("\n" + "="*80)
    print("📊 GENERATING COMPLETE COMPARISON PLOT")
    print("="*80)
    
    plot_complete_analysis(
        all_data['0% mix'], 
        all_data['20% mix'], 
        iterations, 
        save_dir
    )
    
    # 生成单独的详细图
    for model_name, data in all_data.items():
        plot_weight_gap_evolution(data, iterations, save_dir, model_name)
    
    print("\n" + "="*80)
    print("✅ COMPLETE ANALYSIS FINISHED!")
    print("="*80)

if __name__ == "__main__":
    main()