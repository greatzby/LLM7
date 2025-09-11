#!/usr/bin/env python3
"""
comprehensive_layer_comparison.py
全面比较0%和20%模型的所有层，找出关键差异
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import glob
from datetime import datetime

def get_checkpoint_path(d_model, mix_ratio, iteration):
    """获取checkpoint路径"""
    if mix_ratio == 0:
        pattern = f"out/composition_d{d_model}_0mix_*/ckpt_{iteration}.pt"
    else:
        pattern = f"out/composition_d{d_model}_{mix_ratio}mix_*/ckpt_{iteration}.pt"
    
    files = glob.glob(pattern)
    if files:
        return files[0]
    
    # 尝试旧版本的命名格式
    pattern_old = f"out_d{d_model}/composition_mix{mix_ratio}_seed42_*/ckpt_mix{mix_ratio}_seed42_iter{iteration}.pt"
    files = glob.glob(pattern_old)
    if files:
        return files[0]
    return None

def comprehensive_layer_comparison(d_model=92, iteration=50000):
    """
    全面比较0%和20%模型在指定迭代的所有层
    """
    print("="*80)
    print(f"🔬 Comprehensive Layer Comparison: 0% vs 20% Mix @ {iteration} iterations")
    print("="*80)
    
    # 加载checkpoints
    path_0 = get_checkpoint_path(d_model, 0, iteration)
    path_20 = get_checkpoint_path(d_model, 20, iteration)
    
    if not path_0 or not path_20:
        print(f"❌ Cannot find checkpoints!")
        print(f"   0% path: {path_0}")
        print(f"   20% path: {path_20}")
        return None
    
    print(f"Loading 0% model from: {path_0}")
    print(f"Loading 20% model from: {path_20}")
    
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    
    state_0 = ckpt_0['model'] if 'model' in ckpt_0 else ckpt_0
    state_20 = ckpt_20['model'] if 'model' in ckpt_20 else ckpt_20
    
    # 分析结果存储
    layer_results = {}
    
    print("\n📊 Layer-by-Layer Comparison")
    print("-"*100)
    print(f"{'Layer Name':<40} {'Shape':<20} {'Cosine Sim':<12} {'Norm Diff%':<12} {'Mean Cos PA':<12}")
    print("-"*100)
    
    # 遍历所有层
    for key in state_0.keys():
        if key not in state_20:
            continue
            
        w_0 = state_0[key].float()
        w_20 = state_20[key].float()
        
        # 基本统计
        norm_0 = torch.norm(w_0).item()
        norm_20 = torch.norm(w_20).item()
        norm_diff = (norm_20 - norm_0) / norm_0 * 100 if norm_0 > 0 else 0
        
        # Cosine相似度
        w_0_flat = w_0.flatten()
        w_20_flat = w_20.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            w_0_flat.unsqueeze(0), 
            w_20_flat.unsqueeze(0)
        ).item()
        
        # 对于2D矩阵，计算主角度
        mean_pa_cos = None
        if len(w_0.shape) == 2 and min(w_0.shape) >= 2:
            try:
                w_0_np = w_0.numpy()
                w_20_np = w_20.numpy()
                
                # SVD
                U_0, S_0, Vt_0 = svd(w_0_np, full_matrices=False)
                U_20, S_20, Vt_20 = svd(w_20_np, full_matrices=False)
                
                # 计算V空间的主角度（取前min(k, 30)个）
                k = min(30, Vt_0.shape[0])
                V_0 = Vt_0[:k, :].T
                V_20 = Vt_20[:k, :].T
                
                overlap = V_0.T @ V_20
                pa_cosines = svd(overlap, compute_uv=False)
                mean_pa_cos = np.mean(pa_cosines)
            except:
                mean_pa_cos = None
        
        # 存储结果
        layer_results[key] = {
            'shape': tuple(w_0.shape),
            'cosine_sim': cosine_sim,
            'norm_diff_%': norm_diff,
            'mean_pa_cos': mean_pa_cos,
            'norm_0': norm_0,
            'norm_20': norm_20
        }
        
        # 打印结果
        shape_str = str(tuple(w_0.shape))
        pa_str = f"{mean_pa_cos:.4f}" if mean_pa_cos is not None else "N/A"
        
        # 根据相似度着色输出（终端可能不支持）
        if cosine_sim > 0.95:
            marker = "✅"
        elif cosine_sim > 0.90:
            marker = "🟡"
        else:
            marker = "🔴"
            
        print(f"{marker} {key:<38} {shape_str:<20} {cosine_sim:<12.4f} {norm_diff:<12.2f} {pa_str:<12}")
    
    print("-"*100)
    
    # 分类统计
    print("\n🎯 Layer Category Analysis")
    print("-"*80)
    
    categories = {
        'embedding': [],
        'attention': [],
        'mlp/ffn': [],
        'normalization': [],
        'output': [],
        'other': []
    }
    
    for key, result in layer_results.items():
        if 'wte' in key or 'wpe' in key or 'embed' in key:
            categories['embedding'].append((key, result))
        elif 'attn' in key or 'attention' in key:
            categories['attention'].append((key, result))
        elif 'mlp' in key or 'fc' in key or 'ffn' in key:
            categories['mlp/ffn'].append((key, result))
        elif 'ln' in key or 'norm' in key or 'layernorm' in key:
            categories['normalization'].append((key, result))
        elif 'lm_head' in key or 'head' in key:
            categories['output'].append((key, result))
        else:
            categories['other'].append((key, result))
    
    for cat_name, layers in categories.items():
        if not layers:
            continue
            
        print(f"\n{cat_name.upper()}:")
        
        # 计算该类别的平均相似度
        sims = [l[1]['cosine_sim'] for l in layers]
        avg_sim = np.mean(sims) if sims else 0
        
        print(f"  Average cosine similarity: {avg_sim:.4f}")
        print(f"  Layers ({len(layers)}):")
        
        # 按相似度排序
        layers_sorted = sorted(layers, key=lambda x: x[1]['cosine_sim'], reverse=True)
        
        for name, result in layers_sorted[:5]:  # 只显示前5个
            print(f"    {name:<35} sim={result['cosine_sim']:.4f}, norm_diff={result['norm_diff_%']:.1f}%")
        
        if len(layers) > 5:
            print(f"    ... and {len(layers)-5} more layers")
    
    return layer_results

def plot_layer_similarity_heatmap(layer_results, d_model=92, iteration=50000):
    """
    绘制层相似度热图
    """
    # 准备数据
    layer_names = []
    cosine_sims = []
    norm_diffs = []
    
    for key, result in layer_results.items():
        layer_names.append(key)
        cosine_sims.append(result['cosine_sim'])
        norm_diffs.append(abs(result['norm_diff_%']))
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 相似度条形图
    y_pos = np.arange(len(layer_names))
    colors = ['green' if s > 0.95 else 'orange' if s > 0.90 else 'red' for s in cosine_sims]
    
    ax1.barh(y_pos, cosine_sims, color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(layer_names, fontsize=8)
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_title(f'Layer Similarity: 0% vs 20% Mix (d={d_model}, iter={iteration})')
    ax1.axvline(x=0.95, color='green', linestyle='--', alpha=0.5, label='High (>0.95)')
    ax1.axvline(x=0.90, color='orange', linestyle='--', alpha=0.5, label='Medium (>0.90)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 范数差异条形图
    ax2.barh(y_pos, norm_diffs, color='blue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(layer_names, fontsize=8)
    ax2.set_xlabel('Norm Difference (%)')
    ax2.set_title('Weight Norm Changes')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'layer_similarity_comparison_d{d_model}_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Figure saved as: {filename}")
    plt.show()

def analyze_key_layers_svd(d_model=92, iteration=50000):
    """
    对关键层进行深入的SVD分析（只比较最终结果）
    """
    print("\n" + "="*80)
    print("🔬 Deep SVD Analysis of Key Layers")
    print("="*80)
    
    # 加载模型
    path_0 = get_checkpoint_path(d_model, 0, iteration)
    path_20 = get_checkpoint_path(d_model, 20, iteration)
    
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    
    state_0 = ckpt_0['model'] if 'model' in ckpt_0 else ckpt_0
    state_20 = ckpt_20['model'] if 'model' in ckpt_20 else ckpt_20
    
    # 选择关键层进行分析
    key_layers = ['lm_head.weight', 'transformer.wte.weight']  # 根据实际模型调整
    
    fig, axes = plt.subplots(1, len(key_layers), figsize=(6*len(key_layers), 5))
    if len(key_layers) == 1:
        axes = [axes]
    
    for idx, layer_key in enumerate(key_layers):
        # 查找实际的键名
        actual_key = None
        for key in state_0.keys():
            if layer_key in key or key.endswith(layer_key):
                actual_key = key
                break
        
        if actual_key is None:
            print(f"⚠️ Layer {layer_key} not found")
            continue
        
        print(f"\n📊 Analyzing: {actual_key}")
        
        w_0 = state_0[actual_key].float().numpy()
        w_20 = state_20[actual_key].float().numpy()
        
        # SVD
        _, S_0, _ = svd(w_0, full_matrices=False)
        _, S_20, _ = svd(w_20, full_matrices=False)
        
        # 绘制奇异值谱
        ax = axes[idx]
        n_show = min(50, len(S_0))
        ax.semilogy(S_0[:n_show], 'b-', label='0% mix', linewidth=2, alpha=0.7)
        ax.semilogy(S_20[:n_show], 'r--', label='20% mix', linewidth=2, alpha=0.7)
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Singular Value (log scale)')
        ax.set_title(f'{actual_key}\nShape: {w_0.shape}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 计算有效秩
        energy_0 = np.cumsum(S_0**2) / np.sum(S_0**2)
        energy_20 = np.cumsum(S_20**2) / np.sum(S_20**2)
        rank_90_0 = np.argmax(energy_0 >= 0.9) + 1
        rank_90_20 = np.argmax(energy_20 >= 0.9) + 1
        
        print(f"  Effective rank (90% energy):")
        print(f"    0% mix: {rank_90_0}")
        print(f"    20% mix: {rank_90_20}")
        print(f"  Top 5 singular values:")
        print(f"    0% mix: {S_0[:5]}")
        print(f"    20% mix: {S_20[:5]}")
    
    plt.suptitle(f'Singular Value Comparison @ {iteration} iterations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'svd_comparison_d{d_model}_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ SVD figure saved as: {filename}")
    plt.show()

# 主程序
if __name__ == "__main__":
    # 运行全面比较
    layer_results = comprehensive_layer_comparison(d_model=92, iteration=50000)
    
    if layer_results:
        # 绘制相似度热图
        plot_layer_similarity_heatmap(layer_results)
        
        # 深入SVD分析
        analyze_key_layers_svd(d_model=92, iteration=50000)
        
        # 统计总结
        print("\n" + "="*80)
        print("📊 FINAL SUMMARY")
        print("="*80)
        
        all_sims = [r['cosine_sim'] for r in layer_results.values()]
        print(f"\nOverall Statistics:")
        print(f"  Mean cosine similarity: {np.mean(all_sims):.4f}")
        print(f"  Std deviation: {np.std(all_sims):.4f}")
        print(f"  Min similarity: {np.min(all_sims):.4f}")
        print(f"  Max similarity: {np.max(all_sims):.4f}")
        print(f"  Layers with sim > 0.95: {sum(1 for s in all_sims if s > 0.95)}/{len(all_sims)}")
        print(f"  Layers with sim > 0.90: {sum(1 for s in all_sims if s > 0.90)}/{len(all_sims)}")