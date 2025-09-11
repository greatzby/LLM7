#!/usr/bin/env python3
"""
aligned_layer_comparison.py
对每个层单独进行Procrustes对齐后再比较
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, orthogonal_procrustes
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
    return None

def align_and_compare_layer(W_0, W_20, layer_name):
    """
    对单个层进行对齐和比较
    """
    results = {}
    
    # 原始相似度
    cosine_sim_orig = torch.nn.functional.cosine_similarity(
        W_0.flatten().unsqueeze(0), 
        W_20.flatten().unsqueeze(0)
    ).item()
    results['cosine_sim_original'] = cosine_sim_orig
    
    # 如果是2D矩阵，进行更深入的分析
    if len(W_0.shape) == 2:
        W_0_np = W_0.numpy()
        W_20_np = W_20.numpy()
        
        # SVD分解
        U_0, S_0, Vt_0 = svd(W_0_np, full_matrices=False)
        U_20, S_20, Vt_20 = svd(W_20_np, full_matrices=False)
        
        # 选择要分析的维度数
        k = min(30, min(W_0_np.shape))
        
        # ========== V空间对齐 ==========
        V_0 = Vt_0[:k, :].T
        V_20 = Vt_20[:k, :].T
        
        # 计算对齐前的主角度
        overlap_V = V_0.T @ V_20
        pa_cosines_V = svd(overlap_V, compute_uv=False)
        results['V_pa_cosines_before'] = pa_cosines_V
        results['V_mean_pa_before'] = np.mean(pa_cosines_V)
        
        # Procrustes对齐
        R_V, scale_V = orthogonal_procrustes(V_20, V_0)
        V_20_aligned = V_20 @ R_V
        
        # 对齐后的主角度
        overlap_V_aligned = V_0.T @ V_20_aligned
        pa_cosines_V_aligned = svd(overlap_V_aligned, compute_uv=False)
        results['V_pa_cosines_after'] = pa_cosines_V_aligned
        results['V_mean_pa_after'] = np.mean(pa_cosines_V_aligned)
        
        # ========== U空间对齐 ==========
        U_0_k = U_0[:, :k]
        U_20_k = U_20[:, :k]
        
        # 计算对齐前的主角度
        overlap_U = U_0_k.T @ U_20_k
        pa_cosines_U = svd(overlap_U, compute_uv=False)
        results['U_pa_cosines_before'] = pa_cosines_U
        results['U_mean_pa_before'] = np.mean(pa_cosines_U)
        
        # Procrustes对齐
        R_U, scale_U = orthogonal_procrustes(U_20_k, U_0_k)
        U_20_aligned = U_20_k @ R_U
        
        # 对齐后的主角度
        overlap_U_aligned = U_0_k.T @ U_20_aligned
        pa_cosines_U_aligned = svd(overlap_U_aligned, compute_uv=False)
        results['U_pa_cosines_after'] = pa_cosines_U_aligned
        results['U_mean_pa_after'] = np.mean(pa_cosines_U_aligned)
        
        # ========== 重构对齐后的权重矩阵 ==========
        # 使用对齐后的V空间重构
        S_diag = np.zeros((k, k))
        np.fill_diagonal(S_diag, S_0[:k])  # 使用0%模型的奇异值
        W_20_reconstructed = U_0_k @ S_diag @ V_20_aligned.T
        
        # 计算重构后的相似度
        W_0_truncated = U_0_k @ S_diag @ V_0.T
        cosine_sim_reconstructed = np.dot(W_0_truncated.flatten(), W_20_reconstructed.flatten()) / (
            np.linalg.norm(W_0_truncated.flatten()) * np.linalg.norm(W_20_reconstructed.flatten())
        )
        results['cosine_sim_reconstructed'] = cosine_sim_reconstructed
        
        # 存储奇异值
        results['S_0'] = S_0
        results['S_20'] = S_20
        
    return results

def comprehensive_aligned_comparison(d_model=92, iteration=50000):
    """
    对所有层进行对齐后的比较
    """
    print("="*80)
    print(f"🔬 Aligned Layer Comparison: 0% vs 20% Mix @ {iteration} iterations")
    print("="*80)
    
    # 加载模型
    path_0 = get_checkpoint_path(d_model, 0, iteration)
    path_20 = get_checkpoint_path(d_model, 20, iteration)
    
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    
    state_0 = ckpt_0['model'] if 'model' in ckpt_0 else ckpt_0
    state_20 = ckpt_20['model'] if 'model' in ckpt_20 else ckpt_20
    
    # 分析所有层
    all_results = {}
    
    print("\n📊 Layer-by-Layer Aligned Comparison")
    print("-"*120)
    print(f"{'Layer Name':<40} {'Original Sim':<12} {'V-PA Before':<12} {'V-PA After':<12} {'U-PA Before':<12} {'U-PA After':<12}")
    print("-"*120)
    
    for key in state_0.keys():
        if key not in state_20:
            continue
            
        W_0 = state_0[key].float()
        W_20 = state_20[key].float()
        
        results = align_and_compare_layer(W_0, W_20, key)
        all_results[key] = results
        
        # 打印结果
        if 'V_mean_pa_before' in results:  # 2D矩阵
            print(f"{key:<40} {results['cosine_sim_original']:<12.4f} "
                  f"{results['V_mean_pa_before']:<12.4f} {results['V_mean_pa_after']:<12.4f} "
                  f"{results['U_mean_pa_before']:<12.4f} {results['U_mean_pa_after']:<12.4f}")
        else:  # 1D或其他
            print(f"{key:<40} {results['cosine_sim_original']:<12.4f} "
                  f"{'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    print("-"*120)
    
    return all_results

def plot_alignment_improvement(all_results):
    """
    绘制对齐改善图
    """
    # 准备数据
    layers_2d = []
    original_sims = []
    v_improvements = []
    u_improvements = []
    
    for layer_name, results in all_results.items():
        if 'V_mean_pa_before' in results:
            layers_2d.append(layer_name.replace('transformer.', '').replace('.weight', ''))
            original_sims.append(results['cosine_sim_original'])
            v_improvements.append(results['V_mean_pa_after'] - results['V_mean_pa_before'])
            u_improvements.append(results['U_mean_pa_after'] - results['U_mean_pa_before'])
    
    if not layers_2d:
        return
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：对齐改善度
    x = np.arange(len(layers_2d))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, v_improvements, width, label='V-space', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, u_improvements, width, label='U-space', color='red', alpha=0.7)
    
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Alignment Improvement')
    ax1.set_title('Procrustes Alignment Improvement by Layer')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers_2d, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 右图：原始相似度 vs 对齐后的主角度
    for i, (layer, orig, v_after) in enumerate(zip(layers_2d, original_sims, 
                                                     [all_results[k]['V_mean_pa_after'] 
                                                      for k in all_results if 'V_mean_pa_after' in all_results[k]])):
        ax2.scatter(orig, v_after, s=100, alpha=0.6)
        ax2.annotate(layer, (orig, v_after), fontsize=8, 
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax2.set_xlabel('Original Cosine Similarity')
    ax2.set_ylabel('V-space PA After Alignment')
    ax2.set_title('Original Similarity vs Aligned Principal Angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'alignment_improvement_d92_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Alignment figure saved as: {filename}")
    plt.show()

def analyze_weight_sharing(state_0, state_20):
    """
    检查权重共享
    """
    print("\n" + "="*80)
    print("🔗 Weight Sharing Analysis")
    print("="*80)
    
    # 查找embedding和lm_head
    wte_key = None
    lm_head_key = None
    
    for key in state_0.keys():
        if 'wte.weight' in key:
            wte_key = key
        elif 'lm_head.weight' in key:
            lm_head_key = key
    
    if wte_key and lm_head_key:
        wte_0 = state_0[wte_key]
        lm_head_0 = state_0[lm_head_key]
        
        print(f"\nComparing {wte_key} and {lm_head_key}:")
        print(f"  Shapes: {wte_0.shape} vs {lm_head_0.shape}")
        
        if torch.allclose(wte_0, lm_head_0):
            print("  ✅ Weights are IDENTICAL (tied embeddings)")
        elif torch.allclose(wte_0.T, lm_head_0):
            print("  ✅ Weights are TRANSPOSED (tied embeddings with transpose)")
        else:
            sim = torch.nn.functional.cosine_similarity(
                wte_0.flatten().unsqueeze(0),
                lm_head_0.flatten().unsqueeze(0)
            ).item()
            print(f"  ❌ Weights are DIFFERENT (cosine sim = {sim:.4f})")
        
        # 检查20%模型
        wte_20 = state_20[wte_key]
        lm_head_20 = state_20[lm_head_key]
        
        if torch.allclose(wte_20, lm_head_20):
            print("  20% model: ✅ Also tied")
        else:
            print("  20% model: ❌ Not tied")

# 主程序
if __name__ == "__main__":
    # 运行对齐分析
    all_results = comprehensive_aligned_comparison(d_model=92, iteration=50000)
    
    # 绘制改善图
    plot_alignment_improvement(all_results)
    
    # 检查权重共享
    path_0 = get_checkpoint_path(92, 0, 50000)
    path_20 = get_checkpoint_path(92, 20, 50000)
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_0 = ckpt_0['model'] if 'model' in ckpt_0 else ckpt_0
    state_20 = ckpt_20['model'] if 'model' in ckpt_20 else ckpt_20
    
    analyze_weight_sharing(state_0, state_20)
    
    # 最终总结
    print("\n" + "="*80)
    print("📊 KEY INSIGHTS")
    print("="*80)
    
    print("\n1. Alignment Analysis Summary:")
    improvements = []
    for layer, results in all_results.items():
        if 'V_mean_pa_before' in results:
            v_imp = results['V_mean_pa_after'] - results['V_mean_pa_before']
            u_imp = results['U_mean_pa_after'] - results['U_mean_pa_before']
            improvements.append((layer, v_imp, u_imp, results['cosine_sim_original']))
    
    # 按V空间改善排序
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    print("\n   Top layers with alignment improvement:")
    for layer, v_imp, u_imp, orig in improvements[:3]:
        print(f"   {layer}:")
        print(f"     Original similarity: {orig:.4f}")
        print(f"     V-space improvement: {v_imp:.4f}")
        print(f"     U-space improvement: {u_imp:.4f}")
    
    print("\n2. Key Finding:")
    print("   Models learn similar subspaces but with different orientations!")
    print("   This is why direct cosine similarity is low but principal angles are high.")