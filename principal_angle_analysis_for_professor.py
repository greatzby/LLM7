#!/usr/bin/env python3
"""
principal_angle_analysis_for_professor.py

完整回答教授的第二个要求：
1. 计算0% vs 20% model权重矩阵的主角度
2. 对U和V空间都进行分析
3. 绘制奇异值谱的演化
4. 明确说明分析的是哪个权重矩阵
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd, orthogonal_procrustes
import matplotlib.pyplot as plt
from datetime import datetime

# ============ 配置参数 ============
D_MODEL = 92  # 或120，根据你的实验
SEEDS = [42]  # 固定种子，比较0% vs 20%
ITERATIONS = [5000, 10000, 20000, 30000, 40000, 50000]  # 多个时间点
K_ANALYSIS = 60  # top-k维度分析

# ============ 数据加载函数 ============
def get_checkpoint_path(d_model, mix_ratio, iteration):
    """获取checkpoint路径"""
    # 根据你的目录结构调整
    if mix_ratio == 0:
        pattern = f"out/composition_d{d_model}_0mix_*/ckpt_{iteration}.pt"
    else:
        pattern = f"out/composition_d{d_model}_{mix_ratio}mix_*/ckpt_{iteration}.pt"
    
    files = glob.glob(pattern)
    if files:
        return files[0]
    return None

def load_model_weights(d_model, mix_ratio, iteration):
    """加载模型的所有权重矩阵"""
    path = get_checkpoint_path(d_model, mix_ratio, iteration)
    if not path:
        print(f"⚠️ Cannot find checkpoint: d={d_model}, mix={mix_ratio}%, iter={iteration}")
        return None
    
    try:
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        
        # 提取不同的权重矩阵
        weights = {}
        
        # 1. Embedding and LM head
        for key in state_dict.keys():
            if 'wte.weight' in key:
                weights['embedding'] = state_dict[key].float().numpy()
            elif 'lm_head.weight' in key:
                weights['lm_head'] = state_dict[key].float().numpy()
            
            # 2. Attention weights (如果是Transformer)
            elif 'attn.c_attn.weight' in key:  # GPT-style combined QKV
                weights['attn_qkv'] = state_dict[key].float().numpy()
            elif 'attn.q_proj.weight' in key:
                weights['attn_q'] = state_dict[key].float().numpy()
            elif 'attn.k_proj.weight' in key:
                weights['attn_k'] = state_dict[key].float().numpy()
            elif 'attn.v_proj.weight' in key:
                weights['attn_v'] = state_dict[key].float().numpy()
            elif 'attn.c_proj.weight' in key:
                weights['attn_out'] = state_dict[key].float().numpy()
            
            # 3. FFN weights
            elif 'mlp.c_fc.weight' in key or 'mlp.fc1.weight' in key:
                weights['ffn_1'] = state_dict[key].float().numpy()
            elif 'mlp.c_proj.weight' in key or 'mlp.fc2.weight' in key:
                weights['ffn_2'] = state_dict[key].float().numpy()
        
        return weights
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None

# ============ 主角度计算 ============
def compute_principal_angles(W1, W2, k=None):
    """
    计算两个矩阵的主角度（对U和V空间）
    返回主角度的余弦值
    """
    # SVD分解
    U1, S1, Vt1 = svd(W1, full_matrices=False)
    U2, S2, Vt2 = svd(W2, full_matrices=False)
    
    if k is None:
        k = min(U1.shape[1], U2.shape[1], K_ANALYSIS)
    
    results = {}
    
    # V空间的主角度
    V1 = Vt1[:k, :].T
    V2 = Vt2[:k, :].T
    overlap_V = V1.T @ V2
    cosines_V = svd(overlap_V, compute_uv=False)
    results['V_cosines'] = np.clip(cosines_V, 0, 1)
    
    # U空间的主角度
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]
    overlap_U = U1_k.T @ U2_k
    cosines_U = svd(overlap_U, compute_uv=False)
    results['U_cosines'] = np.clip(cosines_U, 0, 1)
    
    # 返回奇异值用于绘图
    results['S1'] = S1
    results['S2'] = S2
    
    return results

# ============ Procrustes对齐（可选） ============
def align_with_procrustes(W1, W2, k=None):
    """
    使用Procrustes分析对齐两个矩阵的子空间
    """
    U1, S1, Vt1 = svd(W1, full_matrices=False)
    U2, S2, Vt2 = svd(W2, full_matrices=False)
    
    if k is None:
        k = min(U1.shape[1], U2.shape[1], K_ANALYSIS)
    
    # V空间对齐
    V1 = Vt1[:k, :].T
    V2 = Vt2[:k, :].T
    R_V, _ = orthogonal_procrustes(V2, V1)
    V2_aligned = V2 @ R_V
    
    # U空间对齐
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]
    R_U, _ = orthogonal_procrustes(U2_k, U1_k)
    U2_aligned = U2_k @ R_U
    
    # 对齐后的主角度
    overlap_V_aligned = V1.T @ V2_aligned
    cosines_V_aligned = svd(overlap_V_aligned, compute_uv=False)
    
    overlap_U_aligned = U1_k.T @ U2_aligned
    cosines_U_aligned = svd(overlap_U_aligned, compute_uv=False)
    
    return {
        'V_cosines_aligned': np.clip(cosines_V_aligned, 0, 1),
        'U_cosines_aligned': np.clip(cosines_U_aligned, 0, 1),
        'R_V': R_V,
        'R_U': R_U
    }

# ============ 主分析和绘图函数 ============
def analyze_and_plot():
    """完整的分析流程"""
    
    print("="*80)
    print("🔬 Principal Angle Analysis: 0% vs 20% Mix Models")
    print("="*80)
    
    # 选择要分析的权重矩阵
    weight_key = 'lm_head'  # 可以改为其他层
    print(f"\n📌 Analyzing weight matrix: {weight_key}")
    print("   (Available: embedding, lm_head, attn_*, ffn_*)")
    
    # 收集所有迭代的结果
    all_results = {}
    
    for iteration in ITERATIONS:
        print(f"\n⏱️ Processing iteration {iteration}...")
        
        # 加载0%和20%模型
        weights_0 = load_model_weights(D_MODEL, 0, iteration)
        weights_20 = load_model_weights(D_MODEL, 20, iteration)
        
        if weights_0 is None or weights_20 is None:
            print(f"   Skip iteration {iteration} (missing data)")
            continue
        
        if weight_key not in weights_0 or weight_key not in weights_20:
            print(f"   Weight matrix '{weight_key}' not found!")
            continue
        
        W1 = weights_0[weight_key]
        W2 = weights_20[weight_key]
        
        # 计算主角度
        results = compute_principal_angles(W1, W2)
        
        # 可选：Procrustes对齐
        aligned_results = align_with_procrustes(W1, W2)
        results.update(aligned_results)
        
        all_results[iteration] = results
        
        # 打印摘要
        mean_cos_V = np.mean(results['V_cosines'])
        mean_cos_U = np.mean(results['U_cosines'])
        mean_cos_V_aligned = np.mean(aligned_results['V_cosines_aligned'])
        mean_cos_U_aligned = np.mean(aligned_results['U_cosines_aligned'])
        
        print(f"   V-space: mean cos = {mean_cos_V:.4f} → {mean_cos_V_aligned:.4f} (after alignment)")
        print(f"   U-space: mean cos = {mean_cos_U:.4f} → {mean_cos_U_aligned:.4f} (after alignment)")
    
    # ============ 绘图 ============
    if not all_results:
        print("\n❌ No data to plot!")
        return
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 主角度余弦谱（V空间）
    ax1 = plt.subplot(2, 3, 1)
    for i, (iter_val, results) in enumerate(all_results.items()):
        color = plt.cm.viridis(i / len(all_results))
        ax1.plot(results['V_cosines'], label=f'Iter {iter_val//1000}k', 
                 color=color, linewidth=2, alpha=0.7)
    ax1.set_xlabel('Principal Angle Index')
    ax1.set_ylabel('Cosine Value')
    ax1.set_title(f'V-Space Principal Angles\n({weight_key})', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # 2. 主角度余弦谱（U空间）
    ax2 = plt.subplot(2, 3, 2)
    for i, (iter_val, results) in enumerate(all_results.items()):
        color = plt.cm.viridis(i / len(all_results))
        ax2.plot(results['U_cosines'], label=f'Iter {iter_val//1000}k', 
                 color=color, linewidth=2, alpha=0.7)
    ax2.set_xlabel('Principal Angle Index')
    ax2.set_ylabel('Cosine Value')
    ax2.set_title(f'U-Space Principal Angles\n({weight_key})', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # 3. 奇异值谱比较
    ax3 = plt.subplot(2, 3, 3)
    for i, (iter_val, results) in enumerate(all_results.items()):
        color = plt.cm.plasma(i / len(all_results))
        ax3.semilogy(results['S1'][:50], '--', label=f'0% mix @{iter_val//1000}k', 
                     color=color, alpha=0.6)
        ax3.semilogy(results['S2'][:50], '-', label=f'20% mix @{iter_val//1000}k', 
                     color=color, alpha=0.8)
    ax3.set_xlabel('Singular Value Index')
    ax3.set_ylabel('Singular Value (log scale)')
    ax3.set_title('Singular Value Spectrum Evolution', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. 平均余弦值演化（V空间）
    ax4 = plt.subplot(2, 3, 4)
    iters = sorted(all_results.keys())
    mean_cos_V = [np.mean(all_results[it]['V_cosines']) for it in iters]
    mean_cos_V_aligned = [np.mean(all_results[it]['V_cosines_aligned']) for it in iters]
    ax4.plot(iters, mean_cos_V, 'o-', label='Before Alignment', linewidth=2, markersize=8)
    ax4.plot(iters, mean_cos_V_aligned, 's-', label='After Alignment', linewidth=2, markersize=8)
    ax4.set_xlabel('Training Iteration')
    ax4.set_ylabel('Mean Cosine')
    ax4.set_title('V-Space Alignment Evolution', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 平均余弦值演化（U空间）
    ax5 = plt.subplot(2, 3, 5)
    mean_cos_U = [np.mean(all_results[it]['U_cosines']) for it in iters]
    mean_cos_U_aligned = [np.mean(all_results[it]['U_cosines_aligned']) for it in iters]
    ax5.plot(iters, mean_cos_U, 'o-', label='Before Alignment', linewidth=2, markersize=8)
    ax5.plot(iters, mean_cos_U_aligned, 's-', label='After Alignment', linewidth=2, markersize=8)
    ax5.set_xlabel('Training Iteration')
    ax5.set_ylabel('Mean Cosine')
    ax5.set_title('U-Space Alignment Evolution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 对齐改善度
    ax6 = plt.subplot(2, 3, 6)
    improve_V = [np.mean(all_results[it]['V_cosines_aligned']) - np.mean(all_results[it]['V_cosines']) 
                 for it in iters]
    improve_U = [np.mean(all_results[it]['U_cosines_aligned']) - np.mean(all_results[it]['U_cosines']) 
                 for it in iters]
    ax6.plot(iters, improve_V, 'o-', label='V-space', linewidth=2, markersize=8)
    ax6.plot(iters, improve_U, 's-', label='U-space', linewidth=2, markersize=8)
    ax6.set_xlabel('Training Iteration')
    ax6.set_ylabel('Alignment Improvement')
    ax6.set_title('Procrustes Alignment Benefit', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.suptitle(f'Principal Angle Analysis: 0% vs 20% Mix (d={D_MODEL}, {weight_key})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'principal_angle_analysis_{weight_key}_d{D_MODEL}_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved as: {filename}")
    
    plt.show()
    
    # ============ 总结报告 ============
    print("\n" + "="*80)
    print("📊 SUMMARY REPORT")
    print("="*80)
    
    print(f"\n1. Weight Matrix Analyzed: {weight_key}")
    print(f"   - Model dimension: d={D_MODEL}")
    print(f"   - Top-k analysis: k={K_ANALYSIS}")
    
    print("\n2. Key Findings:")
    
    # 最终迭代的结果
    if 50000 in all_results:
        final = all_results[50000]
        print(f"\n   At iteration 50000:")
        print(f"   - V-space mean cosine: {np.mean(final['V_cosines']):.4f}")
        print(f"   - U-space mean cosine: {np.mean(final['U_cosines']):.4f}")
        print(f"   - After Procrustes alignment:")
        print(f"     * V-space: {np.mean(final['V_cosines_aligned']):.4f}")
        print(f"     * U-space: {np.mean(final['U_cosines_aligned']):.4f}")
        
        # 主角度分布
        angles_V = np.arccos(np.clip(final['V_cosines'], -1, 1)) * 180 / np.pi
        angles_U = np.arccos(np.clip(final['U_cosines'], -1, 1)) * 180 / np.pi
        
        print(f"\n   Principal angles (degrees):")
        print(f"   - V-space: min={angles_V.min():.1f}°, max={angles_V.max():.1f}°, mean={angles_V.mean():.1f}°")
        print(f"   - U-space: min={angles_U.min():.1f}°, max={angles_U.max():.1f}°, mean={angles_U.mean():.1f}°")
    
    print("\n3. Interpretation:")
    print("   - High cosine values (>0.9) indicate strong alignment between 0% and 20% models")
    print("   - Procrustes alignment reveals hidden similarities after rotation")
    print("   - Different weight matrices show different alignment patterns")
    
    print("\n4. Recommendation:")
    print("   Consider analyzing other weight matrices:")
    print("   - attn_q, attn_k, attn_v: Attention mechanism")
    print("   - ffn_1, ffn_2: Feed-forward network")
    print("   - embedding: Input representations")
    
    print("="*80)

# ============ 运行主程序 ============
if __name__ == "__main__":
    analyze_and_plot()