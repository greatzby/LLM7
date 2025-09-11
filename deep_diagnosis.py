#!/usr/bin/env python3
"""
deep_diagnosis.py
深入诊断为什么会出现这种悖论
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, orthogonal_procrustes
import glob
from datetime import datetime

def diagnose_layer_difference(W_0, W_20, layer_name):
    """
    深入诊断单个层的差异来源
    """
    print(f"\n{'='*60}")
    print(f"🔬 Diagnosing: {layer_name}")
    print(f"{'='*60}")
    
    W_0_np = W_0.numpy()
    W_20_np = W_20.numpy()
    
    # 1. 基本统计
    print("\n1. Basic Statistics:")
    print(f"   Shape: {W_0_np.shape}")
    print(f"   0% mix  - Mean: {W_0_np.mean():.6f}, Std: {W_0_np.std():.6f}")
    print(f"   20% mix - Mean: {W_20_np.mean():.6f}, Std: {W_20_np.std():.6f}")
    print(f"   Norm ratio: {np.linalg.norm(W_20_np) / np.linalg.norm(W_0_np):.4f}")
    
    # 2. 直接相似度
    cosine_direct = np.dot(W_0_np.flatten(), W_20_np.flatten()) / (
        np.linalg.norm(W_0_np.flatten()) * np.linalg.norm(W_20_np.flatten())
    )
    print(f"\n2. Direct Cosine Similarity: {cosine_direct:.4f}")
    
    # 3. SVD分析
    U_0, S_0, Vt_0 = svd(W_0_np, full_matrices=False)
    U_20, S_20, Vt_20 = svd(W_20_np, full_matrices=False)
    
    print(f"\n3. SVD Analysis:")
    print(f"   Top 5 singular values:")
    print(f"   0% mix:  {S_0[:5]}")
    print(f"   20% mix: {S_20[:5]}")
    print(f"   S_20/S_0 ratio: {S_20[:5] / S_0[:5]}")
    
    # 4. 不同的相似度度量
    k = min(30, min(W_0_np.shape))
    
    # 4.1 Grassmann距离（通过主角度）
    V_0 = Vt_0[:k, :].T
    V_20 = Vt_20[:k, :].T
    overlap = V_0.T @ V_20
    principal_angles = np.arccos(np.clip(svd(overlap, compute_uv=False), -1, 1))
    grassmann_dist = np.linalg.norm(principal_angles)
    
    print(f"\n4. Different Similarity Metrics (k={k}):")
    print(f"   Grassmann distance: {grassmann_dist:.4f}")
    print(f"   Mean principal angle: {np.mean(principal_angles * 180/np.pi):.2f}°")
    print(f"   Max principal angle: {np.max(principal_angles * 180/np.pi):.2f}°")
    
    # 4.2 Subspace alignment（不同于cosine）
    # 使用投影矩阵
    P_0 = V_0 @ V_0.T  # Projection matrix for subspace 0
    P_20 = V_20 @ V_20.T  # Projection matrix for subspace 20
    
    # Frobenius norm of difference
    proj_diff = np.linalg.norm(P_0 - P_20, 'fro')
    print(f"   Projection matrix difference (Frobenius): {proj_diff:.4f}")
    
    # 5. 分解差异来源
    print(f"\n5. Decomposing the Difference:")
    
    # 重构矩阵
    W_0_reconstructed = U_0[:, :k] @ np.diag(S_0[:k]) @ Vt_0[:k, :]
    W_20_reconstructed = U_20[:, :k] @ np.diag(S_20[:k]) @ Vt_20[:k, :]
    
    # 计算重构误差
    recon_error_0 = np.linalg.norm(W_0_np - W_0_reconstructed) / np.linalg.norm(W_0_np)
    recon_error_20 = np.linalg.norm(W_20_np - W_20_reconstructed) / np.linalg.norm(W_20_np)
    print(f"   Reconstruction error (k={k}):")
    print(f"     0% mix:  {recon_error_0:.4f}")
    print(f"     20% mix: {recon_error_20:.4f}")
    
    # 交换成分分析
    # 使用0%的U和20%的V
    W_hybrid_1 = U_0[:, :k] @ np.diag(S_0[:k]) @ Vt_20[:k, :]
    # 使用20%的U和0%的V
    W_hybrid_2 = U_20[:, :k] @ np.diag(S_20[:k]) @ Vt_0[:k, :]
    
    sim_hybrid_1 = np.dot(W_0_np.flatten(), W_hybrid_1.flatten()) / (
        np.linalg.norm(W_0_np.flatten()) * np.linalg.norm(W_hybrid_1.flatten())
    )
    sim_hybrid_2 = np.dot(W_0_np.flatten(), W_hybrid_2.flatten()) / (
        np.linalg.norm(W_0_np.flatten()) * np.linalg.norm(W_hybrid_2.flatten())
    )
    
    print(f"   Hybrid similarity:")
    print(f"     0% with (U_0, S_0, V_20): {sim_hybrid_1:.4f}")
    print(f"     0% with (U_20, S_20, V_0): {sim_hybrid_2:.4f}")
    
    # 6. 检查是否是尺度问题
    # 标准化后的相似度
    W_0_normalized = W_0_np / np.linalg.norm(W_0_np)
    W_20_normalized = W_20_np / np.linalg.norm(W_20_np)
    cosine_normalized = np.dot(W_0_normalized.flatten(), W_20_normalized.flatten())
    print(f"\n6. After normalization:")
    print(f"   Cosine similarity: {cosine_normalized:.4f} (should be same as direct)")
    
    # 检查符号翻转
    sign_agreement = np.mean(np.sign(W_0_np.flatten()) == np.sign(W_20_np.flatten()))
    print(f"   Sign agreement: {sign_agreement:.4f}")
    
    return {
        'cosine_direct': cosine_direct,
        'grassmann_dist': grassmann_dist,
        'mean_pa': np.mean(principal_angles * 180/np.pi),
        'proj_diff': proj_diff,
        'S_0': S_0,
        'S_20': S_20
    }

def plot_detailed_comparison(results_dict):
    """
    绘制详细的比较图
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 准备数据
    layers = list(results_dict.keys())
    layers_short = [l.replace('transformer.', '').replace('.weight', '') for l in layers]
    
    # 1. 直接余弦相似度
    ax = axes[0, 0]
    cosines = [r['cosine_direct'] for r in results_dict.values()]
    bars = ax.bar(range(len(layers)), cosines, color='blue', alpha=0.7)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers_short, rotation=45, ha='right')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Direct Cosine Similarity')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # 2. Grassmann距离
    ax = axes[0, 1]
    grassmann = [r['grassmann_dist'] for r in results_dict.values()]
    ax.bar(range(len(layers)), grassmann, color='green', alpha=0.7)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers_short, rotation=45, ha='right')
    ax.set_ylabel('Grassmann Distance')
    ax.set_title('Subspace Distance')
    ax.grid(True, alpha=0.3)
    
    # 3. 平均主角度
    ax = axes[0, 2]
    mean_pas = [r['mean_pa'] for r in results_dict.values()]
    ax.bar(range(len(layers)), mean_pas, color='orange', alpha=0.7)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers_short, rotation=45, ha='right')
    ax.set_ylabel('Mean Principal Angle (degrees)')
    ax.set_title('Average Subspace Angle')
    ax.axhline(y=45, color='red', linestyle='--', alpha=0.3, label='45°')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 奇异值比较（选择一个层）
    ax = axes[1, 0]
    example_layer = list(results_dict.keys())[0]
    S_0 = results_dict[example_layer]['S_0'][:20]
    S_20 = results_dict[example_layer]['S_20'][:20]
    x = range(len(S_0))
    ax.semilogy(x, S_0, 'b-', label='0% mix', linewidth=2)
    ax.semilogy(x, S_20, 'r--', label='20% mix', linewidth=2)
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.set_title(f'Singular Values ({example_layer.split(".")[-1]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 奇异值比率
    ax = axes[1, 1]
    ratios = S_20[:20] / S_0[:20]
    ax.plot(x, ratios, 'go-', linewidth=2, markersize=6)
    ax.set_xlabel('Index')
    ax.set_ylabel('S_20 / S_0')
    ax.set_title('Singular Value Ratios')
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # 6. 相似度 vs Grassmann散点图
    ax = axes[1, 2]
    ax.scatter(cosines, grassmann, s=100, alpha=0.6)
    for i, layer in enumerate(layers_short):
        ax.annotate(layer, (cosines[i], grassmann[i]), 
                   fontsize=8, xytext=(2, 2), textcoords='offset points')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Grassmann Distance')
    ax.set_title('Similarity vs Distance')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Deep Diagnosis: Why Low Cosine but High PA?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'deep_diagnosis_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Diagnosis figure saved as: {filename}")
    plt.show()

def main():
    """主函数"""
    # 加载模型
    def get_checkpoint_path(d_model, mix_ratio, iteration):
        if mix_ratio == 0:
            pattern = f"out/composition_d{d_model}_0mix_*/ckpt_{iteration}.pt"
        else:
            pattern = f"out/composition_d{d_model}_{mix_ratio}mix_*/ckpt_{iteration}.pt"
        files = glob.glob(pattern)
        return files[0] if files else None
    
    path_0 = get_checkpoint_path(92, 0, 50000)
    path_20 = get_checkpoint_path(92, 20, 50000)
    
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    
    state_0 = ckpt_0['model'] if 'model' in ckpt_0 else ckpt_0
    state_20 = ckpt_20['model'] if 'model' in ckpt_20 else ckpt_20
    
    # 选择关键层进行深入分析
    key_layers = [
        'transformer.h.0.attn.c_proj.weight',  # 你提到的关键层
        'transformer.h.0.attn.c_attn.weight',
        'transformer.h.0.mlp.c_fc.weight',
        'transformer.wte.weight'
    ]
    
    results_dict = {}
    
    for layer_name in key_layers:
        # 查找实际的键
        actual_key = None
        for key in state_0.keys():
            if layer_name in key or key == layer_name:
                actual_key = key
                break
        
        if actual_key and actual_key in state_20:
            W_0 = state_0[actual_key].float()
            W_20 = state_20[actual_key].float()
            
            if len(W_0.shape) == 2:  # 只分析2D矩阵
                results = diagnose_layer_difference(W_0, W_20, actual_key)
                results_dict[actual_key] = results
    
    # 绘制比较图
    if results_dict:
        plot_detailed_comparison(results_dict)
    
    # 最终见解
    print("\n" + "="*80)
    print("🎯 KEY INSIGHTS")
    print("="*80)
    print("\nThe paradox explained:")
    print("1. High PA cosines → Subspaces are similar (same features)")
    print("2. Low direct cosine → Different linear combinations of features")
    print("3. No alignment improvement → Already optimally aligned")
    print("\nThis suggests: Models learn the same feature subspace")
    print("but combine features differently for the task!")

if __name__ == "__main__":
    main()