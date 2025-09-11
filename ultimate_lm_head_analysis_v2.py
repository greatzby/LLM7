#!/usr/bin/env python3
"""
ultimate_lm_head_analysis_v2.py

在v1的基础上，增加了对Procrustes对齐后结果的计算和可视化，
以验证“差异源于旋转”的假设。
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd, orthogonal_procrustes
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ============ 1. 全局配置 (请根据您的实验修改) ============
D_MODEL = 92
ITERATIONS = [5000, 10000, 20000, 30000, 40000, 50000]
K_ANALYSIS = 60
LAYER_TO_ANALYZE = 'lm_head'

# ============ 2. 数据加载与指标计算 (核心函数) ============

def get_checkpoint_path(d_model, mix_ratio, iteration):
    base_path = "out"
    dir_pattern = os.path.join(base_path, f"composition_d{d_model}_{mix_ratio}mix_*")
    dirs = glob.glob(dir_pattern)
    if not dirs: return None
    latest_dir = sorted(dirs)[-1]
    path = os.path.join(latest_dir, f'ckpt_{iteration}.pt')
    return path if os.path.exists(path) else None

def load_single_weight(d_model, mix_ratio, iteration, layer_key):
    path = get_checkpoint_path(d_model, mix_ratio, iteration)
    if not path:
        print(f"⚠️  Checkpoint not found: d={d_model}, mix={mix_ratio}, iter={iteration}")
        return None
    try:
        state_dict = torch.load(path, map_location='cpu').get('model', {})
        for key, w in state_dict.items():
            if layer_key in key:
                return w.float().numpy()
        print(f"⚠️  Layer '{layer_key}' not found in {path}")
        return None
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None

def compute_all_metrics(W1, W2, k): # <<< MODIFIED FUNCTION
    """计算两个矩阵之间的所有几何指标，包括对齐前和对齐后"""
    if W1 is None or W2 is None or W1.shape != W2.shape: return None
    k = min(k, W1.shape[0], W1.shape[1])
    
    metrics = {}
    U1, S1, Vt1 = svd(W1, full_matrices=False)
    U2, S2, Vt2 = svd(W2, full_matrices=False)
    metrics['S1'], metrics['S2'] = S1, S2

    V1_k, V2_k = Vt1[:k, :].T, Vt2[:k, :].T
    U1_k, U2_k = U1[:, :k], U2[:, :k]
    
    # --- 对齐前 ---
    metrics['V_cosines'] = np.clip(svd(V1_k.T @ V2_k, compute_uv=False), 0, 1)
    metrics['U_cosines'] = np.clip(svd(U1_k.T @ U2_k, compute_uv=False), 0, 1)

    # --- Procrustes 对齐 ---
    R_V, _ = orthogonal_procrustes(V2_k, V1_k)
    R_U, _ = orthogonal_procrustes(U2_k, U1_k)
    metrics['R_V'], metrics['R_U'] = R_V, R_U
    
    # --- 对齐后 (<<< NEW) ---
    V2_aligned = V2_k @ R_V
    U2_aligned = U2_k @ R_U
    metrics['V_cosines_aligned'] = np.clip(svd(V1_k.T @ V2_aligned, compute_uv=False), 0, 1)
    metrics['U_cosines_aligned'] = np.clip(svd(U1_k.T @ U2_aligned, compute_uv=False), 0, 1)
    
    return metrics

# ============ 3. 超级绘图函数 (为lm_head生成8合1报告) ============

def generate_ultimate_report(layer_name, all_results): # <<< MODIFIED FUNCTION
    """为lm_head生成包含8个子图的究极分析报告 (v2)"""
    if not all_results: return
    
    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    fig.suptitle(f'Ultimate Geometric Analysis (with Procrustes Alignment) for "{layer_name}"', fontsize=22, fontweight='bold')
    
    iters = sorted(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(iters)))
    final_iter = iters[-1]
    final_results = all_results[final_iter]

    # --- 图1 & 2: 保持不变 ---
    ax = axes[0, 0]
    for i, it in enumerate(iters): ax.plot(all_results[it]['V_cosines'], color=colors[i], alpha=0.7, label=f'{it//1000}k')
    ax.set_title('1. V-Space Principal Angle Cosines (Before)', fontweight='bold'); ax.set_xlabel('Principal Angle Index'); ax.set_ylabel('Cosine Value'); ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    for i, it in enumerate(iters): ax.plot(all_results[it]['U_cosines'], color=colors[i], alpha=0.7, label=f'{it//1000}k')
    ax.set_title('2. U-Space Principal Angle Cosines (Before)', fontweight='bold'); ax.set_xlabel('Principal Angle Index'); ax.legend(); ax.grid(True, alpha=0.3)

    # --- 图3 & 4: 保持不变 ---
    ax = axes[0, 2]
    ax.semilogy(final_results['S1'], 'o-', label='0% mix (final)', markersize=4, alpha=0.8); ax.semilogy(final_results['S2'], 's-', label='20% mix (final)', markersize=4, alpha=0.8)
    ax.set_title(f'3. Final SVD Spectrum (@{final_iter//1000}k)', fontweight='bold'); ax.set_xlabel('Singular Value Index'); ax.set_ylabel('Singular Value (log)'); ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[0, 3]
    ratio = final_results['S2'][:K_ANALYSIS] / final_results['S1'][:K_ANALYSIS]
    ax.plot(ratio, 'o-', markersize=5); ax.axhline(1.0, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'4. Final Singular Value Ratio (@{final_iter//1000}k)', fontweight='bold'); ax.set_xlabel('Singular Value Index'); ax.set_ylabel('Ratio (S_20 / S_0)'); ax.grid(True, alpha=0.3)

    # --- 图5: 平均主角度余弦演化 (<<< MODIFIED) ---
    ax = axes[1, 0]
    mean_v = [np.mean(all_results[it]['V_cosines']) for it in iters]
    mean_u = [np.mean(all_results[it]['U_cosines']) for it in iters]
    mean_v_aligned = [np.mean(all_results[it]['V_cosines_aligned']) for it in iters]
    mean_u_aligned = [np.mean(all_results[it]['U_cosines_aligned']) for it in iters]
    ax.plot(iters, mean_v, 'o-', c='blue', label='V-Space (Before)', markersize=8)
    ax.plot(iters, mean_v_aligned, 'o--', c='cyan', label='V-Space (Aligned)', markersize=8)
    ax.plot(iters, mean_u, 's-', c='red', label='U-Space (Before)', markersize=8)
    ax.plot(iters, mean_u_aligned, 's--', c='orange', label='U-Space (Aligned)', markersize=8)
    ax.set_title('5. Mean Subspace Alignment (Before vs. After)', fontweight='bold')
    ax.set_xlabel('Training Iteration'); ax.set_ylabel('Mean Cosine'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=min(ax.get_ylim()[0], 0.85), top=1.01)

    # --- 图6: 最终主角度分布对比 (<<< NEW) ---
    ax = axes[1, 1]
    angles_v_pre = np.arccos(final_results['V_cosines']) * 180 / np.pi
    angles_u_pre = np.arccos(final_results['U_cosines']) * 180 / np.pi
    angles_v_post = np.arccos(final_results['V_cosines_aligned']) * 180 / np.pi
    angles_u_post = np.arccos(final_results['U_cosines_aligned']) * 180 / np.pi
    bp = ax.boxplot([angles_v_pre, angles_v_post, angles_u_pre, angles_u_post], 
                    labels=['V (Pre)', 'V (Post)', 'U (Pre)', 'U (Post)'], patch_artist=True)
    colors = ['lightblue', 'cyan', 'lightcoral', 'orange']
    for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color)
    ax.set_title(f'6. Alignment Effect on Final Angles', fontweight='bold')
    ax.set_ylabel('Angle (degrees)'); ax.grid(True, axis='y', alpha=0.5)

    # --- 图7 & 8: 保持不变 ---
    ax = axes[1, 2]
    im = ax.imshow(final_results['R_V'], cmap='coolwarm', vmin=-1, vmax=1); ax.set_title('7. V-Space Rotation Matrix (R_V)', fontweight='bold'); ax.set_xlabel('Source Basis'); ax.set_ylabel('Target Basis'); fig.colorbar(im, ax=ax)
    ax = axes[1, 3]
    im = ax.imshow(final_results['R_U'], cmap='coolwarm', vmin=-1, vmax=1); ax.set_title('8. U-Space Rotation Matrix (R_U)', fontweight='bold'); ax.set_xlabel('Source Basis'); ax.set_ylabel('Target Basis'); fig.colorbar(im, ax=ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'ultimate_report_v2_{layer_name}_d{D_MODEL}_{timestamp}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"✅ Ultimate report v2 for '{layer_name}' saved as: {filename}")
    plt.close(fig)

# ============ 4. 主程序与汇总报告 ============

def main(): # <<< MODIFIED FUNCTION
    """主执行函数"""
    print("="*80); print(f"🚀 Starting Ultimate Analysis v2 for Layer: {LAYER_TO_ANALYZE.upper()}"); print("="*80)

    all_results = {}
    for it in ITERATIONS:
        W1 = load_single_weight(D_MODEL, 0, it, LAYER_TO_ANALYZE)
        W2 = load_single_weight(D_MODEL, 20, it, LAYER_TO_ANALYZE)
        if W1 is None or W2 is None:
            print(f"  Skipping iteration {it} due to missing data.")
            continue
        metrics = compute_all_metrics(W1, W2, K_ANALYSIS)
        if metrics:
            all_results[it] = metrics
            print(f"  Iter {it: <6}: V-cos={np.mean(metrics['V_cosines']):.4f} -> {np.mean(metrics['V_cosines_aligned']):.4f} (Aligned) | "
                  f"U-cos={np.mean(metrics['U_cosines']):.4f} -> {np.mean(metrics['U_cosines_aligned']):.4f} (Aligned)")

    if not all_results:
        print("❌ No data was processed. Analysis failed.")
        return
        
    generate_ultimate_report(LAYER_TO_ANALYZE, all_results)

    print("\n\n" + "="*80); print("📊 FINAL NUMERICAL REPORT (v2) for " + LAYER_TO_ANALYZE.upper() + " (@ " + str(max(ITERATIONS)) + " iterations)"); print("="*80)
    
    final_metrics = all_results.get(max(ITERATIONS))
    if final_metrics:
        angles_v = np.arccos(final_metrics['V_cosines']) * 180 / np.pi
        angles_u = np.arccos(final_metrics['U_cosines']) * 180 / np.pi
        
        print("--- Subspace Alignment (Before Alignment) ---")
        print(f"V-Space Mean Cosine: {np.mean(final_metrics['V_cosines']):.4f}")
        print(f"U-Space Mean Cosine: {np.mean(final_metrics['U_cosines']):.4f}")
        
        print("\n--- Subspace Alignment (After Procrustes Alignment) ---") # <<< NEW
        print(f"V-Space Mean Cosine (Aligned): {np.mean(final_metrics['V_cosines_aligned']):.4f}")
        print(f"U-Space Mean Cosine (Aligned): {np.mean(final_metrics['U_cosines_aligned']):.4f}")

        print("\n--- Principal Angle Distribution (degrees) ---")
        print(f"V-Space Angles (Before): mean={np.mean(angles_v):.2f}, std={np.std(angles_v):.2f}, min={np.min(angles_v):.2f}, max={np.max(angles_v):.2f}")
        print(f"U-Space Angles (Before): mean={np.mean(angles_u):.2f}, std={np.std(angles_u):.2f}, min={np.min(angles_u):.2f}, max={np.max(angles_u):.2f}")
        
        print("\n--- Singular Value Analysis ---")
        ratio = final_metrics['S2'] / final_metrics['S1']
        print(f"Singular Value Ratio (S_20/S_0) for top 5: {ratio[:5]}")
        
        print("\n--- Procrustes Rotation Analysis ---")
        R_V, R_U = final_metrics['R_V'], final_metrics['R_U']
        dist_v = np.linalg.norm(R_V - np.eye(R_V.shape[0]), 'fro')
        dist_u = np.linalg.norm(R_U - np.eye(R_U.shape[0]), 'fro')
        print(f"V-Space Rotation Distance (from Identity): {dist_v:.4f}")
        print(f"U-Space Rotation Distance (from Identity): {dist_u:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()