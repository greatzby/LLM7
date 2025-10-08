#!/usr/bin/env python3
"""
prepare_all_data.py
批量准备所有图的二进制数据 - 兼容您的训练脚本
"""

import os
import subprocess
import argparse

def prepare_graph_data(graph_dir, graph_name):
    """为单个图准备数据"""
    print(f"  Preparing data for: {graph_name}")
    
    # 使用您提供的prepare_composition_multi.py
    cmd = [
        'python', 'data/simple_graph/prepare_composition_multi.py',
        '--data_dir', graph_dir,
        '--train_file', 'train_alpine.txt',  # 输入的txt文件
        '--total_nodes', '90'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"    ✓ Success - generated train_alpine.bin and val.bin")
    else:
        print(f"    ✗ Failed: {result.stderr}")
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='data/graph_repair_experiment')
    args = parser.parse_args()
    
    graph_names = ['G_base', 'G_A_low', 'G_A_high', 'G_B_low', 'G_B_high', 'G_C_low', 'G_C_high']
    
    print("\n" + "="*60)
    print("PREPARING BINARY DATA FOR ALL GRAPHS")
    print("="*60)
    
    success_count = 0
    for name in graph_names:
        graph_dir = os.path.join(args.base_dir, name)
        
        if not os.path.exists(graph_dir):
            print(f"⚠️ Skipping {name} - not found")
            continue
        
        print(f"\n{name}:")
        if prepare_graph_data(graph_dir, name):
            success_count += 1
    
    print(f"\n✅ Prepared {success_count}/{len(graph_names)} graphs")
    print("\n📌 Generated files in each graph directory:")
    print("  - train_alpine.bin (training data)")
    print("  - val.bin (validation data)")
    print("  - meta.pkl (metadata)")

if __name__ == "__main__":
    main()