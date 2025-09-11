# verify_dataset.py
import pickle
import os
import sys

def verify_dataset(data_dir):
    """验证数据集的正确性"""
    
    # 加载阶段信息
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    
    S1, S2, S3 = stage_info['stages']
    S1 = set(S1)
    S2 = set(S2)
    S3 = set(S3)
    
    print(f"Stage sizes: S1={len(S1)}, S2={len(S2)}, S3={len(S3)}")
    print(f"S1 nodes: {sorted(list(S1))[:10]}...")
    print(f"S2 nodes: {sorted(list(S2))[:10]}...")
    print(f"S3 nodes: {sorted(list(S3))[:10]}...")
    print("\n" + "="*60)
    
    # 检查各个训练文件
    for filename in ['train_0.txt', 'train_20.txt']:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        print(f"\nChecking {filename}:")
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        counts = {'S1->S2': 0, 'S2->S3': 0, 'S1->S3': 0, 'Other': 0}
        s1_s3_examples = []
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 4:
                continue
                
            source = int(parts[0])
            target = int(parts[1])
            
            # 分类路径
            if source in S1 and target in S2:
                counts['S1->S2'] += 1
            elif source in S2 and target in S3:
                counts['S2->S3'] += 1
            elif source in S1 and target in S3:
                counts['S1->S3'] += 1
                if len(s1_s3_examples) < 5:
                    s1_s3_examples.append((i, line.strip()))
            else:
                counts['Other'] += 1
        
        # 打印统计
        total = sum(counts.values())
        print(f"  Total paths: {total}")
        for path_type, count in counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {path_type}: {count} ({percentage:.1f}%)")
        
        # 如果有S1->S3路径，打印例子
        if s1_s3_examples:
            print(f"\n  ⚠️ WARNING: Found S1->S3 paths in {filename}!")
            print("  Examples:")
            for line_num, example in s1_s3_examples:
                print(f"    Line {line_num}: {example}")
                parts = example.split()
                source, target = int(parts[0]), int(parts[1])
                print(f"      Source {source} in S1: {source in S1}")
                print(f"      Target {target} in S3: {target in S3}")

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/simple_graph/standardized_alpine_90_seed42'
    verify_dataset(data_dir)