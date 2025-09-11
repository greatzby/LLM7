# verify_consistency.py
import pickle
import os

def verify_consistency(data_dir):
    """验证各个阶段使用相同的节点分组"""
    
    # 1. 从stage_info.pkl加载分组
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    
    S1, S2, S3 = stage_info['stages']
    S1_set = set(S1)
    S2_set = set(S2)
    S3_set = set(S3)
    
    print("="*60)
    print(f"Stage Info from stage_info.pkl:")
    print(f"S1 ({len(S1)}): {sorted(S1)[:10]}...")
    print(f"S2 ({len(S2)}): {sorted(S2)[:10]}...")
    print(f"S3 ({len(S3)}): {sorted(S3)[:10]}...")
    
    # 2. 验证测试文件
    print("\n" + "="*60)
    print("Verifying test.txt consistency:")
    
    with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
        test_lines = f.readlines()
    
    consistent = True
    for i, line in enumerate(test_lines[:10]):  # 检查前10行
        parts = line.strip().split()
        if len(parts) >= 2:
            source, target = int(parts[0]), int(parts[1])
            
            # 判断类型
            if source in S1_set and target in S2_set:
                path_type = "S1->S2"
            elif source in S2_set and target in S3_set:
                path_type = "S2->S3"
            elif source in S1_set and target in S3_set:
                path_type = "S1->S3"
            else:
                path_type = "UNKNOWN"
                consistent = False
            
            print(f"  Line {i}: {source}→{target} = {path_type}")
    
    if consistent:
        print("\n✅ All paths are correctly classified using stage_info.pkl")
    else:
        print("\n❌ Some paths cannot be classified - check data generation!")
    
    # 3. 检查meta.pkl中的词汇表
    print("\n" + "="*60)
    print("Checking vocabulary consistency:")
    
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi = meta['stoi']
    
    # 检查所有节点是否在词汇表中
    all_nodes = S1 + S2 + S3
    missing = []
    for node in all_nodes:
        if str(node) not in stoi:
            missing.append(node)
    
    if not missing:
        print(f"✅ All {len(all_nodes)} nodes are in vocabulary")
    else:
        print(f"❌ Missing nodes in vocabulary: {missing}")
    
    return consistent

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/simple_graph/standardized_alpine_90_seed42'
    verify_consistency(data_dir)