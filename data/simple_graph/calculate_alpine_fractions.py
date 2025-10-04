import networkx as nx
import pickle
import os
import sys

def calculate_reachable_pairs(G, stages):
    """计算各类型可达对的数量"""
    S1, S2, S3 = stages
    
    # 转换为字符串格式（与您的图一致）
    S1_str = set(str(s) for s in S1)
    S2_str = set(str(s) for s in S2)
    S3_str = set(str(s) for s in S3)
    
    # 计算可达对数量
    N_12 = 0  # S1->S2
    N_23 = 0  # S2->S3
    N_13 = 0  # S1->S3
    
    # 计算S1->S2可达对
    print("Counting S1->S2 reachable pairs...")
    for s1 in S1:
        for s2 in S2:
            if nx.has_path(G, str(s1), str(s2)):
                N_12 += 1
    
    # 计算S2->S3可达对
    print("Counting S2->S3 reachable pairs...")
    for s2 in S2:
        for s3 in S3:
            if nx.has_path(G, str(s2), str(s3)):
                N_23 += 1
    
    # 计算S1->S3可达对
    print("Counting S1->S3 reachable pairs...")
    for s1 in S1:
        for s3 in S3:
            if nx.has_path(G, str(s1), str(s3)):
                N_13 += 1
    
    return N_12, N_23, N_13

def main():
    # 设置数据目录
    data_dir = 'standardized_alpine_90_seed42'
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found!")
        print("Please run create_standardized_composition.py first.")
        return
    
    # 读取图
    graph_file = os.path.join(data_dir, 'composition_graph.graphml')
    print(f"Loading graph from {graph_file}...")
    G = nx.read_graphml(graph_file)
    
    # 读取节点分组信息
    stage_info_file = os.path.join(data_dir, 'stage_info.pkl')
    print(f"Loading stage info from {stage_info_file}...")
    with open(stage_info_file, 'rb') as f:
        stage_info = pickle.load(f)
    
    stages = stage_info['stages']
    S1, S2, S3 = stages
    
    print(f"\nGraph statistics:")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    print(f"S1 nodes: {len(S1)} nodes")
    print(f"S2 nodes: {len(S2)} nodes")
    print(f"S3 nodes: {len(S3)} nodes")
    
    # 计算可达对
    print("\n" + "="*50)
    print("Calculating reachable pairs...")
    print("="*50)
    
    N_12, N_23, N_13 = calculate_reachable_pairs(G, stages)
    
    # 计算总数和比例
    N_total = N_12 + N_23 + N_13
    
    if N_total == 0:
        print("Error: No reachable pairs found!")
        return
    
    fraction_12 = N_12 / N_total
    fraction_23 = N_23 / N_total
    fraction_13 = N_13 / N_total
    
    # 打印结果
    print("\n" + "="*50)
    print("RESULTS: Natural Fractions in Pure ALPINE Mode")
    print("="*50)
    
    print(f"\nReachable pair counts:")
    print(f"  N_12 (S1→S2): {N_12}")
    print(f"  N_23 (S2→S3): {N_23}")
    print(f"  N_13 (S1→S3): {N_13}")
    print(f"  N_total: {N_total}")
    
    print(f"\nNatural fractions:")
    print(f"  S1→S2 paths: {fraction_12:.1%}")
    print(f"  S2→S3 paths: {fraction_23:.1%}")
    print(f"  S1→S3 paths: {fraction_13:.1%}")
    
    # 验证总和
    print(f"\nVerification: Sum = {fraction_12 + fraction_23 + fraction_13:.4f}")
    
    # 保存结果
    results = {
        'N_12': N_12,
        'N_23': N_23,
        'N_13': N_13,
        'N_total': N_total,
        'fraction_12': fraction_12,
        'fraction_23': fraction_23,
        'fraction_13': fraction_13
    }
    
    results_file = os.path.join(data_dir, 'alpine_fractions.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_file}")
    
    # 生成简短的回复给教授
    print("\n" + "="*50)
    print("RESPONSE FOR PROFESSOR:")
    print("="*50)
    print(f"The fractions in pure ALPINE mode (based on reachable pairs):")
    print(f"- S1→S2 paths: {fraction_12:.1%}")
    print(f"- S2→S3 paths: {fraction_23:.1%}")
    print(f"- S1→S3 paths: {fraction_13:.1%}")
    print(f"\nThese are determined by the graph topology with N_12={N_12}, N_23={N_23}, N_13={N_13} unique reachable pairs.")

if __name__ == "__main__":
    main()