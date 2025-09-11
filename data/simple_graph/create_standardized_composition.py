# data/simple_graph/create_standardized_composition.py
import networkx as nx
import random
import os
import argparse
import numpy as np
import pickle

def generate_alpine_style_graph(num_nodes, edge_prob, DAG=True):
    """ALPINE风格的图生成"""
    G = nx.DiGraph()
    
    for i in range(num_nodes):
        G.add_node(i)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if DAG:
                if i < j and random.random() < edge_prob:
                    G.add_edge(i, j)
            else:
                if i != j and random.random() < edge_prob:
                    G.add_edge(i, j)
    
    return G

def divide_and_remove_s1_s3_edges(G0, nodes_per_stage=30):
    """分割节点并删除S1-S3边"""
    num_nodes = G0.number_of_nodes()
    node_list = list(G0.nodes())
    
    # 随机打乱后均分
    random.shuffle(node_list)
    
    size = num_nodes // 3
    S1 = set(node_list[:size])
    S2 = set(node_list[size:2*size])
    S3 = set(node_list[2*size:])
    
    # 删除S1-S3边
    G = G0.copy()
    edges_to_remove = [(u, v) for u, v in G.edges() 
                      if (u in S1 and v in S3) or (u in S3 and v in S1)]
    G.remove_edges_from(edges_to_remove)
    
    print(f"Removed {len(edges_to_remove)} edges between S1 and S3")
    
    # 转为字符串节点
    G_str = nx.DiGraph()
    for node in G.nodes():
        G_str.add_node(str(node))
    for u, v in G.edges():
        G_str.add_edge(str(u), str(v))
    
    stages = [sorted(list(S1)), sorted(list(S2)), sorted(list(S3))]
    
    return G_str, stages

class ALPINEPathGenerator:
    def __init__(self, graph):
        self.G = graph
        self.reachability = {}
        self._compute_reachability()
    
    def _compute_reachability(self):
        """预计算可达性"""
        print("Computing reachability...")
        TC = nx.transitive_closure(self.G)
        for node in self.G.nodes():
            self.reachability[node] = set(TC.predecessors(node))
    
    def random_walk(self, source, target):
        """随机游走生成路径"""
        source_str = str(source)
        target_str = str(target)
        
        if not nx.has_path(self.G, source_str, target_str):
            return None
        
        path = [source]
        current = source_str
        visited = set([current])
        max_steps = len(self.G.nodes()) * 2
        
        for _ in range(max_steps):
            if current == target_str:
                return path
            
            neighbors = list(self.G.successors(current))
            valid_next = [
                n for n in neighbors 
                if (n in self.reachability.get(target_str, set()) or n == target_str)
                and n not in visited
            ]
            
            if not valid_next:
                return self.random_walk(source, target)  # 重试
            
            next_node = random.choice(valid_next)
            path.append(int(next_node))
            visited.add(next_node)
            current = next_node
        
        return None

def generate_path_pools(G, stages, path_generator, paths_per_pair=20):
    """生成所有路径池"""
    S1, S2, S3 = stages
    P12, P23, P13 = [], [], []
    
    # S1->S2路径
    print("\nGenerating S1->S2 paths...")
    for s1 in S1:
        for s2 in S2:
            if nx.has_path(G, str(s1), str(s2)):
                for _ in range(paths_per_pair):
                    path = path_generator.random_walk(s1, s2)
                    if path:
                        P12.append([s1, s2] + path)
    
    # S2->S3路径
    print("Generating S2->S3 paths...")
    for s2 in S2:
        for s3 in S3:
            if nx.has_path(G, str(s2), str(s3)):
                for _ in range(paths_per_pair):
                    path = path_generator.random_walk(s2, s3)
                    if path:
                        P23.append([s2, s3] + path)
    
    # S1->S3路径
    print("Generating S1->S3 paths...")
    for s1 in S1:
        for s3 in S3:
            if nx.has_path(G, str(s1), str(s3)):
                for _ in range(paths_per_pair):
                    path = path_generator.random_walk(s1, s3)
                    if path and any(node in S2 for node in path[1:-1]):
                        P13.append([s1, s3] + path)
    
    print(f"\nPath pools: P12={len(P12)}, P23={len(P23)}, P13={len(P13)}")
    
    return {'P12': P12, 'P23': P23, 'P13': P13}

def construct_t_percent_mix(path_pools, t_percent, total_size):
    """构建t%-mix数据集"""
    t = t_percent / 100.0
    
    n_p12 = int((1 - t) * total_size / 2)
    n_p23 = int((1 - t) * total_size / 2)
    n_p13 = int(t * total_size)
    
    print(f"\n{t_percent}%-mix: P12={n_p12}, P23={n_p23}, P13={n_p13}")
    
    dataset = []
    
    # 采样
    if len(path_pools['P12']) >= n_p12:
        dataset.extend(random.sample(path_pools['P12'], n_p12))
    else:
        dataset.extend(random.choices(path_pools['P12'], k=n_p12))
    
    if len(path_pools['P23']) >= n_p23:
        dataset.extend(random.sample(path_pools['P23'], n_p23))
    else:
        dataset.extend(random.choices(path_pools['P23'], k=n_p23))
    
    if n_p13 > 0 and path_pools['P13']:
        if len(path_pools['P13']) >= n_p13:
            dataset.extend(random.sample(path_pools['P13'], n_p13))
        else:
            dataset.extend(random.choices(path_pools['P13'], k=n_p13))
    
    random.shuffle(dataset)
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=90)
    parser.add_argument('--edge_prob', type=float, default=0.1)
    parser.add_argument('--paths_per_pair', type=int, default=20)
    parser.add_argument('--dataset_size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*60)
    print("Standardized Data Generation (ALPINE Method)")
    print("="*60)
    
    # 生成图
    print("\nStep 1: Building ALPINE-style graph...")
    G0 = generate_alpine_style_graph(args.num_nodes, args.edge_prob)
    
    print("\nStep 2: Dividing nodes and removing S1-S3 edges...")
    G, stages = divide_and_remove_s1_s3_edges(G0)
    
    print("\nStep 3: Initializing path generator...")
    path_generator = ALPINEPathGenerator(G)
    
    print("\nStep 4: Generating path pools...")
    path_pools = generate_path_pools(G, stages, path_generator, args.paths_per_pair)
    
    # 创建输出目录
    output_dir = f'standardized_alpine_{args.num_nodes}_seed{args.seed}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成0%-mix
    dataset_0 = construct_t_percent_mix(path_pools, 0, args.dataset_size)
    with open(os.path.join(output_dir, 'train_0.txt'), 'w') as f:
        for path_data in dataset_0:
            f.write(' '.join(map(str, path_data)) + '\n')
    
    # 生成20%-mix
    dataset_20 = construct_t_percent_mix(path_pools, 20, args.dataset_size)
    with open(os.path.join(output_dir, 'train_20.txt'), 'w') as f:
        for path_data in dataset_20:
            f.write(' '.join(map(str, path_data)) + '\n')
    
    # 测试集
    test_dataset = []
    test_dataset.extend(random.sample(path_pools['P12'], min(50, len(path_pools['P12']))))
    test_dataset.extend(random.sample(path_pools['P23'], min(50, len(path_pools['P23']))))
    test_dataset.extend(random.sample(path_pools['P13'], min(50, len(path_pools['P13']))))
    random.shuffle(test_dataset)
    
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        for path_data in test_dataset:
            f.write(' '.join(map(str, path_data)) + '\n')
    
    # 保存图和元信息
    nx.write_graphml(G, os.path.join(output_dir, 'composition_graph.graphml'))
    
    stage_info = {
        'stages': stages,
        'nodes_per_stage': args.num_nodes // 3
    }
    with open(os.path.join(output_dir, 'stage_info.pkl'), 'wb') as f:
        pickle.dump(stage_info, f)
    
    print(f"\n✅ Data saved to: {output_dir}/")

if __name__ == "__main__":
    main()