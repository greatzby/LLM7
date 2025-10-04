import os
import pickle
import numpy as np
import shutil

def prepare_single_alpha(alpha, track, source_dir='alpha_mixing_fixed', 
                        base_graph_dir='standardized_alpine_90_seed42'):
    """为单个alpha配置准备完整的训练目录"""
    
    # 创建目标目录
    target_dir = f'alpha_{track}_{alpha:.1f}'
    os.makedirs(target_dir, exist_ok=True)
    print(f"\nPreparing {target_dir}...")
    
    # 1. 复制txt文件并重命名
    src_train = os.path.join(source_dir, f'{track}_alpha_{alpha:.1f}.txt')
    dst_train = os.path.join(target_dir, 'train.txt')
    shutil.copy(src_train, dst_train)
    print(f"  Copied training data")
    
    # 2. 复制测试文件
    src_test = os.path.join(base_graph_dir, 'test.txt')
    dst_test = os.path.join(target_dir, 'test.txt')
    shutil.copy(src_test, dst_test)
    print(f"  Copied test data")
    
    # 3. 复制图和stage信息
    shutil.copy(
        os.path.join(base_graph_dir, 'composition_graph.graphml'),
        os.path.join(target_dir, 'composition_graph.graphml')
    )
    shutil.copy(
        os.path.join(base_graph_dir, 'stage_info.pkl'),
        os.path.join(target_dir, 'stage_info.pkl')
    )
    print(f"  Copied graph and stage info")
    
    # 4. 转换为bin格式
    print(f"  Converting to binary format...")
    
    # 读取数据
    with open(dst_train, 'r') as f:
        train_data = f.read()
    with open(dst_test, 'r') as f:
        val_data = f.read()  # 用test作为validation
    
    # 词汇表
    total_nodes = 90
    vocab_size = total_nodes + 2
    stoi = {str(i): i + 2 for i in range(total_nodes)}
    itos = {i + 2: str(i) for i in range(total_nodes)}
    stoi['[PAD]'] = 0
    itos[0] = '[PAD]'
    stoi['\n'] = 1
    itos[1] = '\n'
    
    def encode_string(s, stonum):
        ss = s.split(" ")
        return [stonum[ch] for ch in ss if ch in stonum]
    
    def process_reasoning(s, block_size):
        split_text = s.split('\n')
        ret = []
        for st in split_text:
            if st != "":
                enc_str = encode_string(st, stoi) + [1]
                ret += enc_str + [0] * (block_size + 1 - len(enc_str))
        return ret
    
    def get_block_size(s):
        split_text = s.split('\n')
        bs = 0
        for st in split_text:
            if st != "":
                enc_str = encode_string(st, stoi) + [1]
                bs = max(bs, len(enc_str))
        return bs
    
    # 计算block size
    block_size = (max(get_block_size(train_data), get_block_size(val_data)) // 32 + 1) * 32
    
    # 编码数据
    train_ids = np.array(process_reasoning(train_data, block_size), dtype=np.uint16)
    val_ids = np.array(process_reasoning(val_data, block_size), dtype=np.uint16)
    
    # 保存bin文件 - 注意文件名！
    train_bin_name = f'train_{int(alpha*100)}.bin'  # train_0.bin, train_10.bin, train_20.bin...
    train_ids.tofile(os.path.join(target_dir, train_bin_name))
    val_ids.tofile(os.path.join(target_dir, 'val.bin'))
    
    # 保存元信息
    meta = {
        'unreachable': False,
        'simple_format': True,
        'block_size': block_size,
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    
    with open(os.path.join(target_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"  ✅ Binary files created: {train_bin_name}, val.bin")
    print(f"  Block size: {block_size}")
    
    return target_dir, train_bin_name

def main():
    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 准备所有数据集
    all_configs = []
    
    print("="*60)
    print("Preparing Track A datasets...")
    print("="*60)
    
    for alpha in alpha_values:
        target_dir, train_file = prepare_single_alpha(alpha, 'track_A')
        all_configs.append({
            'alpha': alpha,
            'track': 'A',
            'data_dir': target_dir,
            'train_file': train_file
        })
    
    print("\n" + "="*60)
    print("Preparing Track B datasets...")
    print("="*60)
    
    for alpha in alpha_values:
        target_dir, train_file = prepare_single_alpha(alpha, 'track_B')
        all_configs.append({
            'alpha': alpha,
            'track': 'B',
            'data_dir': target_dir,
            'train_file': train_file
        })
    
    # 保存配置列表
    with open('alpha_training_configs.pkl', 'wb') as f:
        pickle.dump(all_configs, f)
    
    print("\n" + "="*60)
    print("✅ All datasets prepared!")
    print("="*60)
    print("\nTraining configurations saved to: alpha_training_configs.pkl")
    
    # 打印示例命令
    print("\n" + "="*60)
    print("Example training commands:")
    print("="*60)
    
    for i, config in enumerate(all_configs[:3]):  # 只显示前3个例子
        print(f"\n# Track {config['track']}, Alpha={config['alpha']:.1f}")
        print(f"python train_composition_final_fixed.py \\")
        print(f"    --data_dir=data/simple_graph/{config['data_dir']} \\")
        print(f"    --train_file={config['train_file']} \\")
        print(f"    --n_layer=3 --n_head=4 --n_embd=92 \\")
        print(f"    --max_iters=5000 --batch_size=32")

if __name__ == "__main__":
    main()