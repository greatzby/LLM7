# data/simple_graph/prepare_composition_multi.py
import os
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
parser.add_argument('--train_file', type=str, required=True, help='Training file name (e.g., train_0.txt)')
parser.add_argument('--total_nodes', type=int, default=90)
args = parser.parse_args()

# 文件路径
base_dir = args.data_dir
train_file_path = os.path.join(base_dir, args.train_file)
val_file_path = os.path.join(base_dir, 'test.txt')

# 读取数据
with open(train_file_path, 'r') as f:
    train_data = f.read()

with open(val_file_path, 'r') as f:
    val_data = f.read()

# 词汇表
vocab_size = args.total_nodes + 2
stoi = {str(i): i + 2 for i in range(args.total_nodes)}
itos = {i + 2: str(i) for i in range(args.total_nodes)}
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
print(f"Block size: {block_size}")

# 编码数据
train_ids = np.array(process_reasoning(train_data, block_size), dtype=np.uint16)
val_ids = np.array(process_reasoning(val_data, block_size), dtype=np.uint16)

# 保存
output_name = args.train_file.replace('.txt', '.bin')
train_ids.tofile(os.path.join(base_dir, output_name))
val_ids.tofile(os.path.join(base_dir, 'val.bin'))

# 保存元信息
meta = {
    'unreachable': False,
    'simple_format': True,
    'block_size': block_size,
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(base_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"✅ Processed {args.train_file}")