import os
import pickle
import numpy as np
import torch

data_dir = 'data/simple_graph/standardized_alpine_90_seed42'

# 1. 检查元数据
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

print("="*60)
print("META INFO:")
print(f"Vocab size: {meta['vocab_size']}")
print(f"Block size: {meta['block_size']}")
print(f"Vocab: {list(meta['stoi'].keys())[:20]}...")  # 显示前20个
print("="*60)

# 2. 检查训练数据
train_data = np.memmap(os.path.join(data_dir, 'train_0.bin'), dtype=np.uint16, mode='r')
print(f"\nTRAIN DATA INFO:")
print(f"Shape: {len(train_data)}")
print(f"Min value: {train_data.min()}")
print(f"Max value: {train_data.max()}")
print(f"Unique values: {len(np.unique(train_data[:10000]))}")

# 3. 检查是否有超出vocab的值
if train_data.max() >= meta['vocab_size']:
    print(f"⚠️ WARNING: Max value {train_data.max()} >= vocab_size {meta['vocab_size']}")
else:
    print(f"✅ All values within vocab range")

# 4. 检查序列结构
block_size = meta['block_size']
sequence_length = block_size + 1
num_sequences = len(train_data) // sequence_length

print(f"\nSEQUENCE INFO:")
print(f"Sequence length: {sequence_length}")
print(f"Number of sequences: {num_sequences}")
print(f"First sequence: {train_data[:sequence_length]}")

# 5. 测试批次生成
batch_size = 32
seq_indices = torch.randint(0, num_sequences, (batch_size,))
print(f"\nBATCH TEST:")
print(f"Random indices: {seq_indices[:5].tolist()}...")

try:
    ix = seq_indices * sequence_length
    x = torch.stack([torch.from_numpy(train_data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(train_data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    print(f"✅ Batch creation successful")
    print(f"X shape: {x.shape}, Y shape: {y.shape}")
    print(f"X max: {x.max().item()}, X min: {x.min().item()}")
except Exception as e:
    print(f"❌ Error in batch creation: {e}")