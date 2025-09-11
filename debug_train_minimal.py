import torch
import numpy as np
import pickle
import os
from model import GPTConfig, GPT

# 设置调试环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

data_dir = 'data/simple_graph/standardized_alpine_90_seed42'

# 加载数据
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
block_size = meta['block_size']

train_data = np.memmap(os.path.join(data_dir, 'train_0.bin'), dtype=np.uint16, mode='r')

print(f"Data loaded: {len(train_data)} tokens")
print(f"Vocab size: {vocab_size}, Block size: {block_size}")

# 创建模型 - 使用实际的参数
model_args = dict(
    n_layer=1, 
    n_head=1, 
    n_embd=120,  # 使用实际的120
    block_size=block_size, 
    bias=False, 
    vocab_size=vocab_size, 
    dropout=0.0
)

gptconf = GPTConfig(**model_args)
model = GPT(gptconf).to('cuda')
print(f"Model created with n_embd=120")

# 创建优化器
optimizer = model.configure_optimizers(
    weight_decay=1e-1, 
    learning_rate=5e-4, 
    betas=(0.9, 0.95), 
    device_type='cuda'
)

# 测试不同的batch size
for batch_size in [32, 256, 512, 1024]:
    print(f"\n{'='*40}")
    print(f"Testing batch_size={batch_size}")
    
    sequence_length = block_size + 1
    num_sequences = len(train_data) // sequence_length
    
    try:
        # 创建批次
        seq_indices = torch.randint(0, num_sequences, (batch_size,))
        ix = seq_indices * sequence_length
        
        x = torch.stack([torch.from_numpy(train_data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(train_data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        
        x = x.to('cuda').contiguous()
        y = y.to('cuda').contiguous()
        
        print(f"Batch created: X={x.shape}, Y={y.shape}")
        
        # 前向传播
        logits, loss = model(x, y)
        print(f"Forward pass: loss={loss.item():.4f}")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        print(f"Backward pass successful")
        
        # 优化器步骤
        optimizer.step()
        print(f"✅ batch_size={batch_size} successful")
        
    except Exception as e:
        print(f"❌ batch_size={batch_size} failed: {e}")
        break