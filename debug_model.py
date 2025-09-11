import torch
import numpy as np
import pickle
import os
from model import GPTConfig, GPT

# 加载数据
data_dir = 'data/simple_graph/standardized_alpine_90_seed42'
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
block_size = meta['block_size']

print(f"Vocab size: {vocab_size}, Block size: {block_size}")

# 创建小模型在CPU上测试
model_args = dict(
    n_layer=1, 
    n_head=1, 
    n_embd=64,  # 用小一点的
    block_size=block_size, 
    bias=False, 
    vocab_size=vocab_size, 
    dropout=0.0
)

gptconf = GPTConfig(**model_args)
model = GPT(gptconf).to('cpu')  # 先在CPU上测试
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# 创建小批次测试
batch_size = 4
x = torch.randint(0, vocab_size, (batch_size, block_size))
y = torch.randint(0, vocab_size, (batch_size, block_size))

print(f"\nTest batch shapes: X={x.shape}, Y={y.shape}")

# 测试前向传播
try:
    logits, loss = model(x, y)
    print(f"✅ CPU Forward pass successful")
    print(f"Loss: {loss.item():.4f}")
    
    # 测试反向传播
    loss.backward()
    print(f"✅ CPU Backward pass successful")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# 如果CPU成功，尝试GPU
if torch.cuda.is_available():
    print("\n" + "="*60)
    print("Testing on GPU...")
    
    model = model.to('cuda')
    x = x.to('cuda')
    y = y.to('cuda')
    
    try:
        logits, loss = model(x, y)
        print(f"✅ GPU Forward pass successful")
        print(f"Loss: {loss.item():.4f}")
        
        loss.backward()
        print(f"✅ GPU Backward pass successful")
    except Exception as e:
        print(f"❌ GPU Error: {e}")
        import traceback
        traceback.print_exc()