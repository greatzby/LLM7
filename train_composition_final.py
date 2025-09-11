# train_composition_final.py (最终修复版)
import os
import pickle
import argparse
import numpy as np
import torch
import networkx as nx
import random
from datetime import datetime
from collections import defaultdict

from model import GPTConfig, GPT
from logger import get_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train a GPT model for composition task")
    
    # --- 核心参数 ---
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing the processed .bin files and meta.pkl')
    parser.add_argument('--train_file', type=str, required=True, 
                        help="Name of the training .bin file (e.g., train_0.bin or train_20.bin)")
    
    # --- 模型架构参数 ---
    parser.add_argument('--n_layer', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=120, help='Embedding dimension (d)')
    
    # --- 训练过程参数 ---
    parser.add_argument('--max_iters', type=int, default=50000, help='Total number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    
    # --- 评估与保存参数 ---
    parser.add_argument('--eval_interval', type=int, default=1000, help='Interval for evaluation and logging')
    parser.add_argument('--checkpoint_interval', type=int, default=5000, help='Interval for saving model checkpoints')
    
    # --- 环境参数 ---
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., cuda:0, cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

@torch.no_grad()
def evaluate_composition(model, test_file, stages, stoi, itos, device, G, 
                        vocab_size, temperature=0.1, top_k=10):
    """在测试集上评估模型的组合推理能力"""
    model.eval()
    
    S1, S2, S3 = [set(s) for s in stages]
    
    try:
        with open(test_file, 'r') as f:
            test_lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Test file not found at {test_file}")
        return {}

    test_by_type = defaultdict(list)
    for line in test_lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                source, target = int(parts[0]), int(parts[1])
                true_path = [int(p) for p in parts[2:]]
                
                if source in S1 and target in S2:
                    test_by_type['S1->S2'].append((source, target, true_path))
                elif source in S2 and target in S3:
                    test_by_type['S2->S3'].append((source, target, true_path))
                elif source in S1 and target in S3:
                    test_by_type['S1->S3'].append((source, target, true_path))
            except (ValueError, IndexError):
                print(f"Warning: Could not parse line: {line}")
                continue
    
    results = {}
    
    for path_type, test_cases in test_by_type.items():
        if not test_cases:
            continue
            
        correct_count = 0
        
        for source, target, _ in test_cases:
            prompt_str = f"{source} {target} {source}"
            prompt_tokens = [stoi[token] for token in prompt_str.split() if token in stoi]
            x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            y = model.generate(x, max_new_tokens=30, temperature=temperature, top_k=top_k)
            
            generated_ids = y[0].tolist()
            eos_token_id = stoi.get('\n', 1)
            if eos_token_id in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(eos_token_id)]
            
            generated_path = [int(itos[tid]) for tid in generated_ids[len(prompt_tokens):] 
                            if tid in itos and itos[tid].isdigit()]
            
            is_valid = False
            if len(generated_path) >= 2 and generated_path[0] == source and generated_path[-1] == target:
                path_is_connected = all(G.has_edge(str(u), str(v)) 
                                       for u, v in zip(generated_path[:-1], generated_path[1:]))
                if path_is_connected:
                    if path_type == 'S1->S3':
                        if any(node in S2 for node in generated_path[1:-1]):
                            is_valid = True
                    else:
                        is_valid = True
            
            if is_valid:
                correct_count += 1
        
        results[path_type] = {
            'correct': correct_count, 
            'total': len(test_cases),
            'accuracy': correct_count / len(test_cases) if test_cases else 0
        }
    
    model.train()
    return results

def main():
    args = parse_args()
    
    # --- 1. 设置环境和日志 ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    mix_type = "0mix" if "train_0" in args.train_file else "20mix"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f'out/composition_d{args.n_embd}_{mix_type}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    logger = get_logger(os.path.join(out_dir, "train.log"))
    logger.info("="*60)
    logger.info(f"Starting Composition Training")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Output directory: {out_dir}")
    logger.info("="*60)

    # --- 2. 加载数据和元信息 ---
    data_dir = args.data_dir
    
    try:
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        block_size, vocab_size = meta['block_size'], meta['vocab_size']
        
        with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        stages = stage_info['stages']

        graph_path = os.path.join(data_dir, 'composition_graph.graphml')
        G = nx.read_graphml(graph_path)

        train_data = np.memmap(os.path.join(data_dir, args.train_file), dtype=np.uint16, mode='r')
        val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        test_file = os.path.join(data_dir, 'test.txt')
    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        return

    logger.info(f"Vocab size: {vocab_size}, Block size: {block_size}")

    # --- 3. 初始化模型和优化器 ---
    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      block_size=block_size, bias=False, vocab_size=vocab_size, dropout=0.0)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(args.device)
    
    logger.info(f"Model initialized: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=args.learning_rate, 
                                           betas=(0.9, 0.95), device_type=args.device.split(':')[0])

    # ####################################################################
    # ##               ↓↓↓ 唯一的修改在这里 ↓↓↓                     ##
    # ####################################################################
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        # 每个序列在文件中占用的长度是 block_size + 1
        sequence_length_in_file = block_size + 1
        
        # 计算文件中有多少个完整的序列
        num_sequences = len(data) // sequence_length_in_file
        
        # 随机选择 batch_size 个序列的起始索引
        # torch.randint 在 [low, high) 区间内生成整数
        seq_indices = torch.randint(0, num_sequences, (args.batch_size,))
        
        # 将序列索引转换为在扁平化数据数组中的实际起始位置
        ix = seq_indices * sequence_length_in_file

        # 使用列表推导式高效地构建批次
        # 这是创建批次最高效、最直接的方式
        x = torch.stack([torch.from_numpy(data[i : i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1 : i+1+block_size].astype(np.int64)) for i in ix])
        
        # 将数据移动到指定设备，并确保内存是连续的
        # .contiguous() 是解决这类CUDA错误的关键！
        return x.to(args.device).contiguous(), y.to(args.device).contiguous()
    # ####################################################################
    # ##               ↑↑↑ 唯一的修改在这里 ↑↑↑                     ##
    # ####################################################################

    # --- 5. 训练循环 ---
    logger.info("\nStarting training...")
    running_loss, loss_count = 0.0, 0
    
    for iter_num in range(args.max_iters + 1):
        if iter_num < 2000:
            lr = args.learning_rate * iter_num / 2000
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        if iter_num % args.eval_interval == 0:
            model.eval()
            
            avg_train_loss = running_loss / loss_count if loss_count > 0 else float('nan')
            val_losses = []
            for _ in range(10):
                X_val, Y_val = get_batch('val')
                with torch.no_grad():
                    _, loss = model(X_val, Y_val)
                    val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses)
            
            results = evaluate_composition(model, test_file, stages, stoi, itos, args.device, G, vocab_size)
            
            log_msg = f"iter:{iter_num} | loss(train):{avg_train_loss:.4f} | loss(val):{avg_val_loss:.4f}"
            for path_type, res in results.items():
                log_msg += f" | acc({path_type}):{res['accuracy']:.2%}"
            logger.info(log_msg)
            
            running_loss, loss_count = 0.0, 0
            model.train()
        
        if iter_num > 0 and iter_num % args.checkpoint_interval == 0:
            checkpoint = {'model': model.state_dict(), 'model_args': model_args,
                          'iter_num': iter_num, 'config': vars(args)}
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
            logger.info(f"Checkpoint saved at iteration {iter_num}")
        
        if iter_num == args.max_iters:
            break

        X, Y = get_batch('train')
        logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        loss_count += 1
    
    logger.info(f"\nTraining finished! Results are in: {out_dir}")

if __name__ == "__main__":
    main()