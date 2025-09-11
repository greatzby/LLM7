# train_composition_final_fixed.py
import os
import pickle
import argparse
import numpy as np
import torch
import networkx as nx
from datetime import datetime
from collections import defaultdict
import random
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
                        vocab_size, temperature=0.1, top_k=10, debug=False):
    """
    在测试集上评估模型的组合推理能力（根据您的逻辑修复）
    """
    model.eval()
    
    # 确保阶段信息是整数集合
    S1, S2, S3 = [set(s) for s in stages]
    
    # 读取测试数据
    try:
        with open(test_file, 'r') as f:
            test_lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Test file not found at {test_file}")
        return {}

    # 按路径类型分类
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
                continue
    
    results = {}
    
    for path_type, test_cases in test_by_type.items():
        if not test_cases:
            continue
            
        correct_count = 0
        
        for idx, (source, target, _) in enumerate(test_cases):
            # 构造输入 prompt: "source target source"
            prompt_str = f"{source} {target} {source}"
            prompt_tokens = [stoi[token] for token in prompt_str.split() if token in stoi]
            
            if not prompt_tokens:
                if debug and idx < 3:
                    print(f"  WARNING: No valid tokens for {source}→{target}")
                continue
                
            x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            # 模型生成
            y = model.generate(x, max_new_tokens=30, temperature=temperature, top_k=top_k)
            
            # ========== 关键修复：正确提取路径 (您的逻辑) ==========
            all_numbers = []
            generated_ids = y[0].tolist()
            
            for tid in generated_ids:
                if tid == stoi.get('\n', 1):  # EOS token
                    break
                if tid in itos:
                    token_str = itos[tid]
                    if token_str.isdigit():
                        try:
                            all_numbers.append(int(token_str))
                        except ValueError:
                            pass
            
            # 从第3个数字开始作为路径（跳过prompt的"source target"，保留最后的source）
            if len(all_numbers) >= 3:
                generated_path = all_numbers[2:]
            else:
                generated_path = []
            
            # 调试输出
            if debug and idx < 3:
                print(f"  [{path_type}] {source}→{target}:")
                print(f"    Prompt: {prompt_str} | All numbers generated: {all_numbers}")
                print(f"    Extracted path: {generated_path}")
            
            # 验证路径
            is_valid = False
            if len(generated_path) >= 2:
                if generated_path[0] == source and generated_path[-1] == target:
                    path_is_connected = all(
                        G.has_edge(str(u), str(v)) 
                        for u, v in zip(generated_path[:-1], generated_path[1:])
                    )
                    
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
    
    # ========== 1. 设置环境 ==========
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    mix_type = "0mix" if "train_0" in args.train_file else ("10mix" if "train_10" in args.train_file else "20mix")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f'out/composition_d{args.n_embd}_{mix_type}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    logger = get_logger(os.path.join(out_dir, "train.log"))
    
    # ========== 终端和日志都打印 (您的改进) ==========
    header_msgs = [
        "="*60,
        f"Composition Training - FINAL FIXED VERSION",
        f"Model: {args.n_layer}L-{args.n_head}H-{args.n_embd}D",
        f"Data: {args.data_dir}",
        f"Train file: {args.train_file}",
        f"Output dir: {out_dir}",
        "="*60
    ]
    for msg in header_msgs:
        print(msg)
        logger.info(msg)

    # ========== 2. 加载数据 ==========
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
        error_msg = f"Data loading error: {e}"
        print(f"ERROR: {error_msg}")
        logger.error(error_msg)
        return

    info_msg = f"Vocab size: {vocab_size}, Block size: {block_size}"
    print(info_msg)
    logger.info(info_msg)

    # ========== 3. 初始化模型 ==========
    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      block_size=block_size, bias=False, vocab_size=vocab_size, dropout=0.0)
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    info_msg = f"Model initialized: {num_params:.2f}M parameters"
    print(info_msg)
    logger.info(info_msg)
    
    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=args.learning_rate, 
                                           betas=(0.9, 0.95), device_type=args.device.split(':')[0])

    # ========== 4. 数据加载函数 ==========
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        data_size = block_size + 1
        
        num_sequences = len(data) // data_size
        if num_sequences < args.batch_size:
            raise ValueError(f"Not enough data for a full batch. Num sequences: {num_sequences}, batch_size: {args.batch_size}")

        seq_indices = torch.randint(0, num_sequences, (args.batch_size,))
        ix = seq_indices * data_size
        
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        return x.to(args.device), y.to(args.device)

    # ========== 5. 训练循环 ==========
    print("\nStarting training...")
    logger.info("Starting training...")
    
    running_loss, loss_count = 0.0, 0
    
    for iter_num in range(args.max_iters + 1):
        if iter_num < 2000 and iter_num > 0: # iter 0 is for eval only
            lr = args.learning_rate * iter_num / 2000
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # ========== 评估 ==========
        if iter_num % args.eval_interval == 0:
            model.eval()
            
            avg_train_loss = running_loss / loss_count if loss_count > 0 else float('nan')
            
            val_losses = []
            with torch.no_grad():
                for _ in range(10):
                    X_val, Y_val = get_batch('val')
                    _, loss = model(X_val, Y_val)
                    val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses)
            
            debug = (iter_num == 0)
            results = evaluate_composition(model, test_file, stages, stoi, itos, 
                                           args.device, G, vocab_size, debug=debug)
            
            msg = f"Iter {iter_num:5d} | Loss: train={avg_train_loss:.4f}, val={avg_val_loss:.4f} | "
            acc_parts = []
            for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                if path_type in results:
                    res = results[path_type]
                    acc_parts.append(f"{path_type}={res['accuracy']:.1%}")
            msg += "Acc: " + ", ".join(acc_parts)
            
            print(msg)
            logger.info(msg)
            
            running_loss, loss_count = 0.0, 0
            model.train()
        
        # ========== 保存检查点 ==========
        if iter_num > 0 and iter_num % args.checkpoint_interval == 0:
            checkpoint = {'model': model.state_dict(), 'model_args': model_args,
                          'iter_num': iter_num, 'config': vars(args)}
            save_path = os.path.join(out_dir, f'ckpt_{iter_num}.pt')
            torch.save(checkpoint, save_path)
            
            save_msg = f"Checkpoint saved to {save_path}"
            print(save_msg)
            logger.info(save_msg)
        
        if iter_num == args.max_iters:
            break

        # ========== 训练步 ==========
        X, Y = get_batch('train')
        logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        loss_count += 1
    
    finish_msg = f"\nTraining finished! Results are in: {out_dir}"
    print(finish_msg)
    logger.info(finish_msg)

if __name__ == "__main__":
    main()