# debug_data.py
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--train_file', type=str, required=True)
args = parser.parse_args()

# 加载 meta.pkl 获取预期的 vocab_size
meta_path = f"{args.data_dir}/meta.pkl"
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
print(f"✅ Meta Info: Expected vocab_size = {vocab_size}")

# 加载 .bin 文件
data_path = f"{args.data_dir}/{args.train_file}"
data = np.memmap(data_path, dtype=np.uint16, mode='r')

# 查找数据中实际的最大token ID
max_token_id = data.max()
print(f"✅ Data Info: Found maximum token ID in {args.train_file} = {max_token_id}")

# 进行检查
if max_token_id >= vocab_size:
    print("\n❌ Mismatch Found! The maximum token ID in the data is out of bounds for the embedding layer.")
    print(f"   The model expects token IDs from 0 to {vocab_size - 1}, but found a value of {max_token_id}.")
else:
    print("\n✅ OK! Data seems consistent with vocab_size.")