import os
import glob

def fix_format(input_file, output_file):
    """移除路径长度字段，统一格式"""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 3:
            # 当前格式：src tgt path_len node0 node1 ...
            # 目标格式：src tgt node0 node1 ...
            src, tgt, path_len = parts[0], parts[1], parts[2]
            path = parts[3:]  # 跳过path_len
            fixed_line = ' '.join([src, tgt] + path)
            fixed_lines.append(fixed_line)
    
    with open(output_file, 'w') as f:
        for line in fixed_lines:
            f.write(line + '\n')
    
    return len(fixed_lines)

# 修复所有alpha_mixing文件
input_dir = 'alpha_mixing_results'
output_dir = 'alpha_mixing_fixed'
os.makedirs(output_dir, exist_ok=True)

txt_files = glob.glob(os.path.join(input_dir, '*.txt'))
for txt_file in sorted(txt_files):
    basename = os.path.basename(txt_file)
    output_file = os.path.join(output_dir, basename)
    count = fix_format(txt_file, output_file)
    print(f"Fixed {basename}: {count} paths")

# 同时复制experiment_summary.pkl
import shutil
shutil.copy(
    os.path.join(input_dir, 'experiment_summary.pkl'),
    os.path.join(output_dir, 'experiment_summary.pkl')
)

print(f"\n✅ Fixed data saved to {output_dir}/")