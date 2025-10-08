#!/bin/bash
# analyze_all.sh - 批量分析所有weight gaps

echo "========================================="
echo "Analyzing Weight Gaps for All 7 Models"
echo "========================================="

# G_base
echo -e "\n[1/7] Analyzing G_base..."
python analyze_repair_weight_gap.py \
    --graph G_base \
    --checkpoint_dir out/G_base_d92_seed42_20251008_233913

# G_A_low
echo -e "\n[2/7] Analyzing G_A_low..."
python analyze_repair_weight_gap.py \
    --graph G_A_low \
    --checkpoint_dir out/G_A_low_d92_seed42_20251008_230559

# G_A_high
echo -e "\n[3/7] Analyzing G_A_high..."
python analyze_repair_weight_gap.py \
    --graph G_A_high \
    --checkpoint_dir out/G_A_high_d92_seed42_20251008_232401

# G_B_low
echo -e "\n[4/7] Analyzing G_B_low..."
python analyze_repair_weight_gap.py \
    --graph G_B_low \
    --checkpoint_dir out/G_B_low_d92_seed42_20251008_224846

# G_B_high
echo -e "\n[5/7] Analyzing G_B_high..."
python analyze_repair_weight_gap.py \
    --graph G_B_high \
    --checkpoint_dir out/G_B_high_d92_seed42_20251008_220337

# G_C_low
echo -e "\n[6/7] Analyzing G_C_low..."
python analyze_repair_weight_gap.py \
    --graph G_C_low \
    --checkpoint_dir out/G_C_low_d92_seed42_20251008_221912

# G_C_high
echo -e "\n[7/7] Analyzing G_C_high..."
python analyze_repair_weight_gap.py \
    --graph G_C_high \
    --checkpoint_dir out/G_C_high_d92_seed42_20251008_223335

echo -e "\n========================================="
echo "✅ All analyses complete!"
echo "========================================="