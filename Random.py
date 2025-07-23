import json
import random

# 配置参数
input_file = '../../data/alpaca_data_en_52k.json'  # 输入文件名
output_file = '../../data/alpaca_data_en_1k.json'  # 输出文件名
sample_size = 1000  # 需要抽取的样本数量

# 读取原始数据
with open(input_file, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

# 检查数据量是否足够
if len(full_data) < sample_size:
    raise ValueError(f"数据集只有 {len(full_data)} 条，不足 {sample_size} 条")

# 随机抽取样本
sampled_data = random.sample(full_data, sample_size)

# 保存采样数据
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, indent=2, ensure_ascii=False)

print(f"成功从 {len(full_data)} 条数据中随机抽取 {sample_size} 条，已保存到 {output_file}")