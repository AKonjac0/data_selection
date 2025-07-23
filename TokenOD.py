import json
import torch
from transformers import AutoTokenizer, AutoModel
import math
import time

def load_and_tokenize(
    json_path: str,
    model_name: str = 'bert-base-uncased',
    device: str = None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    max_len = tokenizer.model_max_length

    X_list = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        parts = [item.get('instruction', '').strip(), item.get('input', '').strip(), item.get('output', '').strip()]
        text = '\n\n'.join([p for p in parts if p])

        tokens = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_len,
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden = outputs.hidden_states[-1].squeeze(0)

        seq_len, d = hidden.size()
        X_i = torch.zeros((seq_len, d), device=device)
        X_i[1:] = hidden[:-1]

        X_list.append(X_i)

    return X_list, device

def save_tokenized(X_list, save_path):
    torch.save(X_list, save_path)

def load_tokenized(load_path, device):
    return torch.load(load_path, map_location=device)

def greedy_token_od(X_list, n, device, batch_size=64):
    """
    基于 TokenOD 算法的 Greedy 选句实现，全部操作在 GPU 上完成。
    输入：
      - X_list: List[Tensor], 每个元素形状为 (M_i, d)，已在 device 上
      - n: 要选择的句子数
      - device: 'cuda' 或 'cpu'
      - batch_size: 每次并行计算增益的 batch 大小 B
    输出：
      - selected: List[int] 被选中句子的全局索引列表
    """
    N = len(X_list)
    d = X_list[0].size(1)

    # 初始化 V = I_d
    V = torch.eye(d, device=device)
    # 已选索引
    selected = []
    # 缓存增益 g_i，初始为 +inf
    g = torch.full((N,), float('inf'), device=device)
    print("start selecting")
    for t in range(n):
        # 本轮的当前最高增益
        
        t0 = time.time()
        g_max = torch.tensor(0., device=device)

        # 按 batch 分块更新 g_i
        num_batches = math.ceil(N / batch_size)
        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, N)
            idx_block = torch.arange(start, end, device=device)

            # 对于尚未更新过且可能超过 g_max 的样本，计算其真实增益
            mask = g[idx_block] > g_max
            if mask.any():
                idx_masked = idx_block[mask]
                # 计算这些样本的 C_i = X_i^T X_i
                C_list = []
                for i in idx_masked.tolist():
                    Xi = X_list[i]              # (M_i, d)
                    Ci = Xi.transpose(0, 1) @ Xi  # (d, d)
                    C_list.append(Ci)
                C_batch = torch.stack(C_list, dim=0)  # (B', d, d)

                # 先算 det(V)
                sign_V, logdet_V = torch.slogdet(V)

                # 并行算 det(V + C_i)
                M = V.unsqueeze(0) + C_batch       # (B', d, d)
                signs, logdets = torch.slogdet(M)  # (B',), (B',)
                gains = logdets - logdet_V         # (B',)

                # 写回 g
                g[idx_masked] = gains

            # 更新 g_max
            block_max = g[idx_block].max()
            g_max = torch.max(g_max, block_max)

        # 在所有未选样本中选出最大增益样本 k
        g_tmp = g.clone()
        if selected:
            g_tmp[torch.tensor(selected, device=device)] = float('-inf')
        k = int(torch.argmax(g_tmp).item())

        # 记录并更新 V
        selected.append(k)
        Xk = X_list[k]                 # (M_k, d)
        Ck = Xk.transpose(0, 1) @ Xk   # (d, d)
        V = V + Ck
        t1 = time.time()
        print(f"Step {t + 1} / {n}, time: {t1 - t0}s")
    
    return selected


# --------------------------------------

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """保存数据到 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_sample_json():
    # 假设你的 JSON 文件路径为 'data.json'
    json_file_path = '../data/alpaca_data_en_52k.json'
    # 索引表文件路径假设为 'index.txt'，每一行包含一个数字代表原始 JSON 文件中的索引
    index_file_path = 'output_1k.txt'
    # 输出文件路径
    output_file_path = '../data/alpaca_data_en_1k_TokenOD.json'

    # 加载原始 JSON 数据
    data_list = load_json(json_file_path)
    
    # 读取索引表
    with open(index_file_path, 'r') as idx_f:
        indices = [int(line.strip()) for line in idx_f if line.strip()]  # 忽略空行

    # 提取数据
    filtered_data = [data_list[i] for i in indices if 0 <= i < len(data_list)]

    # 保存到新文件
    save_json(filtered_data, output_file_path)
    print(f"Filtered data has been saved to {output_file_path}")

# ------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Greedy TokenOD on Alpaca dataset')
    parser.add_argument('--input', type=str, default='alpaca_en_52k.json', help='Path to input JSON dataset')
    parser.add_argument('--output', type=str, default='output_1k.txt', help='File to write selected indices')
    parser.add_argument('--model', type=str, default='bert-base-uncased', help='HuggingFace model name')
    parser.add_argument('--num_select', type=int, default=100, help='Number of sentences to select n')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda or cpu)')
    parser.add_argument('--cached_tokens', type=str, default=None, help='Optional path to cached token file (.pt)')
    parser.add_argument('--save_tokens', type=str, default=None, help='Optional path to save token file (.pt)')
    parser.add_argument('--save_json_only', type=str, default=None, help='')
    args = parser.parse_args()
    if args.save_json_only is None:
        if args.cached_tokens is not None:
            print(f"Loading cached tokens from {args.cached_tokens}")
            device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
            X_list = load_tokenized(args.cached_tokens, device)
        else:
            X_list, device = load_and_tokenize(args.input, model_name=args.model, device=args.device)
            if args.save_tokens:
                save_tokenized(X_list, args.save_tokens)
                print(f"Saved tokenized features to {args.save_tokens}")
    
        selected = greedy_token_od(X_list, n=args.num_select, device=device)
    
        with open(args.output, 'w') as f:
            for idx in selected:
                f.write(f"{idx}\n")
        print(f"Wrote {len(selected)} selected indices to {args.output}")
    
    get_sample_json()

    
# python TokenOD.py --input ../../data/alpaca_data_en_52k.json --save_tokens ../saves/tokens.pt

# python TokenOD.py --cached_tokens ../saves/tokens.pt
