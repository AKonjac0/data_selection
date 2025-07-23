import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from tqdm import tqdm

def load_data_and_embeddings(json_path, model_name='bert-base-uncased', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    embeddings = []
    for item in tqdm(data, desc='Embedding'):
        parts = [item.get('instruction', '').strip(), item.get('input', '').strip(), item.get('output', '').strip()]
        text = '\n\n'.join([p for p in parts if p])
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length)
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()  # [CLS] embedding
        embeddings.append(cls_emb)
    embeddings = np.stack(embeddings)
    return data, embeddings

def compute_loss(example):
    # TODO: 用你的模型推理，返回 loss
    # 例如: return model_loss(example)
    # 这里只是占位，需用户实现
    return 1.0

def cluster_and_select(data, embeddings, k=1000, z=2, model_name='bert-base-uncased'):
    k_prime = int(0.2 * k)
    k_centers = int(0.2 * (k - k_prime))
    all_indices = np.arange(len(data))
    pretrain_indices = np.random.choice(all_indices, size=k_prime, replace=False)
    pretrain_set = [data[i] for i in pretrain_indices]
    remaining_indices = np.array(list(set(all_indices) - set(pretrain_indices)))
    embeddings_remain = embeddings[remaining_indices]
    data_remain = [data[i] for i in remaining_indices]
    kmeans = KMeans(n_clusters=k_centers, random_state=42)
    labels = kmeans.fit_predict(embeddings_remain)
    centers = kmeans.cluster_centers_
    def find_closest_idx(center, embeddings):
        dists = np.linalg.norm(embeddings - center, axis=1)
        return np.argmin(dists)
    center_indices_local = [find_closest_idx(center, embeddings_remain) for center in centers]
    center_indices = [remaining_indices[i] for i in center_indices_local]
    center_points = [data[i] for i in center_indices]
    center_losses = [compute_loss(data[i]) for i in center_indices]
    Lambda = np.zeros(k_centers)
    for i in range(k_centers):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            Lambda[i] = 1.0
            continue
        c_idx_local = center_indices_local[i]
        c_idx = center_indices[i]
        c_loss = center_losses[i]
        n_sample = int(np.ceil(np.log(100 * k) / np.log(1/0.9)))
        sample_idx = np.random.choice(idx, min(n_sample, len(idx)), replace=False)
        ratios = []
        for j in sample_idx:
            x_idx = remaining_indices[j]
            x_loss = compute_loss(data[x_idx])
            dist = np.linalg.norm(embeddings[x_idx] - embeddings[c_idx]) ** z
            if dist > 1e-8:
                ratios.append(abs(x_loss - c_loss) / dist)
        Lambda[i] = max(ratios) if ratios else 1.0
    tilde_l = np.zeros(len(remaining_indices))
    for i in range(len(remaining_indices)):
        cluster = labels[i]
        c_idx_local = center_indices_local[cluster]
        c_idx = center_indices[cluster]
        c_loss = center_losses[cluster]
        dist = np.linalg.norm(embeddings[remaining_indices[i]] - embeddings[c_idx]) ** z
        tilde_l[i] = c_loss + Lambda[cluster] * dist
    selected_indices = set(center_indices)
    remaining_for_sample = list(set(remaining_indices) - selected_indices)
    remaining_for_sample_local = [np.where(remaining_indices == idx)[0][0] for idx in remaining_for_sample]
    probs = tilde_l[remaining_for_sample_local] / tilde_l[remaining_for_sample_local].sum()
    n_remaining = k - k_prime - k_centers
    sampled_local = np.random.choice(remaining_for_sample_local, size=n_remaining, replace=False, p=probs)
    sampled = [remaining_indices[i] for i in sampled_local]
    final_indices = list(pretrain_indices) + list(selected_indices) + list(sampled)
    sampled_data = [data[i] for i in final_indices]
    return sampled_data, final_indices

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Cluster-based sensitivity sampling for data selection')
    parser.add_argument('--input', type=str, default='alpaca_data_en_52k.json', help='Input JSON file')
    parser.add_argument('--output', type=str, default='sampled_data.json', help='Output JSON file')
    parser.add_argument('--k', type=int, default=1000, help='Number of samples to select')
    parser.add_argument('--model', type=str, default='bert-base-uncased', help='HuggingFace model name')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda or cpu)')
    args = parser.parse_args()
    data, embeddings = load_data_and_embeddings(args.input, model_name=args.model, device=args.device)
    sampled_data, final_indices = cluster_and_select(data, embeddings, k=args.k, model_name=args.model)
    with open(args.output, 'w', encoding='utf-8') as f:
        for d in sampled_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f"Wrote {len(sampled_data)} samples to {args.output}")

if __name__ == '__main__':
    main()


# 然后第二就是把除了那些特别吃计算资源的其他方法都尽量试一试, 然后看看有没有可以改进的