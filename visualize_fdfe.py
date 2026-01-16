"""
可视化 FrequencyAttention (FDFE) 模块的效果
对比 Item Embedding 序列在经过 FDFE 前后的频谱图（幅度谱）
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data import load_split_data, CCFSeqSplitDataset, Collator
from torch.utils.data import DataLoader
from model import CCFRec, FrequencyAttention
from utils import load_json


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument("--dataset", type=str, default="Musical_Instruments")
    parser.add_argument("--bidirectional", type=bool, default=False)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--cl_weight', type=float, default=0.4)
    parser.add_argument('--mlm_weight', type=float, default=0.6)
    parser.add_argument('--neg_num', type=int, default=49)
    parser.add_argument('--text_types', nargs='+', type=str, default=["meta"])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--learner', type=str, default="AdamW")
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument('--map_path', type=str, default=".emb_map.json")
    parser.add_argument('--text_index_path', type=str, default=".code.pq.64_128.json")
    parser.add_argument('--text_emb_path', type=str, default=".t5.meta.emb.npy")
    parser.add_argument('--lr_scheduler_type', type=str, default="constant")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_his_len', type=int, default=20)
    parser.add_argument('--n_codes_per_lel', type=int, default=256)
    parser.add_argument('--code_level', type=int, default=4)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_layers_cross', type=int, default=2)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--dropout_prob_cross', type=float, default=0.3)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--metrics', type=str, default="recall@5,ndcg@5,recall@10,ndcg@10")
    parser.add_argument('--valid_metric', type=str, default="ndcg@10")
    parser.add_argument("--log_dir", type=str, default="./logs/")
    parser.add_argument("--ckpt_dir", type=str, default="./myckpt/")
    
    # 可视化专用参数
    parser.add_argument("--ckpt_path", type=str, required=True, help="训练好的模型checkpoint路径")
    parser.add_argument("--sample_idx", type=int, default=0, help="要可视化的样本索引")
    parser.add_argument("--save_path", type=str, default="./fdfe_visualization.png", help="保存图片路径")
    
    args, _ = parser.parse_known_args()
    return args


def compute_spectrum(x):
    """
    计算信号的频谱（幅度谱）
    x: [L, H] 或 [H]
    返回: 频率轴和幅度谱
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    # 对序列维度做FFT
    x_fft = torch.fft.rfft(x, dim=0, norm='ortho')
    magnitude = torch.abs(x_fft)  # [freq_bins, H]
    
    # 对所有维度取平均，得到整体频谱
    avg_magnitude = magnitude.mean(dim=-1)  # [freq_bins]
    
    L = x.shape[0]
    freq_bins = avg_magnitude.shape[0]
    freqs = torch.fft.rfftfreq(L)[:freq_bins]
    
    return freqs.numpy(), avg_magnitude.numpy()


def visualize_fdfe_effect(model, data_batch, device, sample_idx=0, save_path="./fdfe_visualization.png"):
    """
    可视化 FDFE 前后的频谱变化
    """
    model.eval()
    
    with torch.no_grad():
        item_seq = data_batch["item_inters"].to(device)
        code_seq = data_batch["code_inters"].to(device)
        session_ids = data_batch["session_inters"].to(device)
        inter_lens = data_batch["inter_lens"].to(device)
        
        B, L = item_seq.size(0), item_seq.size(1)
        item_flatten_seq = item_seq.reshape(-1)
        query_seq_emb = model.query_code_embedding(code_seq)
        
        # 获取文本嵌入
        text_embs = []
        for i in range(model.text_num):
            text_emb = model.item_text_embedding[i](item_flatten_seq)
            text_embs.append(text_emb)
        encoder_output = torch.stack(text_embs, dim=1)
        
        # 通过 QFormer 得到 item embedding
        item_seq_emb = model.qformer(query_seq_emb, encoder_output)[-1]
        item_emb = item_seq_emb.mean(dim=1) + query_seq_emb.mean(dim=1)
        item_emb = item_emb.view(B, L, -1)
        
        # FDFE 之前的 embedding
        item_emb_before = item_emb.clone()
        
        # FDFE 之后的 embedding
        item_emb_after = model.fourier_attention(item_emb)
        
        # 选择一个样本进行可视化
        seq_len = inter_lens[sample_idx].item()
        emb_before = item_emb_before[sample_idx, :seq_len, :].cpu()  # [seq_len, H]
        emb_after = item_emb_after[sample_idx, :seq_len, :].cpu()    # [seq_len, H]
        
        # 计算频谱
        freqs_before, mag_before = compute_spectrum(emb_before)
        freqs_after, mag_after = compute_spectrum(emb_after)
        
        # 同时获取 FDFE 内部的中间结果用于更详细的分析
        x = item_emb_before[sample_idx:sample_idx+1, :seq_len, :].cpu()
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        freq_magnitude_raw = torch.abs(x_fft).squeeze(0)  # [freq_bins, H]
        
    # ==================== 绘图 ====================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 原始 vs 增强后的频谱对比（平均）
    ax1 = axes[0, 0]
    ax1.plot(freqs_before, mag_before, 'b-', label='Before FDFE', linewidth=2, alpha=0.8)
    ax1.plot(freqs_after, mag_after, 'r-', label='After FDFE', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Normalized Frequency', fontsize=12)
    ax1.set_ylabel('Magnitude (Averaged over dims)', fontsize=12)
    ax1.set_title('Spectrum Comparison: Before vs After FDFE', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. 频谱差异图
    ax2 = axes[0, 1]
    diff = mag_after - mag_before
    colors = ['green' if d >= 0 else 'red' for d in diff]
    ax2.bar(range(len(diff)), diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Frequency Bin', fontsize=12)
    ax2.set_ylabel('Magnitude Difference', fontsize=12)
    ax2.set_title('Spectrum Difference (After - Before)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 3. 原始频谱热力图
    ax3 = axes[0, 2]
    im3 = ax3.imshow(freq_magnitude_raw.numpy().T, aspect='auto', cmap='viridis', origin='lower')
    ax3.set_xlabel('Frequency Bin', fontsize=12)
    ax3.set_ylabel('Embedding Dimension', fontsize=12)
    ax3.set_title('Original Frequency Magnitude Heatmap', fontsize=14)
    plt.colorbar(im3, ax=ax3)
    
    # 4. 时域信号对比（选择几个维度）
    ax4 = axes[1, 0]
    dims_to_show = [0, emb_before.shape[1]//4, emb_before.shape[1]//2, -1]
    for i, dim in enumerate(dims_to_show):
        ax4.plot(emb_before[:, dim].numpy(), '--', alpha=0.6, label=f'Before (dim {dim})')
        ax4.plot(emb_after[:, dim].numpy(), '-', alpha=0.8, label=f'After (dim {dim})')
    ax4.set_xlabel('Sequence Position', fontsize=12)
    ax4.set_ylabel('Embedding Value', fontsize=12)
    ax4.set_title('Time Domain: Selected Dimensions', fontsize=14)
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    
    # 5. 各频率分量的能量分布
    ax5 = axes[1, 1]
    energy_before = mag_before ** 2
    energy_after = mag_after ** 2
    x_pos = np.arange(len(energy_before))
    width = 0.35
    ax5.bar(x_pos - width/2, energy_before, width, label='Before FDFE', alpha=0.8)
    ax5.bar(x_pos + width/2, energy_after, width, label='After FDFE', alpha=0.8)
    ax5.set_xlabel('Frequency Bin', fontsize=12)
    ax5.set_ylabel('Energy (Magnitude²)', fontsize=12)
    ax5.set_title('Energy Distribution by Frequency', fontsize=14)
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # 6. 增强比例图
    ax6 = axes[1, 2]
    ratio = np.where(mag_before > 1e-8, mag_after / mag_before, 1.0)
    ax6.bar(range(len(ratio)), ratio, color='purple', alpha=0.7)
    ax6.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='No change (ratio=1)')
    ax6.set_xlabel('Frequency Bin', fontsize=12)
    ax6.set_ylabel('Enhancement Ratio (After/Before)', fontsize=12)
    ax6.set_title('FDFE Enhancement Ratio by Frequency', fontsize=14)
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'FrequencyAttention (FDFE) Visualization\nSample Index: {sample_idx}, Sequence Length: {seq_len}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存到: {save_path}")
    
    # 打印一些统计信息
    print("\n========== FDFE 效果统计 ==========")
    print(f"序列长度: {seq_len}")
    print(f"Embedding 维度: {emb_before.shape[1]}")
    print(f"频率分量数: {len(mag_before)}")
    print(f"\n原始频谱能量总和: {(mag_before**2).sum():.4f}")
    print(f"增强后频谱能量总和: {(mag_after**2).sum():.4f}")
    print(f"能量变化比例: {(mag_after**2).sum() / (mag_before**2).sum():.4f}")
    print(f"\n低频增强比例 (前1/3): {ratio[:len(ratio)//3].mean():.4f}")
    print(f"中频增强比例 (中1/3): {ratio[len(ratio)//3:2*len(ratio)//3].mean():.4f}")
    print(f"高频增强比例 (后1/3): {ratio[2*len(ratio)//3:].mean():.4f}")


def main():
    args = parse_arguments()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    print(f"加载数据集: {args.dataset}")
    
    # 加载数据
    data_path = args.data_path
    dataset = args.dataset
    dataset_path = os.path.join(data_path, dataset)
    
    item2id, n_items, train, val, test = load_split_data(args)
    index = load_json(os.path.join(dataset_path, dataset + args.text_index_path))
    
    # 使用测试集
    test_dataset = CCFSeqSplitDataset(args, n_items, test, index, 'test')
    collator = Collator(args)
    test_loader = DataLoader(test_dataset, num_workers=0, collate_fn=collator,
                             batch_size=32, shuffle=False, pin_memory=False)
    
    # 加载文本嵌入
    text_embs = []
    for ttype in args.text_types:
        text_emb_file = f".t5.{ttype}.emb.npy"
        text_emb = np.load(os.path.join(args.data_path, args.dataset, args.dataset + text_emb_file))
        text_emb = PCA(n_components=args.embedding_size, whiten=True).fit_transform(text_emb)
        text_embs.append(text_emb)
    args.text_embedding_size = text_embs[0].shape[-1]
    
    # 初始化模型
    print("初始化模型...")
    model = CCFRec(args, test_dataset, index, device).to(device)
    
    # 加载文本嵌入到模型
    for i in range(len(args.text_types)):
        model.item_text_embedding[i].weight.data[1:] = torch.tensor(text_embs[i], dtype=torch.float32, device=device)
    
    # 加载训练好的模型权重
    print(f"加载模型权重: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    # 获取一个batch的数据
    data_batch = next(iter(test_loader))
    
    # 可视化
    print(f"\n开始可视化样本 {args.sample_idx}...")
    visualize_fdfe_effect(model, data_batch, device, 
                          sample_idx=args.sample_idx, 
                          save_path=args.save_path)


if __name__ == "__main__":
    main()
