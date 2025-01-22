import torch
import torch.nn.functional as F
import random
from src.argument import parse_args
import numpy as np


class ContrastiveLearningModel(torch.nn.Module):
    def __init__(self, item_emb_size, seq_len, hidden_units):
        super(ContrastiveLearningModel, self).__init__()
        self.item_emb_size = item_emb_size
        self.seq_len = seq_len
        self.hidden_units = hidden_units
        self.item_embedding = torch.nn.Embedding(item_emb_size, hidden_units)
        self.transform = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, seq):
        # 获取序列的item嵌入
        seq_emb = self.item_embedding(seq)
        seq_emb = self.transform(seq_emb)
        return seq_emb


# def mask_sequence(seq, mask_rate=0.5):
#     """
#     随机遮住一半序列，返回mask后的序列和未mask的序列。
#     """
#     # print(seq.shape)
#     # print(type(seq))
#     # seq = torch.tensor(seq)
#     # print(seq.size())
#     seq_len = parse_args().maxlen
#     mask = torch.rand(seq_len) < mask_rate  # 根据mask_rate随机生成mask
#     masked_seq = seq
#     masked_seq[:, mask] = 0  # 遮住部分序列
#     unmasked_seq = seq
#     unmasked_seq[:, ~mask] = 0  # 保留未遮住的部分
#     return masked_seq, unmasked_seq
# def mask_sequence(seq, overlap_rate=0.2, mask_rate=0.6):
#     """
#     随机遮住一部分序列生成两个子序列，它们占原始序列的 mask_rate，且具有 overlap_rate 的重合部分。
#     """
#     # 如果输入是 numpy.ndarray，需要先转成 torch.Tensor
#     if isinstance(seq, np.ndarray):
#         seq = torch.tensor(seq)
#
#     # 确保 seq 是 torch.Tensor
#     seq = seq.to(dtype=torch.long)  # 或根据需要指定 dtype
#     seq_len = parse_args().maxlen  # 假设输入为 [batch_size, seq_len]
#     total_mask_len = int(seq_len * mask_rate)
#     overlap_len = int(seq_len * overlap_rate)
#     unique_len = total_mask_len - overlap_len
#
#     # 随机生成重合部分的索引
#     overlap_indices = torch.randperm(seq_len)[:overlap_len]
#
#     # 为两个序列生成各自的独特部分索引
#     remaining_indices = torch.tensor([i for i in range(seq_len) if i not in overlap_indices])
#     unique_indices_1 = remaining_indices[torch.randperm(len(remaining_indices))[:unique_len]]
#     unique_indices_2 = remaining_indices[torch.randperm(len(remaining_indices))[:unique_len]]
#
#     # 构造两个掩码
#     mask_1 = torch.zeros(seq_len, dtype=torch.bool)
#     mask_2 = torch.zeros(seq_len, dtype=torch.bool)
#     mask_1[overlap_indices] = True
#     mask_2[overlap_indices] = True
#     mask_1[unique_indices_1] = True
#     mask_2[unique_indices_2] = True
#
#     # 应用掩码
#     masked_seq_1 = seq.clone()
#     masked_seq_2 = seq.clone()
#     masked_seq_1[:, ~mask_1] = 0  # 遮住未选中的部分
#     masked_seq_2[:, ~mask_2] = 0  # 遮住未选中的部分
#
#     return masked_seq_1, masked_seq_2

def mask_sequence(seq, overlap_rate=0.2, mask_rate=0.6, strategy='random', block_size=5):
    """
    随机遮住一部分序列生成两个子序列，支持多种掩码策略（随机、区块、混合）。

    Args:
        seq (torch.Tensor or np.ndarray): 输入序列，形状为 [batch_size, seq_len]。
        overlap_rate (float): 两个子序列之间的重叠比例。
        mask_rate (float): 每个子序列的掩码比例。
        strategy (str): 掩码策略，可选值为 ['random', 'block', 'hybrid']。
        block_size (int): 如果使用 block 掩码，指定区块的大小。

    Returns:
        masked_seq_1 (torch.Tensor): 第一个子序列。
        masked_seq_2 (torch.Tensor): 第二个子序列。
    """
    # 如果输入是 numpy.ndarray，需要先转成 torch.Tensor
    if isinstance(seq, np.ndarray):
        seq = torch.tensor(seq)

    # 确保 seq 是 torch.Tensor
    seq = seq.to(dtype=torch.long)
    batch_size, seq_len = seq.size()

    # 计算总掩码长度和重叠长度
    total_mask_len = int(seq_len * mask_rate)
    overlap_len = int(seq_len * overlap_rate)
    unique_len = total_mask_len - overlap_len

    def generate_random_mask():
        """随机掩码生成函数"""
        indices = torch.randperm(seq_len)[:total_mask_len]
        return indices

    def generate_block_mask():
        """连续区块掩码生成函数"""
        num_blocks = total_mask_len // block_size
        start_indices = torch.randint(0, seq_len - block_size + 1, (num_blocks,))
        indices = torch.cat([torch.arange(start, start + block_size) for start in start_indices])
        return indices[:total_mask_len]  # 保证掩码长度不超过 total_mask_len

    def generate_hybrid_mask():
        """混合掩码生成函数"""
        random_indices = torch.randperm(seq_len)[:unique_len]
        block_indices = generate_block_mask()[:overlap_len]
        return torch.cat([random_indices, block_indices])

    # 根据策略生成掩码索引
    if strategy == 'random':
        mask_indices_1 = generate_random_mask()
        mask_indices_2 = generate_random_mask()
    elif strategy == 'block':
        mask_indices_1 = generate_block_mask()
        mask_indices_2 = generate_block_mask()
    elif strategy == 'hybrid':
        mask_indices_1 = generate_hybrid_mask()
        mask_indices_2 = generate_hybrid_mask()
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")

    # 应用重叠部分
    overlap_indices = mask_indices_1[:overlap_len]
    mask_indices_2 = torch.cat([overlap_indices, mask_indices_2[overlap_len:]])

    # 构造掩码
    mask_1 = torch.zeros(seq_len, dtype=torch.bool)
    mask_2 = torch.zeros(seq_len, dtype=torch.bool)
    mask_1[mask_indices_1] = True
    mask_2[mask_indices_2] = True

    # 应用掩码到序列
    masked_seq_1 = seq.clone()
    masked_seq_2 = seq.clone()
    masked_seq_1[:, ~mask_1] = 0  # 遮住未选中的部分
    masked_seq_2[:, ~mask_2] = 0  # 遮住未选中的部分

    return masked_seq_1, masked_seq_2


def contrastive_loss(model, seq, negative_seq, temperature=0.5):
    """
    计算对比学习的损失
    seq: 原始用户序列
    negative_seq: 其他用户序列作为负样本
    """
    # 随机遮住一半生成两个子序列的embedding
    seq_masked_1 = mask_sequence(seq)
    seq_masked_2 = mask_sequence(seq)

    emb_1 = model(seq_masked_1).mean(dim=1)  # 平均池化生成嵌入
    emb_2 = model(seq_masked_2).mean(dim=1)

    # 负样本用户序列处理
    negative_emb_1 = model(mask_sequence(negative_seq)).mean(dim=1)
    negative_emb_2 = model(mask_sequence(negative_seq)).mean(dim=1)

    # 计算相似度
    positive_similarity = F.cosine_similarity(emb_1, emb_2, dim=-1)
    negative_similarity_1 = F.cosine_similarity(emb_1, negative_emb_1, dim=-1)
    negative_similarity_2 = F.cosine_similarity(emb_2, negative_emb_2, dim=-1)

    # 损失函数：正样本越相似越好，负样本越不相似越好
    pos_loss = -torch.log(torch.exp(positive_similarity / temperature))
    neg_loss_1 = -torch.log(1 - torch.exp(negative_similarity_1 / temperature))
    neg_loss_2 = -torch.log(1 - torch.exp(negative_similarity_2 / temperature))

    return (pos_loss.mean() + neg_loss_1.mean() + neg_loss_2.mean()) / 2

# 示例训练步骤
# model = ContrastiveLearningModel(item_emb_size=1000, seq_len=50, hidden_units=128)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# for epoch in range(epochs):
#     # 输入用户序列和负样本序列
#     user_seq = torch.randint(0, 1000, (batch_size, 50))  # 随机生成用户序列
#     neg_user_seq = torch.randint(0, 1000, (batch_size, 50))  # 随机生成负样本序列
#
#     # 计算对比损失
#     loss = contrastive_loss(model, user_seq, neg_user_seq)
#
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
