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
        seq_emb = self.item_embedding(seq)
        seq_emb = self.transform(seq_emb)
        return seq_emb


def mask_sequence(seq, overlap_rate=0.2, mask_rate=0.6, strategy='random', block_size=5):

    if isinstance(seq, np.ndarray):
        seq = torch.tensor(seq)

    seq = seq.to(dtype=torch.long)
    batch_size, seq_len = seq.size()

    total_mask_len = int(seq_len * mask_rate)
    overlap_len = int(seq_len * overlap_rate)
    unique_len = total_mask_len - overlap_len

    def generate_random_mask():
        indices = torch.randperm(seq_len)[:total_mask_len]
        return indices

    def generate_block_mask():
        num_blocks = total_mask_len // block_size
        start_indices = torch.randint(0, seq_len - block_size + 1, (num_blocks,))
        indices = torch.cat([torch.arange(start, start + block_size) for start in start_indices])
        return indices[:total_mask_len]

    def generate_hybrid_mask():
        random_indices = torch.randperm(seq_len)[:unique_len]
        block_indices = generate_block_mask()[:overlap_len]
        return torch.cat([random_indices, block_indices])

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

    overlap_indices = mask_indices_1[:overlap_len]
    mask_indices_2 = torch.cat([overlap_indices, mask_indices_2[overlap_len:]])

    mask_1 = torch.zeros(seq_len, dtype=torch.bool)
    mask_2 = torch.zeros(seq_len, dtype=torch.bool)
    mask_1[mask_indices_1] = True
    mask_2[mask_indices_2] = True

    masked_seq_1 = seq.clone()
    masked_seq_2 = seq.clone()
    masked_seq_1[:, ~mask_1] = 0
    masked_seq_2[:, ~mask_2] = 0

    return masked_seq_1, masked_seq_2


def contrastive_loss(model, seq, negative_seq, temperature=0.5):
    seq_masked_1 = mask_sequence(seq)
    seq_masked_2 = mask_sequence(seq)

    emb_1 = model(seq_masked_1).mean(dim=1)
    emb_2 = model(seq_masked_2).mean(dim=1)

    negative_emb_1 = model(mask_sequence(negative_seq)).mean(dim=1)
    negative_emb_2 = model(mask_sequence(negative_seq)).mean(dim=1)

    positive_similarity = F.cosine_similarity(emb_1, emb_2, dim=-1)
    negative_similarity_1 = F.cosine_similarity(emb_1, negative_emb_1, dim=-1)
    negative_similarity_2 = F.cosine_similarity(emb_2, negative_emb_2, dim=-1)

    pos_loss = -torch.log(torch.exp(positive_similarity / temperature))
    neg_loss_1 = -torch.log(1 - torch.exp(negative_similarity_1 / temperature))
    neg_loss_2 = -torch.log(1 - torch.exp(negative_similarity_2 / temperature))

    return (pos_loss.mean() + neg_loss_1.mean() + neg_loss_2.mean()) / 2


