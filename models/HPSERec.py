import json
import os
import pickle

import math
import numpy as np
import torch
from src.utils2 import setupt_logger, set_random_seeds, Checker, devicer
from src.sampler import NegativeSampler
from embedder import embedder
from src.data import ValidData, TestData, TrainData
import torch.utils.data as data_utils
from src.argument import parse_args
from src.data_augmentation import mask_sequence, contrastive_loss
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm


def distillation_loss(student_logits, teacher_logits, temperature=0.7):
    student_probs = F.softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    # 计算KL散度
    loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')
    return loss


def bpr_loss(pos_logits, neg_logits):
    return -torch.mean(torch.log(torch.sigmoid(pos_logits - neg_logits)))


class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, bias=False),  # 输入维度由 share_model 决定
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, output_dim, bias=True)  # 投影到对比学习的空间，默认为 128
        )

    def forward(self, x):
        return self.mlp(x)


class Trainer(embedder):

    def __init__(self, args, loggers=None):
        self.args = args
        if loggers is None:
            self.loggers = []

            self.loggers = [setupt_logger(args, f'log/{args.model}/{args.dataset}/{args.dataset}{i}',
                                          name=f'Model{i}', filename=f'log{i}.txt') for i in range(args.num_experts)]
            self.logger = self.logger = setupt_logger(args, f'log/{args.model}/{args.dataset}', name='Model',
                                                      filename=f'log.txt')

        embedder.__init__(self, args, self.logger)
        self.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)
        self.split_head_tail()
        self.save_user_item_context()

    def train(self):
        """
        Train the model
        """
        set_random_seeds(self.args.seed)
        self.logger.info(f"============Start Training (MyModel)=======================")
        self.share_model = ShareModel(self.item_num, self.args.hidden_units, self.args.maxlen,
                                      self.args.dropout_rate).to(self.device)
        self.experts = [
            Expert(self.args, self.share_model).to(self.device) for _ in range(self.args.num_experts)
        ]
        for i in range(self.args.num_experts):
            self.init_param(self.experts[i])
        self.inference_negative_sampler = NegativeSampler(self.args, self.dataset)
        self.inference_negative_samplers = [NegativeSampler(self.args, self.datasets[i]) for i in
                                            range(self.args.num_experts)]
        self.projection_head = ProjectionHead(64, 256, 128).to(self.device)
        df = pd.read_csv('dataset/BeautyTail.txt', sep=' ', header=None)
        df.columns = ['userId', 'movieId', 'timestamp']

        set_tail = set(df['movieId'])
        self.tail_num = len(set_tail)

        # Build the train, valid, test datasets

        train_dataset = TrainData(self.train_data, self.user_num, self.item_num, batch_size=self.args.batch_size,
                                  maxlen=self.args.maxlen)
        train_loader = data_utils.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                             drop_last=False)

        train_datasets = [
            TrainData(self.train_datas[i], self.user_nums[i], self.item_nums[i], batch_size=self.args.batch_size,
                      maxlen=self.args.maxlen) for i in range(self.args.num_experts)]

        train_loaders = [data_utils.DataLoader(train_datasets[i], batch_size=self.args.batch_size, shuffle=True,
                                               drop_last=False) for i in range(self.args.num_experts)]

        valid_datasets = [
            ValidData(self.args, self.item_context, self.train_datas[i], self.test_datas[i], self.valid_datas[i],
                      self.inference_negative_samplers[i]) for i in range(self.args.num_experts)]

        self.valid_loaders = [data_utils.DataLoader(valid_datasets[i], batch_size=self.args.batch_size, shuffle=True,
                                                    drop_last=False) for i in range(self.args.num_experts)]

        test_dataset = TestData(self.args, self.item_context, self.train_data, self.test_data,
                                self.valid_data,
                                self.inference_negative_sampler)
        self.test_loader = data_utils.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True,
                                                 drop_last=False)

        test_datasets = [TestData(self.args, self.item_context, self.train_datas[i], self.test_datas[i],
                                  self.valid_datas[i],
                                  self.inference_negative_samplers[i]) for i in range(self.args.num_experts)]

        self.test_loaders = [data_utils.DataLoader(test_datasets[i], batch_size=self.args.batch_size, shuffle=True,
                                                   drop_last=False) for i in range(self.args.num_experts)]

        self.validcheck = Checker(self.logger)
        self.validchecks = [Checker(self.loggers[i]) for i in range(self.args.num_experts)]

        bce_criterion = torch.nn.BCEWithLogitsLoss()
        adam_optimizers = [torch.optim.Adam(self.experts[i].parameters(), lr=self.args.lr, betas=(0.9, 0.98)) for i in
                           range(self.args.num_experts)]
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()  # 二分类损失函数

        big_epoch = 200

        user_embs = [{} for _ in range(self.args.num_experts)]
        user_lens = [{} for _ in range(self.args.num_experts)]
        n = self.args.num_experts - 1

        for E in tqdm(range(big_epoch)):
            for i in range(self.args.num_experts):
                print(f"len(train_loaders[i]):{len(train_loaders[i])}")
                for epoch in range(1, self.args.e_max + 1):
                    self.experts[i].train()
                    training_loss = 0.0

                    for _, (u, seq, pos, neg) in enumerate(train_loaders[i]):
                        adam_optimizers[i].zero_grad()
                        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                        pos_logits, neg_logits, _ = self.experts[i](u, seq, pos, neg)

                        pos_labels, neg_labels = torch.ones(pos_logits.shape).to(self.device), torch.zeros(
                            neg_logits.shape).to(self.device)
                        indices = np.where(pos != 0)
                        pos_logits_filtered = pos_logits[indices]
                        neg_logits_filtered = neg_logits[indices]
                        loss = bpr_loss(pos_logits_filtered, neg_logits_filtered)
                        loss += self.args.l2_emb * sum(
                            torch.norm(param) for param in self.share_model.item_emb.parameters())

                        user_rep = self.experts[i].user_representation(seq)
                        for idx in range(len(user_rep)):
                            # print(u[idx])
                            user_embs[i][u[idx]] = user_rep[idx].detach()
                            c = 0
                            for d in seq[idx]:
                                if d != 0:
                                    c += 1
                            user_lens[i][u[idx]] = c

                        # 总损失
                        total_loss = loss
                        total_loss.backward()
                        adam_optimizers[i].step()

                        training_loss += total_loss.item()

                    if epoch % 10 == 0:
                        print(
                            f'Epoch: {epoch}, Evaluating: Dataset({self.args.dataset}{i}), Model: ({self.args.model}), Training Loss: {training_loss:.8f}')

                    if epoch % 5 == 0:
                        self.experts[i].eval()
                        result_valid = self.evaluate_with_tail(self.experts[i], n, k=10, is_valid='valid')
                        best_valid = self.validchecks[i](result_valid, epoch, self.experts[i],
                                                         f'{self.args.model}_{self.args.dataset}{i}.pth')

                        # Evaluation
                with torch.no_grad():
                    self.validchecks[i].best_model.eval()
                    result_5 = self.evaluate_with_tail(self.validchecks[i].best_model, n, k=5, is_valid='test')
                    result_10 = self.evaluate_with_tail(self.validchecks[i].best_model, n, k=10, is_valid='test')
                    self.validchecks[i].refine_test_result(result_5, result_10)
                    self.validchecks[i].print_result()

                folder = f"save_model/{self.args.dataset}"
                os.makedirs(folder, exist_ok=True)
                torch.save(self.validchecks[i].best_model.state_dict(),
                           os.path.join(folder, self.validchecks[i].best_name))

                torch.cuda.empty_cache()
                self.validchecks[i].print_result()
                for j in range(i + 1, i + 2):
                    if j == self.args.num_experts:
                        break
                    for epoch in range(1, self.args.e_max + 1):
                        self.experts[i].eval()
                        self.experts[j].train()
                        training_loss = 0.0

                        for _, (u, seq, pos, neg) in enumerate(train_loaders[i]):
                            adam_optimizers[j].zero_grad()

                            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

                            indices = np.where(pos != 0)
                            pos_i_logits, neg_i_logits, contrastiveLoss_i = self.experts[i](u, seq, pos, neg)

                            pos_n_logits, neg_n_logits, _ = self.experts[j](u, seq, pos, neg)

                            pos_i_labels, neg_i_labels = (
                                torch.ones(pos_i_logits.shape).to(self.device),
                                torch.zeros(neg_i_logits.shape).to(self.device)
                            )

                            pos_n_labels, neg_n_labels = (
                                torch.ones(pos_n_logits.shape).to(self.device),
                                torch.zeros(neg_n_logits.shape).to(self.device)
                            )
                            # 计算学生模型的损失
                            tail_loss = bce_criterion(pos_n_logits[indices], pos_n_labels[indices])
                            tail_loss += bce_criterion(neg_n_logits[indices], neg_n_labels[indices])

                            # 计算蒸馏损失
                            tail_distillation_loss = distillation_loss(pos_i_logits, pos_n_logits,
                                                                       self.args.tau) + distillation_loss(neg_i_logits,
                                                                                                          neg_n_logits,
                                                                                                          self.args.tau)

                            # 总损失
                            loss = tail_loss + tail_distillation_loss

                            loss.backward()
                            adam_optimizers[j].step()
                            training_loss += loss.item()
                    torch.cuda.empty_cache()

            Resurt = self.evaluate(self.validchecks[self.args.num_experts - 1].best_model, self.args.num_experts - 1,
                                   k=10,
                                   is_valid='test')
            print(Resurt)

            for i in range(self.args.num_experts - 2, -1, -1):
                print(f'{i}号专家user_emb')
                print(f"len(train_loaders[i]):{len(train_loaders[i])}")
                for epoch in range(1, self.args.e_max + 1):
                    self.experts[i].train()
                    training_loss = 0.0

                    for _, (u, seq, pos, neg) in enumerate(train_loaders[i]):
                        adam_optimizers[i].zero_grad()
                        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                        user_rep = self.experts[i].user_representation(seq)

                        loss2 = 0
                        for idx in range(len(user_rep)):
                            if u[idx] in user_embs[i + 1]:
                                sui1 = user_lens[i + 1][u[idx]]
                                su = user_lens[i][u[idx]]
                                # print(su)
                                if su == 0:
                                    continue
                                frac = (sui1 - su) / su
                                beta = 2 / (1 + math.e * (self.args.sigamma * (E / big_epoch + frac)))
                                user_emb = beta * user_embs[i + 1][u[idx]] + user_embs[i][u[idx]]
                                loss2 += ((user_rep[idx] - user_emb) ** 2).mean()

                        # print(loss)
                        # print(loss2)

                        # 总损失
                        total_loss = self.args.alpha * loss2
                        # print(total_loss)
                        total_loss.backward()
                        adam_optimizers[i].step()

                        training_loss += total_loss.item()

                    if epoch % 10 == 0:
                        print(
                            f'Epoch: {epoch}, Evaluating: Dataset({self.args.dataset}{i}), Model: ({self.args.model}), Training Loss: {training_loss:.8f}')

                    if epoch % 5 == 0:
                        self.experts[i].eval()
                        result_valid = self.evaluate_with_tail(self.experts[i], n, k=10, is_valid='valid')
                        best_valid = self.validchecks[i](result_valid, epoch, self.experts[i],
                                                         f'{self.args.model}_{self.args.dataset}{i}.pth')

                        # Evaluation
                with torch.no_grad():
                    self.validchecks[i].best_model.eval()
                    result_5 = self.evaluate_with_tail(self.validchecks[i].best_model, n, k=5, is_valid='test')
                    result_10 = self.evaluate_with_tail(self.validchecks[i].best_model, n, k=10, is_valid='test')
                    self.validchecks[i].refine_test_result(result_5, result_10)
                    self.validchecks[i].print_result()

                folder = f"save_model/{self.args.dataset}"
                os.makedirs(folder, exist_ok=True)
                torch.save(self.validchecks[i].best_model.state_dict(),
                           os.path.join(folder, self.validchecks[i].best_name))
            for i in range(self.args.num_experts):
                model_path = f"save_model/{self.args.dataset}/{self.args.model}_{self.args.dataset}{i}.pth"
                self.experts[i].load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

        Hits = 0.0
        NDCGs = 0.0
        nums = 0.0
        for i in range(self.args.num_experts):
            model_path = f"save_model/{self.args.dataset}/{self.args.model}_{self.args.dataset}{i}.pth"
            self.experts[i].load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
            self.validchecks[i].best_model.eval()
            with torch.no_grad():
                result, Hit, NDCG, n_user = self.evaluate_with_tail(self.validchecks[i].best_model, i, k=10,
                                                                    is_valid='test',
                                                                    is_return=True)
                Hits += Hit
                NDCGs += NDCG
                nums += n_user

        print(" =====TEST_ALL====="
              f"TEST HIT: {Hits / nums:.4f}, "
              f"TEST NDCG: {NDCGs / nums:.4f}")

        for i in range(self.args.num_experts):
            with torch.no_grad():
                self.validchecks[i].best_model.eval()
                result_5 = self.evaluate_with_tail(self.validchecks[i].best_model, i, k=5, is_valid='test')
                result_10 = self.evaluate_with_tail(self.validchecks[i].best_model, i, k=10, is_valid='test')
                self.validchecks[i].refine_test_result(result_5, result_10)
                self.validchecks[i].print_result()

            folder = f"save_model/{self.args.dataset}"
            os.makedirs(folder, exist_ok=True)
            torch.save(self.validchecks[i].best_model.state_dict(), os.path.join(folder, self.validchecks[i].best_name))
            self.validchecks[i].print_result()

        Resurt = self.evaluate(self.validchecks[self.args.num_experts - 1].best_model, self.args.num_experts - 1, k=10,
                               is_valid='test')
        print(Resurt)

    def init_param(self, model):
        """
        Initialization of parameters
        """
        for _, param in model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass

    def evaluate_with_tail(self, model, i, k=10, is_valid='test', is_head='head', is_return=False):
        if is_head == 'head' and is_valid == 'test':
            loader = self.test_loaders[i]
        elif is_head == 'head' and is_valid == 'valid':
            loader = self.valid_loaders[i]
        if is_head == 'Head' and is_valid == 'test':
            loader = self.Head_test_loaders[i]
        elif is_head == 'Head' and is_valid == 'valid':
            loader = self.Head_valid_loaders[i]
        elif is_head == 'tail' and is_valid == 'test':
            loader = self.test_loader_tails[i]
        elif is_head == 'tail' and is_valid == 'valid':
            loader = self.valid_loader_tails[i]

        HIT = 0.0  # Overall HIT
        NDCG = 0.0  # Overall NDCG
        n_all_user = 0.0  # Total number of users

        for _, (u, seq, item_idx, test_idx) in enumerate(loader):
            predictions = -model.predict(seq.numpy(), item_idx.numpy())  # Sequence Encoder
            # print(f"predictions:{predictions.shape}")

            rank = predictions.argsort(1).argsort(1)[:, 0].cpu().numpy()
            n_all_user += len(predictions)
            hit_user = rank < k
            HIT += np.sum(hit_user).item()

            NDCG += np.sum(1 / np.log2(rank[hit_user] + 2)).item()

        # Calculate average HIT and NDCG across all users
        result = {
            'HIT': HIT / n_all_user,
            'NDCG': NDCG / n_all_user
        }

        if not is_return:
            return result
        else:
            return result, HIT, NDCG, n_all_user

    def evaluate(self, model, i, k=10, is_valid='test', is_head='head', is_return=False):
        if is_head == 'head' and is_valid == 'test':
            loader = self.test_loaders[i]
        elif is_head == 'head' and is_valid == 'valid':
            loader = self.valid_loaders[i]
        if is_head == 'Head' and is_valid == 'test':
            loader = self.Head_test_loaders[i]
        elif is_head == 'Head' and is_valid == 'valid':
            loader = self.Head_valid_loaders[i]
        elif is_head == 'tail' and is_valid == 'test':
            loader = self.test_loader_tails[i]
        elif is_head == 'tail' and is_valid == 'valid':
            loader = self.valid_loader_tails[i]

        HIT = 0.0  # Overall Hit
        NDCG = 0.0  # Overall NDCG

        TAIL_USER_NDCG = 0.0
        HEAD_USER_NDCG = 0.0
        TAIL_ITEM_NDCG = 0.0
        HEAD_ITEM_NDCG = 0.0

        TAIL_USER_HIT = 0.0
        HEAD_USER_HIT = 0.0
        TAIL_ITEM_HIT = 0.0
        HEAD_ITEM_HIT = 0.0

        n_all_user = 0.0
        n_head_user = 0.0
        n_tail_user = 0.0
        n_head_item = 0.0
        n_tail_item = 0.0

        for _, (u, seq, item_idx, test_idx) in enumerate(loader):
            u_head = (self.u_head_set[None, ...] == u.numpy()[..., None]).nonzero()[0]  # Index of head users
            u_tail = np.setdiff1d(np.arange(len(u)), u_head)  # Index of tail users
            i_head = (self.i_head_set[None, ...] == test_idx.numpy()[..., None]).nonzero()[0]  # Index of head items
            i_tail = np.setdiff1d(np.arange(len(u)), i_head)  # Index of tail items
            predictions = -model.predict(seq.numpy(), item_idx.numpy())  # Sequence Encoder
            # print(f"predictions:{predictions.shape}")

            rank = predictions.argsort(1).argsort(1)[:, 0].cpu().numpy()
            n_all_user += len(predictions)
            hit_user = rank < k
            ndcg = 1 / np.log2(rank + 2)

            n_head_user += len(u_head)
            n_tail_user += len(u_tail)
            n_head_item += len(i_head)
            n_tail_item += len(i_tail)

            HIT += np.sum(hit_user).item()
            HEAD_USER_HIT += sum(hit_user[u_head])
            TAIL_USER_HIT += sum(hit_user[u_tail])
            HEAD_ITEM_HIT += sum(hit_user[i_head])
            TAIL_ITEM_HIT += sum(hit_user[i_tail])

            NDCG += np.sum(1 / np.log2(rank[hit_user] + 2)).item()
            HEAD_ITEM_NDCG += sum(ndcg[i_head[hit_user[i_head]]])
            TAIL_ITEM_NDCG += sum(ndcg[i_tail[hit_user[i_tail]]])
            HEAD_USER_NDCG += sum(ndcg[u_head[hit_user[u_head]]])
            TAIL_USER_NDCG += sum(ndcg[u_tail[hit_user[u_tail]]])

        result = {'Overall': {'NDCG': NDCG / n_all_user, 'HIT': HIT / n_all_user},
                  'Head_User': {'NDCG': HEAD_USER_NDCG / n_head_user, 'HIT': HEAD_USER_HIT / n_head_user},
                  'Tail_User': {'NDCG': TAIL_USER_NDCG / n_tail_user, 'HIT': TAIL_USER_HIT / n_tail_user},
                  'Head_Item': {'NDCG': HEAD_ITEM_NDCG / n_head_item, 'HIT': HEAD_ITEM_HIT / n_head_item},
                  'Tail_Item': {'NDCG': TAIL_ITEM_NDCG / n_tail_item, 'HIT': TAIL_ITEM_HIT / n_tail_item}
                  }

        if not is_return:
            return result
        else:
            return result, HIT, NDCG, n_all_user


class ShareModel(torch.nn.Module):
    def __init__(self, item_num, hidden_units, maxlen, dropout_rate):
        super(ShareModel, self).__init__()
        self.item_emb = torch.nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

    def get_item_emb(self, item_seq, device):
        item_embs = self.item_emb(torch.LongTensor(item_seq).to(device))
        item_embs *= self.item_emb.embedding_dim ** 0.5  # Scale embedding
        return item_embs

    def get_single_item_emb(self, item_id, device):
        item_emb = self.item_emb(torch.LongTensor([item_id]).to(device))
        item_emb *= self.item_emb.embedding_dim ** 0.5  # Scale embedding
        return item_emb

    def get_pos_emb(self, log_seqs, device):
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        pos_embs = self.pos_emb(torch.LongTensor(positions).to(device))
        return pos_embs

    def apply_dropout(self, seqs):
        return self.emb_dropout(seqs)


class Expert(torch.nn.Module):
    def __init__(self, args, sharemodel):
        super(Expert, self).__init__()
        self.args = args
        self.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu"

        # Use ShareModel for embeddings
        self.share_model = sharemodel

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # MLP 投影头
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(64, 256, bias=False),  # 输入维度由 share_model 决定
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 128, bias=True)  # 投影到对比学习的空间，默认为 128
        )

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        """
        Sequence Encoder: f_{\theta}(S_u)
        """
        seqs = self.share_model.get_item_emb(log_seqs, self.device)
        seqs += self.share_model.get_pos_emb(log_seqs, self.device)
        seqs = self.share_model.apply_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.device)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        """
        Forward pass with contrastive learning and original loss.
        """
        # Original forward pass for positive and negative logits

        log_feats = self.log2feats(log_seqs)
        pos_embs = self.share_model.get_item_emb(pos_seqs, self.device)
        neg_embs = self.share_model.get_item_emb(neg_seqs, self.device)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # Contrastive learning process
        masked_seq1, masked_seq2 = mask_sequence(log_seqs)
        log_feats_1 = self.log2feats(masked_seq1)
        log_feats_2 = self.log2feats(masked_seq2)

        log_feats_1 = self.projector(log_feats_1[:, -1, :])  # 取最后一个 embedding 并通过 MLP
        log_feats_2 = self.projector(log_feats_2[:, -1, :])

        log_feats_1 = F.normalize(log_feats_1, dim=-1)
        log_feats_2 = F.normalize(log_feats_2, dim=-1)

        similarity_matrix = torch.matmul(log_feats_1, log_feats_2.T) / self.args.tau

        batch_size = log_seqs.shape[0]
        target = torch.arange(batch_size).to(self.device)

        contrastive_loss = F.cross_entropy(similarity_matrix, target)

        # Return contrastive loss and original logits (pos_logits, neg_logits)
        return pos_logits, neg_logits, contrastive_loss

    def predict(self, log_seqs, item_indices):
        """
        MELT - Prediction
        """
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]

        item_embs = self.share_model.get_item_emb(item_indices, self.device)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

    def user_representation(self, log_seqs):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        return final_feat


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs
