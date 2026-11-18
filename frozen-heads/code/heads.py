"""
挂在冻结 LiLT 上的各类 head 的最小实现（神经网络类）。树模型和原型法不是 nn.Module，在各自章节的脚本里。

统一接口：
  forward(x, mask, boxes)    -> 逐词分数 [B, n, K]
  loss(x, mask, y, boxes)    -> 标量
  predict(x, mask, boxes)    -> (pred [B, n], conf [B, n])
  x     冻结 LiLT 的逐词特征 [B, n, D]（D = 768 / 960 / ...，由输出契约决定）
  mask  哪些位置是真实的词 [B, n]
  boxes 0..1000 的词框 [B, n, 4]，只有版面 head 用
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def loss(self, x, mask, y, boxes):
        return F.cross_entropy(self(x, mask, boxes)[mask], y[mask])

    @torch.no_grad()
    def predict(self, x, mask, boxes):
        conf, pred = self(x, mask, boxes).softmax(-1).max(-1)   # 置信度 = 局部 softmax 的最大值
        return pred, conf


# --------------------------------------------------------------------------- 1. 线性 / 小 MLP
class LinearHead(Head):
    """HF LiltForTokenClassification 和官方 LiLT 的 token 分类头都是 Dropout + 一个 Linear"""

    def __init__(self, D, K, p=0.1):
        super().__init__()
        self.drop, self.fc = nn.Dropout(p), nn.Linear(D, K)

    def forward(self, x, mask=None, boxes=None):
        return self.fc(self.drop(x))


class MLPHead(Head):
    """一个隐藏层 + GELU。官方实现里没有这个变体，是为了对照加的"""

    def __init__(self, D, K, h=256, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p), nn.Linear(D, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, K))

    def forward(self, x, mask=None, boxes=None):
        return self.net(x)


# --------------------------------------------------------------------------- 2. 序列上下文
class Conv1DHead(Head):
    """沿阅读顺序做一维卷积：每个词只看左右相邻的几个词。两层 k=3，扩张率 1、2，感受野 7 个词"""

    def __init__(self, D, K, h=256, k=3, dilations=(1, 2), p=0.1):
        super().__init__()
        self.drop = nn.Dropout(p)
        chans = [D] + [h] * len(dilations)
        self.convs = nn.ModuleList(
            nn.Conv1d(chans[i], chans[i + 1], k, dilation=d, padding=d * (k - 1) // 2)
            for i, d in enumerate(dilations))
        self.fc = nn.Linear(h, K)
        self.receptive_field = 1 + sum(d * (k - 1) for d in dilations)

    def forward(self, x, mask, boxes=None):
        m = mask.unsqueeze(1).float()                 # [B, 1, n]
        h = self.drop(x).transpose(1, 2) * m          # 填充位置清零，不让它冒充上下文
        for conv in self.convs:
            h = F.gelu(conv(h)) * m
        return self.fc(self.drop(h.transpose(1, 2)))


class BiLSTMHead(Head):
    """沿阅读顺序再走一遍双向 LSTM（一层，每个方向 128 维）"""

    def __init__(self, D, K, h=128, p=0.1):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.lstm = nn.LSTM(D, h, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * h, K)

    def forward(self, x, mask, boxes=None):
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(x), lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return self.fc(self.drop(out))


def rel_bucket(d, side=16):
    """有符号对数分桶：坐标差（0..1000 单位）-> 0..2*side-2 的桶号。近处分得细，远处分得粗"""
    b = torch.clamp(torch.floor(2 * torch.log2(1 + d.abs() / 2)), max=side - 1)
    return (torch.sign(d) * b + side - 1).long()


class AttnBlock(nn.Module):
    """
    Pre-LN 自注意力块。rel2d=True 时，给注意力分数加上二维相对位置偏置：
        s_ij = q_i·k_j / sqrt(d_h) + bx[bucket(cx_j - cx_i)] + by[bucket(cy_j - cy_i)]
    （LayoutLMv2 spatial-aware self-attention 的思路；这里只加 x、y 两项，不加 1D 项）
    """

    def __init__(self, d, nh, ffn, p=0.1, rel2d=False, side=16):
        super().__init__()
        self.nh, self.dh, self.side, self.rel2d = nh, d // nh, side, rel2d
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(p), nn.Linear(ffn, d))
        self.drop = nn.Dropout(p)
        if rel2d:
            self.bx = nn.Embedding(2 * side - 1, nh)      # 每个注意力头一张 dx 偏置表
            self.by = nn.Embedding(2 * side - 1, nh)      # 每个注意力头一张 dy 偏置表
            nn.init.zeros_(self.bx.weight)
            nn.init.zeros_(self.by.weight)

    def forward(self, h, mask, boxes):
        B, n, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(B, n, 3, self.nh, self.dh).permute(2, 0, 3, 1, 4)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)            # [B, nh, n, n]
        if self.rel2d:
            cx = (boxes[..., 0] + boxes[..., 2]) / 2
            cy = (boxes[..., 1] + boxes[..., 3]) / 2
            dx = cx[:, None, :] - cx[:, :, None]                    # [B, n, n]：词 j 相对词 i
            dy = cy[:, None, :] - cy[:, :, None]
            bias = self.bx(rel_bucket(dx, self.side)) + self.by(rel_bucket(dy, self.side))
            s = s + bias.permute(0, 3, 1, 2)
        s = s.masked_fill(~mask[:, None, None, :], float("-inf"))
        a = self.drop(s.softmax(-1))
        h = h + self.drop(self.o((a @ v).transpose(1, 2).reshape(B, n, d)))
        return h + self.drop(self.ff(self.ln2(h)))


class TransformerHead(Head):
    """960 -> 256 投影，再叠 1 层自注意力（4 头，FFN 512），最后逐词分类。不另加位置编码"""

    def __init__(self, D, K, d=256, nh=4, ffn=512, layers=1, p=0.1, rel2d=False):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(p), nn.Linear(D, d))
        self.blocks = nn.ModuleList(AttnBlock(d, nh, ffn, p, rel2d) for _ in range(layers))
        self.ln, self.fc = nn.LayerNorm(d), nn.Linear(d, K)

    def forward(self, x, mask, boxes):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, mask, boxes)
        return self.fc(self.ln(h))


# --------------------------------------------------------------------------- 3. 版面感知
class Layout2DHead(TransformerHead):
    """和 TransformerHead 完全一样，只多了二维相对位置偏置。两者对比 = 显式二维关系的边际贡献"""

    def __init__(self, D, K, **kw):
        super().__init__(D, K, rel2d=True, **kw)


# --------------------------------------------------------------------------- 4. CRF 解码
class CRFHead(Head):
    """线性发射分数 + 线性链 CRF。训练用负对数似然，预测用 Viterbi，置信度用前向-后向的边际概率"""

    def __init__(self, D, K, p=0.1):
        super().__init__()
        self.emit = LinearHead(D, K, p)
        self.trans = nn.Parameter(torch.zeros(K, K))     # trans[i, j]：前一个词是 i、当前词是 j 的分数
        self.start = nn.Parameter(torch.zeros(K))
        self.end = nn.Parameter(torch.zeros(K))

    def forward(self, x, mask=None, boxes=None):
        return self.emit(x)

    def _log_z(self, em, mask):                          # 前向算法：所有标签序列的总分（log 域）
        alpha = self.start + em[:, 0]
        for t in range(1, em.shape[1]):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans + em[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t:t + 1], nxt, alpha)
        return torch.logsumexp(alpha + self.end, -1)

    def _gold(self, em, y, mask):                        # 金标序列的分数
        y = y.clamp(min=0)
        s = self.start[y[:, 0]] + em[:, 0].gather(1, y[:, :1]).squeeze(1)
        for t in range(1, em.shape[1]):
            step = self.trans[y[:, t - 1], y[:, t]] + em[:, t].gather(1, y[:, t:t + 1]).squeeze(1)
            s = s + step * mask[:, t]
        last = y.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        return s + self.end[last]

    def loss(self, x, mask, y, boxes):
        em = self(x)
        return (self._log_z(em, mask) - self._gold(em, y, mask)).sum() / mask.sum()

    @torch.no_grad()
    def predict(self, x, mask, boxes):
        em = self(x)
        B, n, K = em.shape
        pred, conf = torch.zeros(B, n, dtype=torch.long), torch.zeros(B, n)
        for b in range(B):
            L, e = int(mask[b].sum()), em[b]
            score, back = self.start + e[0], []          # Viterbi
            for t in range(1, L):
                best, idx = (score.unsqueeze(1) + self.trans).max(0)
                back.append(idx)
                score = best + e[t]
            path = [int((score + self.end).argmax())]
            for idx in reversed(back):
                path.append(int(idx[path[-1]]))
            path.reverse()
            alpha = [self.start + e[0]]                  # 前向-后向 -> 每个词的边际概率
            for t in range(1, L):
                alpha.append(torch.logsumexp(alpha[-1].unsqueeze(1) + self.trans, 0) + e[t])
            beta = [self.end]
            for t in range(L - 2, -1, -1):
                beta.append(torch.logsumexp(self.trans + (e[t + 1] + beta[-1]).unsqueeze(0), 1))
            beta.reverse()
            log_z = torch.logsumexp(alpha[-1] + self.end, 0)
            marg = torch.stack([(a + bb - log_z).exp() for a, bb in zip(alpha, beta)])
            p = torch.tensor(path)
            pred[b, :L], conf[b, :L] = p, marg[torch.arange(L), p]
        return pred, conf


# --------------------------------------------------------------------------- 5. 片段 / biaffine
def flat_runs(labels):
    """扁平标签 -> 连续同标签片段 [(起点, 终点(含), 标签)]；other 不算片段"""
    runs, i = [], 0
    while i < len(labels):
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i]:
            j += 1
        if labels[i] != 0:
            runs.append((i, j, int(labels[i])))
        i = j + 1
    return runs


class BiaffineSpanHead(Head):
    """
    Yu et al. (2020) 的 biaffine 片段打分：每个候选片段 (起点 i, 长度 l) 一个 K 类分布，0 类 = 不是片段。
        score(i, j, c) = [h_s(i); 1]^T U_c [h_e(j); 1]
    训练目标从扁平标签派生：连续同标签的一段 = 一个片段。片段最长 max_len 个词。
    """

    def __init__(self, D, K, h=150, max_len=24, p=0.2, none_weight=1.0):
        super().__init__()
        self.K, self.L = K, max_len
        # 绝大多数候选都不是片段（0 类），none_weight < 1 时降低它们在损失里的权重
        self.register_buffer("cw", torch.tensor([none_weight] + [1.0] * (K - 1)))
        self.fs = nn.Sequential(nn.Dropout(p), nn.Linear(D, h), nn.GELU())    # 起点表示
        self.fe = nn.Sequential(nn.Dropout(p), nn.Linear(D, h), nn.GELU())    # 终点表示
        self.U = nn.Parameter(torch.empty(K, h + 1, h + 1))
        nn.init.normal_(self.U, std=0.02)

    def forward(self, x, mask, boxes=None):
        B, n, _ = x.shape
        one = x.new_ones(B, n, 1)
        hs = torch.cat([self.fs(x), one], -1)                       # [B, n, h+1]
        he = torch.cat([self.fe(x), one], -1)
        he_win = F.pad(he, (0, 0, 0, self.L - 1)).unfold(1, self.L, 1).permute(0, 1, 3, 2)  # [B, n, L, h+1]
        s = torch.einsum("bni,kij,bnlj->bnlk", hs, self.U, he_win)   # [B, n, L, K]
        valid = (torch.arange(n)[:, None] + torch.arange(self.L)[None, :]) < mask.sum(1)[:, None, None]
        return s, valid

    def targets(self, y, mask):
        B, n = y.shape
        tgt = torch.zeros(B, n, self.L, dtype=torch.long)
        for b in range(B):
            for i, j, c in flat_runs(y[b, :int(mask[b].sum())].tolist()):
                if j - i < self.L:
                    tgt[b, i, j - i] = c
                else:
                    tgt[b, i:j + 1] = -100                         # 超长片段：起点落在里面的候选都不参与训练
        return tgt

    def loss(self, x, mask, y, boxes):
        s, valid = self(x, mask)
        tgt = self.targets(y, mask)
        tgt[~valid] = -100
        return F.cross_entropy(s.reshape(-1, self.K), tgt.reshape(-1), weight=self.cw, ignore_index=-100)

    @torch.no_grad()
    def decode(self, x, mask):
        """贪心解码（Yu et al. 的 flat 规则）：按片段概率从高到低取，和已取片段重叠的跳过。返回每份文档的片段列表"""
        s, valid = self(x, mask)
        prob = s.softmax(-1)
        out = []
        for b in range(x.shape[0]):
            n = int(mask[b].sum())
            pmax, cls = prob[b, :n, :, 1:].max(-1)
            cand = ((pmax > prob[b, :n, :, 0]) & valid[b, :n]).nonzero().tolist()
            cand.sort(key=lambda t: -float(pmax[t[0], t[1]]))
            used, spans = torch.zeros(n, dtype=torch.bool), []
            for i, l in cand:
                if not used[i:i + l + 1].any():
                    used[i:i + l + 1] = True
                    spans.append((i, i + l, int(cls[i, l]) + 1, float(pmax[i, l])))
            out.append((spans, prob[b, :n, 0, 0]))
        return out

    @torch.no_grad()
    def predict(self, x, mask, boxes):
        B, n = mask.shape
        pred, conf = torch.zeros(B, n, dtype=torch.long), torch.zeros(B, n)
        for b, (spans, p_none) in enumerate(self.decode(x, mask)):
            k = len(p_none)
            conf[b, :k] = p_none            # 没被片段覆盖的词 = other，置信度用单词片段“不是片段”的概率
            for i, j, c, p in spans:
                pred[b, i:j + 1], conf[b, i:j + 1] = c, p     # 片段内每个词共享片段概率
        return pred, conf
