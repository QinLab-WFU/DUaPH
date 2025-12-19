import torch
import torch.nn as nn
from torch.nn.functional import normalize

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, feats, labels, feat2=None):
        assert feats.size(0) == labels.size(0)
        batch_size = feats.size(0)
        if feat2 is None:
            sim_mat = normalize(torch.matmul(feats, torch.t(feats)))
        else:
            sim_mat = normalize(torch.matmul(feats, torch.t(feat2)))

        labels = labels @ labels.t() > 0

        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - 1e-5]

            neg_pair_ = sim_mat[i][(labels[i] == False)]

            if torch.numel(pos_pair_) == 0 or torch.numel(neg_pair_) == 0:
                continue

            neg_pair = neg_pair_[neg_pair_ + 0.1 > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - 0.1 < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 0.5 * torch.log(
                1 + torch.sum(torch.exp(-2.0 * (pos_pair - 0.1))))
            neg_loss = 0.025 * torch.log(
                1 + torch.sum(torch.exp(40.0 * (neg_pair - 0.1))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss