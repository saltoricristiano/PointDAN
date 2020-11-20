import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


# fine
def info_nce(logits, pos_mask, logit_mask=None, temperature=0.2):
    b = logits.size(0)
    device = logits.device
    logits = logits / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    # all matches excluding the main diagonal
    if logit_mask is None:
        logit_mask = ~torch.eye(b, dtype=bool, device=device)
    div = torch.sum(torch.exp(logits) * logit_mask, dim=1, keepdim=True)
    log_prob = torch.log(torch.exp(logits) / div)
    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)
    # filter where there are no positives
    indexes = pos_mask.sum(1) > 0
    pos_mask = pos_mask[indexes]
    mean_log_prob_pos = mean_log_prob_pos[indexes]
    mean_log_prob_pos /= pos_mask.sum(1)
    # loss
    if len(mean_log_prob_pos):
        loss = -mean_log_prob_pos.mean()
    else:
        loss = torch.tensor(0.0, device=device)
    return loss

class InterDomainBasedNCE(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature
    def forward(self, xs, xt, ys, yt, selection=None):
        bs = xs.size(0)
        device = xs.device
        x = torch.cat((xs, xt), dim=0)
        y = torch.cat((ys, yt), dim=0)
        if selection is not None:
            x = x[selection]
            y = y[selection]
        b = x.size(0)
        x = F.normalize(x, dim=1)
        logits = torch.mm(x, x.t())
        labels_matrix = y.reshape(1, -1).repeat(b, 1)
        pos_mask = labels_matrix == labels_matrix.t()
        pos_mask.fill_diagonal_(False)
        pos_mask[:bs, :bs] = False
        pos_mask[bs:, bs:] = False
        logit_mask = torch.ones_like(logits, dtype=bool, device=xs.device)
        logit_mask.fill_diagonal_(False)
        logit_mask[:bs, :bs] = False
        logit_mask[bs:, bs:] = False
        if (logit_mask == 0).all():
            loss = torch.tensor(0.0, device=device)
        else:
            loss = info_nce(
                logits, pos_mask, logit_mask=logit_mask, temperature=self.temperature
            )
        return loss

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


def compute_filter_statistics(y_target, y_target_est, selection):
        acc_pseudo_label_passed = compute_acc(
            y_target[selection], y_target_est[selection]
        )
        acc_pseudo_label_filtered = compute_acc(
            y_target[~selection], y_target_est[~selection]
        )
        acc_pseudo_label_total = compute_acc(y_target, y_target_est)
        # entropies
        dist_per_pseudo_passed = compute_dist_per_class(
            y_target[selection], y_target_est[selection]
        )
        dist_per_pseudo_filtered = compute_dist_per_class(
            y_target[~selection], y_target_est[~selection]
        )
        dist_per_pseudo_total = compute_dist_per_class(y_target, y_target_est)
        hist_passed = histogram(y_target_est[selection])
        hist_filtered = histogram(y_target_est[~selection])
        hist_total = histogram(y_target_est)
        data = {
            "hist_passed": hist_passed,
            "hist_filtered": hist_filtered,
            "hist_total": hist_total,
            "acc_pseudo_label_passed": acc_pseudo_label_passed,
            "acc_pseudo_label_filtered": acc_pseudo_label_filtered,
            "acc_pseudo_label_total": acc_pseudo_label_total,
            "dist_per_pseudo_passed": dist_per_pseudo_passed,
            "dist_per_pseudo_filtered": dist_per_pseudo_filtered,
            "dist_per_pseudo_total": dist_per_pseudo_total,
        }
        return data

def histogram(data):
        hist = torch.zeros(10, device=data.device)
        for elem in data:
            hist[elem] += 1
        return hist
def compute_acc(y, y_hat):
    # compute general acc
    acc = y == y_hat
    if len(acc) > 0:
        acc = acc.sum().detach()/acc.size(0)
    else:
        acc = torch.tensor(0.0, device='cuda')
    return acc
def compute_dist_per_class(y, y_hat):
    # compute acc per class
    dist_per_pseudo = {}
    for c in range(10):
        index = y_hat == c
        filtered_y = y[index]
        if len(filtered_y):
            ind, count = torch.unique(filtered_y, return_counts=True)
            count = count.float()
            dist = torch.zeros(10, device='cuda')
            dist[ind] = count
            dist_per_pseudo[c] = dist
        else:
            dist_per_pseudo[c] = None
    return dist_per_pseudo

def compute_entropy(x):
    epsilon = 1e-5
    H = -x * torch.log(x + epsilon)
    H = torch.sum(H, dim=1)
    return H


class PseudoLabelDistribution(object):
    def on_train_epoch_start(self):
        # histograms
        self.hist_passed = []
        self.hist_filtered = []
        self.hist_total = []
        # distributions
        self.dist_per_pseudo_passed = []
        self.dist_per_pseudo_filtered = []
        self.dist_per_pseudo_total = []

    def on_train_batch_end(self, data):
            hist_passed = data["hist_passed"].cpu()
            hist_filtered = data["hist_filtered"].cpu()
            hist_total = data["hist_total"].cpu()
            self.hist_passed.append(hist_passed)
            self.hist_filtered.append(hist_filtered)
            self.hist_total.append(hist_total)
            dist_per_pseudo_passed = data["dist_per_pseudo_passed"]
            dist_per_pseudo_filtered = data["dist_per_pseudo_filtered"]
            dist_per_pseudo_total = data["dist_per_pseudo_total"]
            self.dist_per_pseudo_passed.append(dist_per_pseudo_passed)
            self.dist_per_pseudo_filtered.append(dist_per_pseudo_filtered)
            self.dist_per_pseudo_total.append(dist_per_pseudo_total)

    def on_train_epoch_end(self):
        if len(self.hist_total):
            hist_total = torch.stack(self.hist_total)
            dist_total = hist_total.sum(0)
            ylim = (0, dist_total.max())
            for name, hist, dist_per_class in zip(
                ["passed", "filtered", "total"],
                [self.hist_passed, self.hist_filtered, self.hist_total],
                [
                    self.dist_per_pseudo_passed,
                    self.dist_per_pseudo_filtered,
                    self.dist_per_pseudo_total,
                ],
            ):
                hist = torch.stack(hist)
                dist = hist.sum(0)
                # get distributions per class
                sum_dist_per_class = defaultdict(list)
                for v in dist_per_class:
                    for c, d in v.items():
                        if d is not None:
                            sum_dist_per_class[c].append(d.cpu())
                temp = {
                    c: torch.stack(d).sum(dim=0)
                    for c, d in sum_dist_per_class.items()
                }
                sum_dist_per_class = {c: None for c in range(10)}
                for c, v in temp.items():
                    sum_dist_per_class[c] = v
                sns.set(rc={"figure.figsize": (30, 30), "font.size": 15})
                sns.set(font_scale=2)
                ax = sns.barplot(data=dist)
                ax.set(ylim=ylim)
                xticks = []
                for c, d in sorted(sum_dist_per_class.items()):
                    if d is not None:
                        acc = round((d[c] / d.sum()).item(), 3)
                        h = round(
                            compute_entropy((d / d.sum()).unsqueeze(0)).item(), 3
                        )
                        d = str(d.int().tolist())
                    else:
                        acc = -1
                        h = -1
                    xticks.append(f"acc={acc},H={h},{d}")
                ax.set_xticklabels(
                    xticks,
                    rotation=20,
                    horizontalalignment="right",
                    fontsize="medium",
                )
                plt.tight_layout()
                wandb.log({f"pseudo_labels_dist_{name}": wandb.Image(ax)})
                plt.close()