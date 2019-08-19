from sklearn.metrics import roc_curve, auc, f1_score
from torch import nn
import torch


def roc(labels, scores):
    """Compute ROC curve and ROC area for each class"""
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    return roc_auc


def f1_score_calc(labels, scores, threshold):
    scores[scores >= threshold] = 1
    scores[scores < threshold] = 0

    return f1_score(labels, scores)


if __name__ == '__main__':
    import torch

    l = torch.LongTensor([0, 1, 1, 0, 0]).cuda()
    s = torch.FloatTensor([0.9, 0.8, 0.8, 0.1, 0.1]).cuda()
    th = torch.FloatTensor([0.99, 0.7, 0.5, 0.5, 0.2]).cuda()
    print(l.size(), s.size())
    a = f1_score_calc(l, s, th)
    print(a)


