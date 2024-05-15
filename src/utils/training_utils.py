import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logging_utils import log_print

def loss(preds, targets):
    out = F.cross_entropy(preds, targets)
    log_print("Loss!")
    return out



@torch.no_grad()
def acc(preds, targets):
    out = (preds.argmax(-1) == targets).float().mean()
    log_print("Accuracy!")
    return out


@torch.no_grad()
def pcorr(preds, targets):
    logp = -F.cross_entropy(
        preds, targets,
        reduction='none'
    )
    p = torch.exp(logp)

    out = p.mean()
    log_print("PCorr!")


@torch.no_grad()
def top5_acc(preds, targets):
    preds = F.log_softmax(preds, dim=-1)

    logp = -F.cross_entropy(
        preds, targets,
        reduction='none'
    )

    rank = (preds >= logp.unsqueeze(-1)).float().sum(-1)
    out = (rank <= 5).mean()
    log_print("Top5 Accuracy!")
    return out


def arc_loss(true_preds, fake_preds):
    return (
        F.logsigmoid(-true_preds).mean() +
        F.logsigmoid(fake_preds).mean()
    )


@torch.no_grad()
def arc_acc(true_preds, fake_preds):
    return (
        (true_preds < 0).float().mean() +
        (fake_preds >= 0).float().mean()
    ) / 2


@torch.no_grad()
def arc_pcorr(true_preds, fake_preds):
    return (
        torch.sigmoid(-true_preds).mean() +
        torch.sigmoid(fake_preds).mean()
    ) / 2