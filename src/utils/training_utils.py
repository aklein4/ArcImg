import torch
import torch.nn as nn
import torch.nn.functional as F


def loss(preds, targets):
    return F.cross_entropy(preds, targets)


@torch.no_grad()
def acc(preds, targets):
    return (preds.argmax(-1) == targets).float().mean()


@torch.no_grad()
def pcorr(preds, targets):
    logp = -F.cross_entropy(
        preds, targets,
        reduction='none'
    )
    p = torch.exp(logp)

    return p.mean()


@torch.no_grad()
def top5_acc(preds, targets):
    preds = F.log_softmax(preds, dim=-1)

    logp = -F.cross_entropy(
        preds, targets,
        reduction='none'
    )

    rank = (preds >= logp.unsqueeze(-1)).float().sum(-1)
    return (rank <= 5).float().mean()


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
