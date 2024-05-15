import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict
from  utils.training_utils import loss, acc, pcorr, top5_acc


class XLAClassTrainer(BaseXLATrainer):

    def train_step(self, model, x, y):
        logits = model(x).logits

        results = DotDict(
            class_loss=loss(logits, y),
            class_acc=acc(logits, y),
            class_top5_acc=top5_acc(logits, y),
            class_pcorr=pcorr(logits, y)
        )
        results.loss = results.class_loss

        return results
