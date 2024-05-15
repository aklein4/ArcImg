import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict
from  utils.training_utils import (
    loss, acc, pcorr, top5_acc,
    arc_loss, arc_acc, arc_pcorr
)

from utils.logging_utils import log_print


class XLAArcClassTrainer(BaseXLATrainer):

    def train_step(self, model, x, y):
        logits, true, fake = model(x, y)
        log_print("Predictions!")

        results = DotDict(
            class_loss=loss(logits, y),
            class_acc=acc(logits, y),
            class_top5_acc=top5_acc(logits, y),
            class_pcorr=pcorr(logits, y),

            arc_loss=arc_loss(true, fake),
            arc_acc=arc_acc(true, fake),
            arc_pcorr=arc_pcorr(true, fake),
        )
        results.loss = results.class_loss + self.w_arc * results.arc_loss

        return results
