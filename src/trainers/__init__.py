""" Training package """

from trainers.xla_class_trainer import XLAClassTrainer
from trainers.xla_arc_class_trainer import XLAArcClassTrainer


TRAINER_DICT = {
    "XLAClassTrainer": XLAClassTrainer,
    "XLAArcClassTrainer": XLAArcClassTrainer
}
