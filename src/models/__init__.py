""" Models """

from transformers import ViTConfig, ViTForImageClassification
from models.arc_vit import ArcVitForImageClassification


CONFIG_DICT = {
    "vit": ViTConfig,
    "arc_vit": ViTConfig,
}

MODEL_DICT = {
    "vit": ViTForImageClassification,
    "arc_vit": ArcVitForImageClassification
}