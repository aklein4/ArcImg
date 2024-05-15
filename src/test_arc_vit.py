import torch

from models import CONFIG_DICT, MODEL_DICT
from utils.config_utils import load_model_config


CLASS_CONFIG = "base-32-class"
ARC_CONFIG = "base-32-arc"


def main():

    print("loading class...")
    class_config = load_model_config(CLASS_CONFIG)
    class_model = MODEL_DICT[class_config["model_type"]](
        CONFIG_DICT[class_config["model_type"]](**class_config)
    )

    print("loading arc...")
    arc_config = load_model_config(ARC_CONFIG)
    arc_model = MODEL_DICT[arc_config["model_type"]](
        CONFIG_DICT[arc_config["model_type"]](**arc_config)
    )

    print("copying weights...")
    arc_model.load_state_dict(class_model.state_dict(), strict=False)
    
    arc_model.eval()
    class_model.eval()

    pixels = torch.randn(1, 3, 224, 224)
    labels = torch.zeros(1, dtype=torch.long)

    class_logits = class_model(pixels).logits
    arc_logits = arc_model(pixels, labels, debug=True)[0]

    diff = (class_logits - arc_logits).abs().max()
    print(f"max diff: {diff}")


if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    torch.enable_grad(False)

    main()
