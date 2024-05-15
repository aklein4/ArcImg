
import torch
import torchvision.transforms as transforms

import torch_xla.distributed.parallel_loader as pl

import os
import datasets

import utils.constants as constants


IMAGENET_URL = "https://huggingface.co/datasets/imagenet-1k"

LABEL_FILE = os.path.join(constants.BASE_PATH, "loaders", "data", "synset_words.txt")


class ImageCollator:

    def __init__(self, size):
        self.size = size
    
        self.comp = transforms.Compose([
            transforms.transforms.Resize(self.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        self.label_map = {}
        ind = 0
        with open(LABEL_FILE, "r") as f:
            for line in f:
                self.label_map[line.strip().split(" ")[0]] = ind
                ind += 1


    def __call__(self, x):

        imgs = []
        labels = []
        for elem in x:

            img = transforms.functional.center_crop(
                elem['jpeg'], 
                min(elem['jpeg'].size)
            )
            img = self.comp(img)
            img = (2 * img) - 1
            imgs.append(img)

            num = elem['__key__'].split("_")[-1]
            labels.append(self.label_map[num])
        
        return torch.stack(imgs), torch.tensor(labels).long()


def _get_data_files():

    data_files = {}
    for split in ["train", "val", "test"]:

        data_files[split] = f"{IMAGENET_URL}/resolve/main/data/{split}*"
    
    return data_files


def get_imagenet_loader(
    split: str,
    bs: int,
    mini_bs: int,
    size
):
    
    # prepare batch sizes
    total_mini_bs = mini_bs * constants.NUM_XLA_DEVICES()
    if bs % total_mini_bs != 0:
        raise ValueError(f"Batch size {bs} not divisible by total mini batch size {total_mini_bs}")
    sample_size = mini_bs * (bs // total_mini_bs)

    # load dataset
    dataset = datasets.load_dataset(
        "webdataset",
        data_files=_get_data_files(),
        split=split, streaming=True
    )

    # wrap in loader with collator
    collator = ImageCollator(size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=sample_size,
        collate_fn=collator,
        drop_last=True
    )

    # wrap with xla loader
    wrapper_type = pl.MpDeviceLoader if constants.NUM_XLA_DEVICES() > 1 else pl.ParallelLoader
    xm_loader = wrapper_type(loader, device=constants.XLA_DEVICE())

    return xm_loader
