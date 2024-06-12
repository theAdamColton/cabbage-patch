"""
This uses seperately lengthed context and target sequences.

This type of sequence is used in DINOv2 and IJEPA
"""

from torch import nn
from huggingface_hub import hf_hub_download
import os
import torch
from torchvision.io import write_jpeg
from torch.utils.data import DataLoader
import jsonargparse
import random

import cabbage_patch
from cabbage_patch import transforms
from cabbage_patch.utils import unpatch


class ContextTargetMasking(nn.Module):
    """
    Very simple random context-target
    """

    def __init__(self):
        super().__init__()

    def forward(self, row):
        patches = row.pop("patches")
        chance = 0.1
        context_mask = torch.rand(patches.size(0)) < chance
        row["x_patches"] = patches.iloc[context_mask]
        row["y_patches"] = patches

        return row


def main(
    patch_size: int = 16,
    token_drop_chance: float = 0.25,
    sequence_length_x: int = 64,
    sequence_length_y: int = 256,
    resize_min_res: int = 64,
    resize_max_res: int = 384,
    packer_batch_size: int = 16,
    final_batch_size: int = 32,
    seed: int = 42,
):
    rng = random.Random(seed)

    # first downloads a sample webdataset tar file,
    # this uses the huggingface hub.
    # any tar file containing an image dataset would work for this
    os.makedirs("tars", exist_ok=True)
    if not os.path.exists("tars/cc3m-train-0000.tar"):
        hf_hub_download(
            "pixparse/cc3m-wds",
            repo_type="dataset",
            filename="cc3m-train-0000.tar",
            cache_dir=None,
            local_dir="tars/",
            local_dir_use_symlinks=False,
        )

    # replace the path with your own tar file! It must have a 'jpg' column and 'txt' column
    # for this example.
    dataset = cabbage_patch.CabbageDataset("tars/cc3m-train-0000.tar")

    assert final_batch_size % packer_batch_size == 0

    dataset = (
        dataset.decode("torchrgb8")
        .rename(pixel_values="jpg")
        .map_dict(
            pixel_values=transforms.RandomResize(
                resize_min_res, resize_max_res, patch_size, rng
            )
        )
        .map(transforms.PatchImageRow(patch_size))
        .map(ContextTargetMasking())
        .map_dict(
            x_patches=transforms.TokenDropper(
                token_drop_chance, sequence_length_x, rng
            ),
            y_patches=transforms.TokenDropper(
                token_drop_chance, sequence_length_y, rng
            ),
        )
        .packed_x_y(sequence_length_x, sequence_length_y, packer_batch_size)
        .to_tuple("x_patches", "y_patches", "metadata")
        .batched(final_batch_size // packer_batch_size)
    )

    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    os.makedirs("output/", exist_ok=True)
    os.makedirs("output/ijepa/", exist_ok=True)

    p_padding_context = []
    p_padding_target = []

    n_images = 0
    for batch_i, batch in enumerate(dataloader):
        context, target, metadata = batch

        b = context.size(0)
        for sequence_i in range(b):
            context_seq = context.iloc[sequence_i]
            context_padding = context_seq["sequence_ids"] == -100
            p_padding_context.append(context_padding.float().mean())

            target_seq = target.iloc[sequence_i]
            target_padding = target_seq["sequence_ids"] == -100
            p_padding_target.append(target_padding.float().mean())

            for sequence_id in torch.unique(context_seq["sequence_ids"]):
                if sequence_id == -100:
                    continue
                context_mask = context_seq["sequence_ids"] == sequence_id
                context_image_seq = context_seq.iloc[context_mask]
                context_image = unpatch(
                    context_image_seq["patches"],
                    context_image_seq["height_ids"],
                    context_image_seq["width_ids"],
                    patch_size,
                    3,
                )

                save_path = f"output/ijepa/batch{batch_i:02}-sequence{sequence_i:04}-id{sequence_id:04}-context.jpg"
                write_jpeg(context_image, save_path)

                target_mask = target_seq["sequence_ids"] == sequence_id
                target_image_seq = target_seq.iloc[target_mask]
                target_image = unpatch(
                    target_image_seq["patches"],
                    target_image_seq["height_ids"],
                    target_image_seq["width_ids"],
                    patch_size,
                    3,
                )

                save_path = f"output/ijepa/batch{batch_i:02}-sequence{sequence_i:04}-id{sequence_id:04}-target.jpg"
                write_jpeg(target_image, save_path)
                print(save_path)

                n_images += 1

    print("percent padding ", torch.stack(p_padding_context).mean().item() * 100, "%")
    print("number of images per sequence", n_images / final_batch_size)


if __name__ == "__main__":
    jsonargparse.CLI(main)
