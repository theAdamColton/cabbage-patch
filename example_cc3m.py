from huggingface_hub import hf_hub_download
import os
import torch
from torchvision.io import write_jpeg
from torch.utils.data import DataLoader
import jsonargparse
import random

import cabbage_patch
from cabbage_patch.utils import unpatch


def draw_patch_grid(image, patch_size=16):
    _, h, w = image.shape
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    mask = (rows % patch_size == 0) | (cols % patch_size == 0)
    image[0][mask] = 255
    image[1][mask] = 0
    image[2][mask] = 0


def main(
    patch_size: int = 16,
    token_drop_chance: float = 0.25,
    sequence_length: int = 256,
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
        .patch_n_pack(
            patch_size=patch_size,
            token_drop_chance=token_drop_chance,
            sequence_length=sequence_length,
            resize_min_res=resize_min_res,
            resize_max_res=resize_max_res,
            batch_size=packer_batch_size,
            rng=rng,
        )
        .to_tuple("patches", "metadata")
        .batched(final_batch_size // packer_batch_size)
    )

    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    os.makedirs("output/", exist_ok=True)
    os.makedirs("output/cc3m", exist_ok=True)

    p_padding = []

    n_images = 0
    for batch_i, batch in enumerate(dataloader):
        patches, metadata = batch
        for sequence_i, sequence in enumerate(patches.iloc):
            padding = sequence["sequence_ids"] == -100
            p_padding.append(padding.float().mean())

            for sequence_id in torch.unique(sequence["sequence_ids"]):
                if sequence_id == -100:
                    continue
                mask = sequence["sequence_ids"] == sequence_id
                image_sequence = sequence.iloc[mask]
                image = unpatch(
                    image_sequence["patches"],
                    image_sequence["height_ids"],
                    image_sequence["width_ids"],
                    patch_size,
                    3,
                )

                image_caption = metadata[sequence_i][int(sequence_id)]["txt"]

                image_caption = "".join(
                    c for c in image_caption if c.isalnum() or c == " "
                )
                image_caption = image_caption[:100]

                draw_patch_grid(image)

                save_path = f"output/batch{batch_i:02}-sequence{sequence_i:04}-id{sequence_id:04}-{image_caption}.jpg"
                write_jpeg(image, save_path)
                print(save_path)
                n_images += 1

    print("percent padding ", torch.stack(p_padding).mean().item() * 100, "%")
    print("number of images per sequence", n_images / final_batch_size)


if __name__ == "__main__":
    jsonargparse.CLI(main)
