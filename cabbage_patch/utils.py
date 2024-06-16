import torch
import einx


def random_a_b(a, b, torch_rng=None):
    u = torch.rand((1,), generator=torch_rng)
    return u * (b - a) + a


def unpatch(patches, height_ids, width_ids, patch_size: int, image_channels: int):
    """
    patches: (S Z)

    Takes a sequence of patches, which correspond to one image
    Uses the height ids and width ids to place the patches back in a reconstructed image
    """
    patches = einx.rearrange(
        "... S (C PH PW) -> ... S C PH PW",
        patches,
        C=image_channels,
        PH=patch_size,
        PW=patch_size,
    )
    min_h = height_ids.min() * patch_size
    min_w = width_ids.min() * patch_size
    max_h = height_ids.max() * patch_size + patch_size
    max_w = width_ids.max() * patch_size + patch_size
    h = max_h - min_h
    w = max_w - min_w
    image = torch.zeros(
        image_channels, h, w, device=patches.device, dtype=patches.dtype
    )
    for idxH, idxW, patch in zip(height_ids, width_ids, patches):
        i, j = idxH * patch_size, idxW * patch_size
        i = i - min_h
        j = j - min_w
        image[
            :,
            i : i + patch_size,
            j : j + patch_size,
        ] = patch

    return image
