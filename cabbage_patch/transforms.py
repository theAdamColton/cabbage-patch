import math
from torchvision import transforms
import numpy as np
from typing import Optional
from torch import nn
import torch
import tensorset as ts

from cabbage_patch.utils import get_rng, random_a_b


class PatchImageRow(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, row):
        pixel_values = row.pop("pixel_values")
        _, h, w = pixel_values.shape
        sequence = patch_image(pixel_values, self.patch_size)
        row["patches"] = sequence
        row["height"] = h
        row["width"] = w
        return row


def patch_image(x: torch.Tensor, patch_size: int):
    """
    x: An image of shape (C H W)

    returns a TensorSet:
        patches: (S Z)
          flattened sequence of patches
        height_ids (S)
        width_ids (S)
    """
    c, h, w = x.shape
    assert h % patch_size == 0, h
    assert w % patch_size == 0, w
    nph = h // patch_size
    npw = w // patch_size

    x = torch.reshape(x, (c, nph, patch_size, npw, patch_size))
    x = x.permute(1, 3, 0, 2, 4)
    x = x.reshape(nph * npw, c * patch_size * patch_size)

    device = x.device
    height_ids, width_ids = torch.meshgrid(
        torch.arange(nph, device=device),
        torch.arange(npw, device=device),
        indexing="ij",
    )
    height_ids = height_ids.flatten()
    width_ids = width_ids.flatten()

    return ts.TensorSet(patches=x, height_ids=height_ids, width_ids=width_ids)


class RandomResize(nn.Module):
    """
    Randomly resizes images by side length, using a uniform distribution.
    The height and width of the final image will both be divisible by `multiple_of`
    """

    def __init__(self, min_res=64, max_res=384, multiple_of=1, rng=None):
        super().__init__()
        assert min_res % multiple_of == 0
        assert max_res % multiple_of == 0

        self.min_res = min_res
        self.max_res = max_res
        self.multiple_of = multiple_of
        if rng is None:
            rng = get_rng()
        self.rng = rng

    def forward(self, pixel_values):
        _, h, w = pixel_values.shape
        # the new resolution is u
        u = random_a_b(self.min_res, self.max_res, self.rng)
        size = (h * w) ** 0.5
        scale = u / size
        new_h = scale * h
        new_h = int(self.multiple_of * math.ceil(new_h / self.multiple_of))
        new_w = scale * w
        new_w = int(self.multiple_of * math.ceil(new_w / self.multiple_of))
        rz = transforms.Resize((new_h, new_w))
        return rz(pixel_values)


class TokenDropper(nn.Module):
    """
    Drops tokens randomly
    Each token has a `drop_chance` of being dropped, unless `max_sequence_length` is specified.
    In the case that the sequence length is larger than the `max_sequence_length`,
    tokens will be randomly dropped until the sequence length is equal to `max_sequence_length`.
    """

    def __init__(
        self,
        drop_chance=0.25,
        max_sequence_length: Optional[int] = None,
        rng=None,
    ):
        super().__init__()
        self.drop_chance = drop_chance
        self.max_sequence_length = max_sequence_length
        if rng is None:
            rng = get_rng()
        self.torch_rng = torch.Generator().manual_seed(
            rng.randint(-999999999, 999999999)
        )

    def forward(self, sequence: ts.TensorSet):
        sequence_length = sequence.size(0)
        sample = torch.rand(
            size=(sequence_length,),
            device=sequence.all_columns[0].device,
            generator=self.torch_rng,
        )
        mask = sample > self.drop_chance

        if (
            self.max_sequence_length is not None
            and sequence_length > self.max_sequence_length
        ):
            quantile = sample[mask].quantile(
                1 - self.max_sequence_length / sequence_length
            )
            mask = mask & (sample > quantile)

        sequence = sequence.iloc[mask]
        return sequence


class ToTorchRGB8(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pixel_values):
        assert isinstance(pixel_values, np.ndarray)

        pixel_values = pixel_values.transpose(2, 0, 1)
        pixel_values = torch.from_numpy(
            pixel_values
        )  # this will print a warning for unwritable array
        return pixel_values
