import torch
import numpy as np
import webdataset as wds
import tensorset as ts

from cabbage_patch.packer import DEFAULT_TENSORSET_PADDING, Packer, PairPacker
from cabbage_patch.transforms import PatchImageRow, RandomResize, TokenDropper


def verify_patches(sample, name="patches"):
    assert name in sample
    patches = sample.get(name)
    assert isinstance(patches, ts.TensorSet)


def _addin_ids(x: ts.TensorSet, id):
    x_length = x.size(0)
    ids = torch.full(
        size=(x_length,),
        fill_value=id,
        dtype=torch.long,
        device=x.all_columns[0].device,
    )
    x.named_columns["sequence_ids"] = ids


def _packed(
    data,
    sequence_length=256,
    batch_size=16,
    pad_value_dict=DEFAULT_TENSORSET_PADDING,
):
    """"""
    packer = Packer(sequence_length, batch_size, pad_value_dict)
    id = 0
    for sample in data:
        verify_patches(sample)
        patches = sample.pop("patches")
        _addin_ids(patches, id)
        packer.append(patches, id, sample)
        if packer.can_pop_batch():
            patches, metadata = packer.pop_batch()
            yield {"patches": patches, "metadata": metadata}

        id += 1


packed = wds.pipelinefilter(_packed)


def _packed_x_y(
    data,
    sequence_length_x=256,
    sequence_length_y=256,
    batch_size=16,
    pad_value_dict=DEFAULT_TENSORSET_PADDING,
):
    """
    Packs x,y pairs into two batches
    an x,y sample will have x and y put in the same sequence of each batch
    """
    packer = PairPacker(
        sequence_length_x, sequence_length_y, batch_size, pad_value_dict
    )
    id = 0
    for sample in data:
        verify_patches(sample, "x_patches")
        verify_patches(sample, "y_patches")
        x, y = sample.pop("x_patches"), sample.pop("y_patches")
        _addin_ids(x, id)
        _addin_ids(y, id)
        packer.append(x, y, id, sample)
        if packer.can_pop_batch():
            x, y, metadata = packer.pop_batch()
            yield {"x_patches": x, "y_patches": y, "metadata": metadata}

        id += 1


packed_x_y = wds.pipelinefilter(_packed_x_y)


def collation_fn(samples, combine_tensors=True, combine_scalars=True):
    batched = list(zip(*samples))
    result = []
    for b in batched:
        if isinstance(b[0], ts.TensorSet):
            b = ts.cat(b, 0)
        elif isinstance(b[0], list):
            # list summation
            b = [x for y in b for x in y]
        else:
            b = list(b)
        result.append(b)
    return result


class CabbageDataset(wds.WebDataset):
    def batched(self, batchsize, collation_fn=collation_fn, partial=True):
        return super().batched(batchsize, collation_fn, partial)

    def packed(self, *args, **kwargs):
        return self.compose(packed(*args, **kwargs))

    def packed_x_y(self, *args, **kwargs):
        return self.compose(packed_x_y(*args, **kwargs))

    def patch_n_pack(
        self,
        patch_size=16,
        token_drop_chance=0.0,
        sequence_length=256,
        resize_min=64,
        resize_max=256,
        resize_max_res=1024,
        batch_size=16,
        rng=None,
    ):
        return (
            self.map_dict(
                pixel_values=RandomResize(
                    resize_min, resize_max, resize_max_res, patch_size, rng=rng
                )
            )
            .map(PatchImageRow(patch_size))
            .map_dict(patches=TokenDropper(token_drop_chance, sequence_length, rng=rng))
            .packed(sequence_length, batch_size)
        )
