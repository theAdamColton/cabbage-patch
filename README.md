# cabbage-patch

This library provides utilities for making datasets that patch images at different resolutions and pack them into batches.

It uses webdataset as the backend dataset. 
Check `example_imagenet1k.py` for an example of how to use `CabbageDataset` with imagenet1k.


# Packing and batch size

The packing algorithm is applied as a map operation on the webdataset. That means that if you use a dataloader,
each worker will have their own instance of the packing algorithm. You may still use a dataloader to collate
the result of multiple CabbageDatasets. There are then two batch sizes to control. `packer_batch_size` is used per worker,
 and the dataloader batch size determines the number of worker results that will get collated together. The overall
batch size is the product of the two.

If you are using a very large overall batch size it would be faster to set a lower `packer_batch_size`. But lowering
the `packer_batch_size` will cause the packer to use more padding.


# TensorSet

I use `TensorSet` to make it easier to deal with multiple related tensor sequences. 
`TensorSet` is basically a dictionary of tensors.


# Padding Rules

* `height_ids` and `width_ids` are padded with zeros
* `patches` is padded with zeros
* `sequence_ids` is padded with `-100`
