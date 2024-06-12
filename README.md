# cabbage-patch

This library provides utilities for making datasets that patch images at different resolutions and pack them into batches.
For more details on sequence packing for images, checkout [NaViT](https://arxiv.org/abs/2307.06304). 


cabbage-patch uses the default configuration described by NaViT. Each image is resized to a random side length using 
a uniform distribution. Image patches can optionally be dropped randomly. 
The authors of NaViT showed that this configuration works well for CLIP and supervised classification. 


cabbage-patch uses webdataset as the backend dataset. 
Check `example_cc3m.py` for an example of how to use `CabbageDataset`.


# Packing and batch size

The packing algorithm is applied as a map operation on the webdataset. That means that if you use a dataloader,
each worker will have their own instance of the packing algorithm. You may still use a dataloader to collate
the result of multiple CabbageDatasets. There are then two batch sizes to control. `packer_batch_size` is used per worker,
 and the dataloader batch size determines the number of worker results that will get collated together. The overall
batch size is the product of the two.

If you are using a very large overall batch size it is faster to use a lower `packer_batch_size`. But lowering
the `packer_batch_size` will cause the packer to use more padding.


# TensorSet

I use `TensorSet` to make it easier to deal with multiple related tensor sequences. 
`TensorSet` is basically a dictionary of tensors.


# Padding Rules

* `height_ids` and `width_ids` are padded with zeros
* `patches` is padded with zeros
* `sequence_ids` is padded with `-100`

# Example outputs
(The red grid lines have been added to show the patch boundaries)

![batch01-sequence0001-id0058-a man wades through a water logged road during rains ](https://github.com/theAdamColton/cabbage-patch/assets/72479734/855aa838-5a32-4a47-bbda-21cfadcdd99b)

![batch01-sequence0026-id0095-according to person we need to run out and get some plaid ](https://github.com/theAdamColton/cabbage-patch/assets/72479734/8285fd97-6d7a-4407-af91-18417eedfe82)
![batch01-sequence0011-id0074-falcon unique design of an old motorcycle](https://github.com/theAdamColton/cabbage-patch/assets/72479734/59395d7e-190d-4caf-89da-207213137acf)

