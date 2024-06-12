from typing import List

import tensorset as ts


MASK_ID = -100
DEFAULT_TENSORSET_PADDING = {
    "height_ids": 0,
    "width_ids": 0,
    "patches": 0,
    "sequence_ids": MASK_ID,
}


def pad_tensorsequence_to_length(
    sequence: List[ts.TensorSet],
    sequence_length: int,
    pad_value_dict=DEFAULT_TENSORSET_PADDING,
) -> ts.TensorSet:
    sequence = ts.cat(sequence, 0)

    # if the sequence length is too short, pads
    pad_amt = sequence_length - sequence.size(0)
    needs_pad = pad_amt > 0
    if needs_pad:
        sequence = sequence.pad(pad_amt, 0, value_dict=pad_value_dict)

    return sequence


class PairPacker:
    """
    A Packer that can pack sequences pairs into batches.
    Produces pairs of batches. Between pairs of batches,
    sequence pairs will be placed in the same batch element
    and have the same sequence id.
    """

    def __init__(
        self,
        sequence_length_x,
        sequence_length_y,
        batch_size,
        pad_value_dict=DEFAULT_TENSORSET_PADDING,
    ):
        self.sequence_length_x = sequence_length_x
        self.sequence_length_y = sequence_length_y
        self.batch_size = batch_size
        self.unpacked_batch_x: List[List[ts.TensorSet]] = [
            [] for _ in range(batch_size)
        ]
        self.unpacked_batch_y: List[List[ts.TensorSet]] = [
            [] for _ in range(batch_size)
        ]
        self.packed_batches_x: List[ts.TensorSet] = []
        self.packed_batches_y: List[ts.TensorSet] = []

        self.unpacked_metadata: List[dict[int, dict]] = [{} for _ in range(batch_size)]
        self.packed_metadata: List[List[dict[int, dict]]] = []
        self.pad_value_dict = pad_value_dict

    def append(self, x: ts.TensorSet, y: ts.TensorSet, id: int, metadata: dict = {}):
        sequence_length_x = x.size(0)
        sequence_length_y = y.size(0)
        assert sequence_length_x <= self.sequence_length_x
        assert sequence_length_y <= self.sequence_length_y
        did_fit = False
        for unpacked_sequence_x, unpacked_sequence_y, unpacked_metadata in zip(
            self.unpacked_batch_x, self.unpacked_batch_y, self.unpacked_metadata
        ):
            unpacked_sequence_length_x = sum(
                element.size(0) for element in unpacked_sequence_x
            )
            can_fit_x = (
                unpacked_sequence_length_x + sequence_length_x <= self.sequence_length_x
            )
            unpacked_sequence_length_y = sum(
                element.size(0) for element in unpacked_sequence_y
            )
            can_fit_y = (
                unpacked_sequence_length_y + sequence_length_y <= self.sequence_length_y
            )

            did_fit = can_fit_x and can_fit_y
            if did_fit:
                unpacked_sequence_x.append(x)
                unpacked_sequence_y.append(y)
                unpacked_metadata[id] = metadata
                break

        if not did_fit:
            self._flush()
            self.append(x, y, id, metadata)

    def _flush(self):
        batch_x = []
        while len(self.unpacked_batch_x) > 0:
            sequence = self.unpacked_batch_x.pop(0)
            sequence = pad_tensorsequence_to_length(
                sequence,
                self.sequence_length_x,
                self.pad_value_dict,
            )
            batch_x.append(sequence)
        self.unpacked_batch_x = [[] for _ in range(self.batch_size)]
        batch_x = ts.stack(batch_x)
        self.packed_batches_x.append(batch_x)

        batch_y = []
        while len(self.unpacked_batch_y) > 0:
            sequence = self.unpacked_batch_y.pop(0)
            sequence = pad_tensorsequence_to_length(
                sequence,
                self.sequence_length_y,
                self.pad_value_dict,
            )
            batch_y.append(sequence)
        self.unpacked_batch_y = [[] for _ in range(self.batch_size)]
        batch_y = ts.stack(batch_y)
        self.packed_batches_y.append(batch_y)

        self.packed_metadata.append(self.unpacked_metadata)
        self.unpacked_metadata = [{} for _ in range(self.batch_size)]

    def can_pop_batch(self):
        return len(self.packed_batches_x) > 0

    def pop_batch(self):
        if not self.can_pop_batch():
            return None
        return (
            self.packed_batches_x.pop(0),
            self.packed_batches_y.pop(0),
            self.packed_metadata.pop(0),
        )


class Packer:
    """
    append sequences to this packer,
    get back batches of packed, uniform length batches.
    """

    def __init__(
        self,
        sequence_length,
        batch_size,
        pad_value_dict=DEFAULT_TENSORSET_PADDING,
    ):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.unpacked_batch: List[List[ts.TensorSet]] = [[] for _ in range(batch_size)]
        self.packed_batches: List[ts.TensorSet] = []

        self.unpacked_metadata: List[dict[int, dict]] = [{} for _ in range(batch_size)]
        self.packed_metadata: List[List[dict[int, dict]]] = []

        self.pad_value_dict = pad_value_dict

    def append(self, sequence: ts.TensorSet, id: int, metadata: dict = {}):
        """
        adds sequence to the first cached sequence with enough remaining space
        """
        sequence_length = sequence.size(0)
        assert sequence_length <= self.sequence_length
        did_fit = False
        for unpacked_sequence, unpacked_metadata in zip(
            self.unpacked_batch, self.unpacked_metadata
        ):
            unpacked_sequence_length = sum(x.size(0) for x in unpacked_sequence)
            did_fit = unpacked_sequence_length + sequence_length <= self.sequence_length
            if did_fit:
                unpacked_sequence.append(sequence)
                unpacked_metadata[id] = metadata
                break

        if not did_fit:
            self._flush()
            self.append(sequence, id, metadata)

    def _flush(self):
        batch = []
        while len(self.unpacked_batch) > 0:
            sequence = self.unpacked_batch.pop(0)
            sequence = pad_tensorsequence_to_length(
                sequence,
                self.sequence_length,
                self.pad_value_dict,
            )
            batch.append(sequence)

        batch = ts.stack(batch)

        self.packed_batches.append(batch)
        self.unpacked_batch = [[] for _ in range(self.batch_size)]

        self.packed_metadata.append(self.unpacked_metadata)
        self.unpacked_metadata = [{} for _ in range(self.batch_size)]

    def can_pop_batch(self):
        return len(self.packed_batches) > 0

    def pop_batch(self):
        if not self.can_pop_batch():
            return None
        return self.packed_batches.pop(0), self.packed_metadata.pop(0)
