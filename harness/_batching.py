#!/usr/bin/env python3
"""Shared helpers for fixed-composition batched greedy generation.

Pure/stdlib functions only, except :func:`padded_batch`, which imports torch
lazily. Importing this module therefore never requires a GPU stack, matching the
import discipline of ``reason_pilot`` and ``speed_matrix``.

The padding convention is the one validated by the speed matrix (see
``speed_matrix._padded_batch``): **left-pad** every row of a fixed batch to the
longest row so that, after generation, ``sequences[:, width:]`` is exactly each
row's generated tokens (Hugging Face right-pads *finished* rows with
``pad_token_id``; :func:`trim_generated` removes that).
"""

from __future__ import annotations


def grouped_batches(n_items, batch_size):
    """Fixed-composition batches over ``range(n_items)``, in index order.

    ``grouped_batches(10, 4)`` -> ``[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]``. The
    composition of each batch is fixed for the whole run (no reshuffling), and
    ``batch_size == 1`` yields one singleton per item -- the N=1 path.
    """
    n_items = int(n_items)
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
    return [
        list(range(start, min(start + batch_size, n_items)))
        for start in range(0, n_items, batch_size)
    ]


def pad_sequences(id_lists, pad_id):
    """Left-pad ``id_lists`` to the longest; return ``(input_ids, attention_mask)``
    as plain Python lists (pure, so it is testable without torch).
    """
    id_lists = [list(ids) for ids in id_lists]
    if not id_lists:
        raise ValueError("cannot pad an empty batch")
    width = max(len(ids) for ids in id_lists)
    input_ids, attention_mask = [], []
    for ids in id_lists:
        pad = width - len(ids)
        input_ids.append([pad_id] * pad + ids)
        attention_mask.append([0] * pad + [1] * len(ids))
    return input_ids, attention_mask


def padded_batch(id_lists, pad_id, device=None):
    """Left-pad a batch into long tensors ``(input_ids, attention_mask)``.

    Thin tensor wrapper around :func:`pad_sequences`; torch is imported here so
    that the pure helpers above stay importable without it.
    """
    import torch

    input_ids, attention_mask = pad_sequences(id_lists, pad_id)
    kwargs = {"dtype": torch.long}
    if device is not None:
        kwargs["device"] = device
    return torch.tensor(input_ids, **kwargs), torch.tensor(attention_mask, **kwargs)


def trim_generated(ids, stop_ids, pad_id=None):
    """Trim one generated id sequence to its stop token, inclusive.

    Returns ``(kept, stopped)``: ``kept`` ends at the first stop token (so the
    token that stopped generation is retained, matching the manual greedy loop),
    and ``stopped`` is True iff a stop token was produced (False means the cap
    was hit). Trailing ``pad_id`` (Hugging Face's right-padding of finished rows)
    is dropped first, but never when ``pad_id`` is itself one of ``stop_ids`` --
    in that ambiguous case the first stop wins and trailing padding after it is
    irrelevant.
    """
    ids = [int(token) for token in ids]
    stop = {int(token) for token in stop_ids}
    if pad_id is not None and int(pad_id) not in stop:
        while ids and ids[-1] == int(pad_id):
            ids.pop()
    for index, token in enumerate(ids):
        if token in stop:
            return ids[: index + 1], True
    return ids, False
