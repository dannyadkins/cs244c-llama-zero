from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import torch
from torch.utils.data import IterableDataset


@dataclass
class TokenizerBundle:
    tokenizer: object
    vocab_size: int
    eos_token_id: int


def load_llama_tokenizer(tokenizer_name: str) -> TokenizerBundle:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for real-data training. Install with `pip install transformers`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer '{tokenizer_name}' has no eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return TokenizerBundle(
        tokenizer=tokenizer,
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
    )


def load_fineweb_stream(
    subset: str,
    split: str = "train",
    shuffle_buffer: int = 10_000,
    seed: int = 42,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for FineWeb-Edu streaming. Install with `pip install datasets`."
        ) from exc

    ds = load_dataset("HuggingFaceFW/fineweb-edu", subset, split=split, streaming=True)
    ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
    return ds


class PackedFineWebDataset(IterableDataset):
    """Turns a text stream into fixed-length next-token-prediction examples."""

    def __init__(
        self,
        text_iterable: Iterable,
        tokenizer,
        seq_len: int,
        eos_token_id: int,
        text_key: str = "text",
        add_eos_between_samples: bool = True,
    ) -> None:
        super().__init__()
        self.text_iterable = text_iterable
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.text_key = text_key
        self.add_eos_between_samples = add_eos_between_samples

    def __iter__(self) -> Iterator[torch.Tensor]:
        token_buffer = []
        target_chunk_size = self.seq_len + 1

        for row in self.text_iterable:
            text = row.get(self.text_key, "") if isinstance(row, dict) else str(row)
            if not text:
                continue

            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if not encoded:
                continue

            token_buffer.extend(encoded)
            if self.add_eos_between_samples:
                token_buffer.append(self.eos_token_id)

            while len(token_buffer) >= target_chunk_size:
                chunk = token_buffer[:target_chunk_size]
                del token_buffer[:target_chunk_size]
                yield torch.tensor(chunk, dtype=torch.long)


class SyntheticPatternDataset(IterableDataset):
    """Deterministic offline fallback used for tests and quick sanity checks."""

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        num_sequences: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_sequences = num_sequences

    def __iter__(self) -> Iterator[torch.Tensor]:
        produced = 0
        cursor = 0
        while self.num_sequences is None or produced < self.num_sequences:
            length = self.seq_len + 1
            chunk = torch.arange(cursor, cursor + length, dtype=torch.long) % self.vocab_size
            cursor = (cursor + length) % self.vocab_size
            produced += 1
            yield chunk


class RankShardIterableDataset(IterableDataset):
    """Strides an iterable dataset so each rank sees a disjoint shard."""

    def __init__(self, dataset: IterableDataset, rank: int, world_size: int) -> None:
        super().__init__()
        if rank < 0 or world_size <= 0 or rank >= world_size:
            raise ValueError(f"invalid rank/world_size: rank={rank}, world_size={world_size}")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        for idx, sample in enumerate(self.dataset):
            if idx % self.world_size == self.rank:
                yield sample
