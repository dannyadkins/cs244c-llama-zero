import torch

from data.fineweb import PackedFineWebDataset


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        # token id is char ordinal modulo 32.
        return [ord(c) % 32 for c in text]


def test_packed_dataset_emits_fixed_length_chunks() -> None:
    rows = [{"text": "abcd"}, {"text": "efgh"}, {"text": "ijkl"}]
    dataset = PackedFineWebDataset(
        text_iterable=rows,
        tokenizer=DummyTokenizer(),
        seq_len=4,
        eos_token_id=1,
    )

    chunks = list(dataset)
    assert len(chunks) >= 2
    assert all(isinstance(chunk, torch.Tensor) for chunk in chunks)
    assert all(chunk.shape == (5,) for chunk in chunks)
