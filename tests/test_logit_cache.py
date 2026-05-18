from __future__ import annotations

import pytest
import torch

from utils.logit_cache import (
    LogitCacheConfig,
    LogitCacheEntry,
    LogitCacheLoadError,
    TeacherLogitCache,
)


def _cache(tmp_path, **kwargs) -> TeacherLogitCache:
    config = LogitCacheConfig(enabled=True, cache_dir=str(tmp_path), **kwargs)
    return TeacherLogitCache(config)


def test_make_key_deterministic_for_same_input_ids(tmp_path) -> None:
    cache = _cache(tmp_path)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    key_a = cache.make_key(input_ids)
    key_b = cache.make_key(input_ids.clone())

    assert key_a == key_b


def test_make_key_changes_when_input_ids_change(tmp_path) -> None:
    cache = _cache(tmp_path)
    input_ids = torch.tensor([[1, 2, 3]])
    changed = torch.tensor([[1, 2, 4]])

    assert cache.make_key(input_ids) != cache.make_key(changed)


def test_make_key_changes_when_attention_mask_changes(tmp_path) -> None:
    cache = _cache(tmp_path)
    input_ids = torch.tensor([[1, 2, 3]])
    mask_a = torch.tensor([[1, 1, 0]])
    mask_b = torch.tensor([[1, 0, 0]])

    assert cache.make_key(input_ids, attention_mask=mask_a) != cache.make_key(
        input_ids,
        attention_mask=mask_b,
    )


def test_save_load_full_logits_preserves_values_shape_dtype_metadata_and_no_grad(tmp_path) -> None:
    cache = _cache(tmp_path, dtype="float32")
    logits = torch.randn(2, 3, 5, requires_grad=True)
    key = "full-logits"

    cache.save(key, logits=logits, metadata={"teacher": "mock"})
    entry = cache.load(key)

    assert entry.logits is not None
    assert entry.logits.shape == logits.shape
    assert entry.logits.dtype == torch.float32
    assert torch.allclose(entry.logits, logits.detach())
    assert entry.logits.requires_grad is False
    assert entry.metadata["teacher"] == "mock"
    assert entry.metadata["input_shape"] == [2, 3]
    assert entry.metadata["vocab_size"] == 5


def test_get_or_compute_computes_once_then_loads(tmp_path) -> None:
    cache = _cache(tmp_path, dtype="float32")
    input_ids = torch.tensor([[1, 2, 3]])
    calls = {"count": 0}

    def compute_fn(input_ids, attention_mask=None):
        calls["count"] += 1
        return torch.full((*input_ids.shape, 7), float(calls["count"]))

    first = cache.get_or_compute(input_ids, compute_fn)
    second = cache.get_or_compute(input_ids, compute_fn)

    assert calls["count"] == 1
    assert first.logits is not None
    assert second.logits is not None
    assert torch.equal(first.logits, second.logits)


def test_load_corrupt_cache_entry_raises_clear_error(tmp_path) -> None:
    cache = _cache(tmp_path, dtype="float32")
    (tmp_path / "bad.pt").write_bytes(b"not a torch checkpoint")

    with pytest.raises(LogitCacheLoadError, match="could not be loaded"):
        cache.load("bad")


def test_get_or_compute_recovers_from_corrupt_cache_entry(tmp_path) -> None:
    cache = _cache(tmp_path, dtype="float32")
    input_ids = torch.tensor([[1, 2, 3]])
    key = cache.make_key(input_ids)
    cache._path_for_key(key).write_bytes(b"partial cache file")
    calls = {"count": 0}

    def compute_fn(input_ids, attention_mask=None):
        calls["count"] += 1
        return torch.full((*input_ids.shape, 4), 3.0)

    entry = cache.get_or_compute(input_ids, compute_fn)
    loaded = cache.load(key)

    assert calls["count"] == 1
    assert entry.logits is not None
    assert loaded.logits is not None
    assert torch.equal(entry.logits, torch.full((*input_ids.shape, 4), 3.0))
    assert torch.equal(loaded.logits, entry.logits)


def test_overwrite_true_recomputes(tmp_path) -> None:
    cache = _cache(tmp_path, dtype="float32", overwrite=True)
    input_ids = torch.tensor([[1, 2, 3]])
    calls = {"count": 0}

    def compute_fn(input_ids, attention_mask=None):
        calls["count"] += 1
        return torch.full((*input_ids.shape, 4), float(calls["count"]))

    first = cache.get_or_compute(input_ids, compute_fn)
    second = cache.get_or_compute(input_ids, compute_fn)

    assert calls["count"] == 2
    assert first.logits is not None
    assert second.logits is not None
    assert torch.all(second.logits == 2.0)


def test_top_k_mode_saves_topk_tensors_without_full_logits(tmp_path) -> None:
    cache = _cache(tmp_path, dtype="float32", use_top_k=True, top_k=2)
    logits = torch.tensor([[[0.1, 0.8, -0.2, 1.5], [2.0, 0.0, 3.0, 1.0]]])

    cache.save("topk", logits=logits)
    entry = cache.load("topk")

    assert entry.logits is None
    assert entry.topk_values is not None
    assert entry.topk_indices is not None
    assert entry.topk_values.shape == (1, 2, 2)
    assert entry.topk_indices.shape == (1, 2, 2)
    expected_values, expected_indices = torch.topk(logits, k=2, dim=-1)
    assert torch.equal(entry.topk_values, expected_values)
    assert torch.equal(entry.topk_indices, expected_indices)
    assert entry.metadata["top_k"] == 2
    assert entry.metadata["vocab_size"] == 4


def test_invalid_top_k_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        LogitCacheConfig(cache_dir=str(tmp_path), top_k=0)


def test_topk_values_indices_shape_mismatch_raises(tmp_path) -> None:
    cache = _cache(tmp_path)
    values = torch.randn(2, 3, 4)
    indices = torch.zeros(2, 3, 5, dtype=torch.long)

    with pytest.raises(ValueError):
        cache.save("bad-topk", topk_values=values, topk_indices=indices)


def test_topk_values_indices_rank_mismatch_raises(tmp_path) -> None:
    cache = _cache(tmp_path)
    values = torch.randn(3, 4)
    indices = torch.zeros(3, 4, dtype=torch.long)

    with pytest.raises(ValueError):
        cache.save("bad-topk-rank", topk_values=values, topk_indices=indices)


def test_get_or_compute_rejects_logits_with_wrong_prefix_shape(tmp_path) -> None:
    cache = _cache(tmp_path)
    input_ids = torch.tensor([[1, 2, 3]])

    def compute_fn(input_ids, attention_mask=None):
        return torch.randn(1, 2, 5)

    with pytest.raises(ValueError, match="prefix shape"):
        cache.get_or_compute(input_ids, compute_fn)


def test_get_or_compute_rejects_topk_with_wrong_prefix_shape(tmp_path) -> None:
    cache = _cache(tmp_path)
    input_ids = torch.tensor([[1, 2, 3]])

    def compute_fn(input_ids, attention_mask=None):
        return LogitCacheEntry(
            topk_values=torch.randn(1, 2, 4),
            topk_indices=torch.zeros(1, 2, 4, dtype=torch.long),
        )

    with pytest.raises(ValueError, match="prefix shape"):
        cache.get_or_compute(input_ids, compute_fn)


def test_compute_fn_called_under_no_grad(tmp_path) -> None:
    cache = _cache(tmp_path)
    input_ids = torch.tensor([[1, 2]])
    grad_modes = []

    def compute_fn(input_ids, attention_mask=None):
        grad_modes.append(torch.is_grad_enabled())
        return LogitCacheEntry(logits=torch.randn(1, 2, 3, requires_grad=True))

    entry = cache.get_or_compute(input_ids, compute_fn)

    assert grad_modes == [False]
    assert entry.logits is not None
    assert entry.logits.requires_grad is False
