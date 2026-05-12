from __future__ import annotations

import torch
import pytest

from utils.checkpointing import (
    TrainingCheckpointState,
    latest_checkpoint,
    load_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
)


def _trained_components() -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    student = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    loss = student(torch.ones(4, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return student, optimizer, scheduler


def test_save_and_load_checkpoint_wrappers_preserve_dict(tmp_path) -> None:
    path = tmp_path / "plain.pt"
    save_checkpoint(path, {"value": torch.tensor([1, 2, 3]), "name": "plain"})

    loaded = load_checkpoint(path)

    assert loaded["name"] == "plain"
    assert torch.equal(loaded["value"], torch.tensor([1, 2, 3]))


def test_save_training_checkpoint_creates_file_and_roundtrips_metadata(tmp_path) -> None:
    student, optimizer, scheduler = _trained_components()

    path = save_training_checkpoint(
        tmp_path,
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        step=12,
        optimizer_step=3,
        config={"batch_size": 2},
        metadata={"run": "mock"},
    )

    assert path == tmp_path / "checkpoint_step_12_opt_3.pt"
    assert path.is_file()

    restored_student = torch.nn.Linear(3, 2)
    state = load_training_checkpoint(
        path,
        restored_student,
        map_location="cpu",
        load_optimizer=False,
        load_rng_state=False,
    )

    assert isinstance(state, TrainingCheckpointState)
    assert state.step == 12
    assert state.optimizer_step == 3
    assert state.config == {"batch_size": 2}
    assert state.metadata == {"run": "mock"}
    assert state.path == path


def test_load_training_checkpoint_restores_student_optimizer_and_scheduler(tmp_path) -> None:
    student, optimizer, scheduler = _trained_components()
    path = save_training_checkpoint(
        tmp_path,
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        step=4,
        optimizer_step=1,
        config=None,
        metadata=None,
    )

    restored_student = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_student.parameters(), lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer,
        step_size=1,
        gamma=0.5,
    )
    load_training_checkpoint(
        path,
        restored_student,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        load_rng_state=False,
    )

    for original, restored in zip(student.parameters(), restored_student.parameters()):
        assert torch.equal(original, restored)
    assert restored_optimizer.state_dict()["state"]
    assert restored_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_load_training_checkpoint_can_skip_optimizer_restore(tmp_path) -> None:
    student, optimizer, _scheduler = _trained_components()
    path = save_training_checkpoint(
        tmp_path,
        student=student,
        optimizer=optimizer,
        step=4,
        optimizer_step=1,
        config=None,
        metadata=None,
    )

    restored_student = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_student.parameters(), lr=0.1)
    assert restored_optimizer.state_dict()["state"] == {}

    load_training_checkpoint(
        path,
        restored_student,
        optimizer=restored_optimizer,
        load_optimizer=False,
        load_rng_state=False,
    )

    assert restored_optimizer.state_dict()["state"] == {}
    for original, restored in zip(student.parameters(), restored_student.parameters()):
        assert torch.equal(original, restored)


def test_latest_checkpoint_returns_highest_step_then_optimizer_step(tmp_path) -> None:
    student = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    save_training_checkpoint(tmp_path, student, optimizer, step=1, optimizer_step=10, config=None, metadata=None)
    expected = save_training_checkpoint(tmp_path, student, optimizer, step=2, optimizer_step=1, config=None, metadata=None)
    save_training_checkpoint(tmp_path, student, optimizer, step=1, optimizer_step=20, config=None, metadata=None)
    (tmp_path / "checkpoint_step_bad_opt_99.pt").write_text("ignore", encoding="utf-8")

    assert latest_checkpoint(tmp_path) == expected


def test_latest_checkpoint_returns_none_when_no_checkpoint_exists(tmp_path) -> None:
    assert latest_checkpoint(tmp_path) is None


def test_load_training_checkpoint_restores_torch_cpu_rng(tmp_path) -> None:
    student = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    torch.manual_seed(12345)
    path = save_training_checkpoint(tmp_path, student, optimizer, step=0, optimizer_step=0, config=None, metadata=None)
    expected = torch.rand(5)

    torch.manual_seed(99999)
    load_training_checkpoint(path, student, load_rng_state=True)
    actual = torch.rand(5)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_load_training_checkpoint_restores_torch_cuda_rng(tmp_path) -> None:
    student = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    torch.cuda.manual_seed_all(12345)
    path = save_training_checkpoint(tmp_path, student, optimizer, step=0, optimizer_step=0, config=None, metadata=None)
    expected = [
        torch.rand(5, device=torch.device("cuda", device_index)).cpu()
        for device_index in range(torch.cuda.device_count())
    ]

    torch.cuda.manual_seed_all(99999)
    load_training_checkpoint(path, student, load_rng_state=True)
    actual = [
        torch.rand(5, device=torch.device("cuda", device_index)).cpu()
        for device_index in range(torch.cuda.device_count())
    ]

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)


def test_save_training_checkpoint_uses_canonical_rng_schema_and_dtype(tmp_path) -> None:
    student = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    path = save_training_checkpoint(tmp_path, student, optimizer, step=0, optimizer_step=0, config=None, metadata=None)

    payload = load_checkpoint(path)
    rng_state = payload["rng_state"]

    assert set(rng_state) == {"torch_cpu", "torch_cuda", "python_random", "numpy_random"}
    assert isinstance(rng_state["torch_cpu"], torch.Tensor)
    assert rng_state["torch_cpu"].device.type == "cpu"
    assert rng_state["torch_cpu"].dtype == torch.uint8
    if rng_state["torch_cuda"] is not None:
        assert all(isinstance(item, torch.Tensor) for item in rng_state["torch_cuda"])
        assert all(item.device.type == "cpu" for item in rng_state["torch_cuda"])
        assert all(item.dtype == torch.uint8 for item in rng_state["torch_cuda"])


def test_load_training_checkpoint_restores_legacy_list_torch_rng_state(tmp_path) -> None:
    student = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    torch.manual_seed(2468)
    path = save_training_checkpoint(
        tmp_path,
        student,
        optimizer,
        step=0,
        optimizer_step=0,
        config=None,
        metadata=None,
        rng_state=False,
    )
    legacy_rng_state = torch.get_rng_state().tolist()
    payload = load_checkpoint(path)
    payload["rng_state"] = {"torch_rng_state": legacy_rng_state}
    save_checkpoint(path, payload)
    expected = torch.rand(5)

    torch.manual_seed(1357)
    load_training_checkpoint(path, student, load_rng_state=True)
    actual = torch.rand(5)

    assert torch.equal(actual, expected)


def test_load_training_checkpoint_restores_legacy_torch_cpu_list_rng_state(tmp_path) -> None:
    student = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    torch.manual_seed(8642)
    path = save_training_checkpoint(
        tmp_path,
        student,
        optimizer,
        step=0,
        optimizer_step=0,
        config=None,
        metadata=None,
        rng_state=False,
    )
    legacy_rng_state = torch.get_rng_state().tolist()
    payload = load_checkpoint(path)
    payload["rng_state"] = {"torch_cpu": legacy_rng_state}
    save_checkpoint(path, payload)
    expected = torch.rand(5)

    torch.manual_seed(9753)
    load_training_checkpoint(path, student, load_rng_state=True)
    actual = torch.rand(5)

    assert torch.equal(actual, expected)


def test_load_training_checkpoint_requires_rng_state_when_requested(tmp_path) -> None:
    student = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    path = save_training_checkpoint(
        tmp_path,
        student,
        optimizer,
        step=0,
        optimizer_step=0,
        config=None,
        metadata=None,
        rng_state=False,
    )

    with pytest.raises(ValueError, match="rng_state"):
        load_training_checkpoint(path, student, load_rng_state=True)

    load_training_checkpoint(path, student, load_rng_state=False)


def test_load_training_checkpoint_requires_torch_rng_state_when_requested(tmp_path) -> None:
    student = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    path = save_training_checkpoint(tmp_path, student, optimizer, step=0, optimizer_step=0, config=None, metadata=None)
    payload = load_checkpoint(path)
    payload["rng_state"] = {"python_random": payload["rng_state"]["python_random"]}
    save_checkpoint(path, payload)

    with pytest.raises(ValueError, match="torch_cpu"):
        load_training_checkpoint(path, student, load_rng_state=True)


def test_load_training_checkpoint_rejects_non_dict_rng_state(tmp_path) -> None:
    student = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    path = save_training_checkpoint(tmp_path, student, optimizer, step=0, optimizer_step=0, config=None, metadata=None)
    payload = load_checkpoint(path)
    payload["rng_state"] = "not-a-dict"
    save_checkpoint(path, payload)

    with pytest.raises(ValueError, match="rng_state must be a dict"):
        load_training_checkpoint(path, student, load_rng_state=True)


def test_load_training_checkpoint_rejects_float_cpu_rng_state(tmp_path) -> None:
    student = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    path = save_training_checkpoint(tmp_path, student, optimizer, step=0, optimizer_step=0, config=None, metadata=None)
    payload = load_checkpoint(path)
    payload["rng_state"]["torch_cpu"] = torch.zeros_like(torch.get_rng_state(), dtype=torch.float32)
    save_checkpoint(path, payload)

    with pytest.raises(ValueError, match="integer byte values"):
        load_training_checkpoint(path, student, load_rng_state=True)


def test_load_training_checkpoint_rejects_2d_cpu_rng_state(tmp_path) -> None:
    student = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    path = save_training_checkpoint(tmp_path, student, optimizer, step=0, optimizer_step=0, config=None, metadata=None)
    payload = load_checkpoint(path)
    payload["rng_state"]["torch_cpu"] = torch.zeros(2, 2, dtype=torch.uint8)
    save_checkpoint(path, payload)

    with pytest.raises(ValueError, match="1D RNG state"):
        load_training_checkpoint(path, student, load_rng_state=True)


def test_load_training_checkpoint_rejects_invalid_cuda_rng_container(tmp_path) -> None:
    student = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    path = save_training_checkpoint(tmp_path, student, optimizer, step=0, optimizer_step=0, config=None, metadata=None)
    payload = load_checkpoint(path)
    payload["rng_state"]["torch_cuda"] = "not-cuda-rng"
    save_checkpoint(path, payload)

    with pytest.raises(ValueError, match="torch_cuda"):
        load_training_checkpoint(path, student, load_rng_state=True)


def test_load_training_checkpoint_skips_incomplete_rng_state_when_disabled(tmp_path) -> None:
    student = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    path = save_training_checkpoint(tmp_path, student, optimizer, step=0, optimizer_step=0, config=None, metadata=None)
    payload = load_checkpoint(path)
    payload["rng_state"] = {"python_random": payload["rng_state"]["python_random"]}
    save_checkpoint(path, payload)

    state = load_training_checkpoint(path, student, load_rng_state=False)

    assert state.step == 0
