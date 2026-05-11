import pytest
import torch

from models.cdm_engine import DeltaPerturbationEngine, OffTrajectoryConfig


def _engine(**kwargs: object) -> DeltaPerturbationEngine:
    return DeltaPerturbationEngine(OffTrajectoryConfig(**kwargs))


@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 8)])
def test_make_off_state_preserves_shape(shape: tuple[int, ...]) -> None:
    h = torch.randn(*shape)
    h_delta_alt = torch.randn(*shape)
    engine = _engine(noise_sigma=0.0)

    h_off = engine.make_off_state(h, h_delta_alt=h_delta_alt)

    assert h_off.shape == h.shape


def test_make_off_state_uses_alt_delta_direction_deterministically() -> None:
    h = torch.zeros(2, 3, 5)
    h_delta_alt = torch.ones_like(h) * 2.0
    engine = _engine(noise_sigma=0.0, rho_min=1.0, rho_max=1.0)

    h_off = engine.make_off_state(h, h_delta_alt=h_delta_alt)

    assert torch.allclose(h_off, h_delta_alt)


def test_make_off_state_changes_values_with_noise() -> None:
    torch.manual_seed(7)
    h = torch.ones(2, 3, 5)
    engine = _engine(noise_sigma=0.25)

    h_off = engine.make_off_state(h)

    assert h_off.shape == h.shape
    assert not torch.allclose(h_off, h)


def test_detach_direction_blocks_gradient_to_h_delta_alt() -> None:
    h = torch.randn(2, 3, 5, requires_grad=True)
    h_delta_alt = torch.randn(2, 3, 5, requires_grad=True)
    engine = _engine(
        noise_sigma=0.0,
        rho_min=1.0,
        rho_max=1.0,
        detach_direction=True,
    )

    loss = engine.make_off_state(h, h_delta_alt=h_delta_alt).sum()
    loss.backward()

    assert h.grad is not None
    assert h.grad.abs().sum() > 0
    assert h_delta_alt.grad is None or torch.count_nonzero(h_delta_alt.grad) == 0


def test_attached_direction_allows_gradient_to_h_delta_alt() -> None:
    h = torch.randn(2, 3, 5, requires_grad=True)
    h_delta_alt = torch.randn(2, 3, 5, requires_grad=True)
    engine = _engine(
        noise_sigma=0.0,
        rho_min=1.0,
        rho_max=1.0,
        detach_direction=False,
    )

    loss = engine.make_off_state(h, h_delta_alt=h_delta_alt).sum()
    loss.backward()

    assert h_delta_alt.grad is not None
    assert h_delta_alt.grad.abs().sum() > 0


def test_supports_rank_2_and_rank_3_states() -> None:
    engine = _engine(noise_sigma=0.0, rho_min=0.5, rho_max=0.5)

    h_rank_2 = torch.zeros(4, 8)
    alt_rank_2 = torch.ones_like(h_rank_2)
    h_rank_3 = torch.zeros(2, 3, 8)
    alt_rank_3 = torch.ones_like(h_rank_3)

    assert engine.make_off_state(h_rank_2, h_delta_alt=alt_rank_2).shape == h_rank_2.shape
    assert engine.make_off_state(h_rank_3, h_delta_alt=alt_rank_3).shape == h_rank_3.shape


@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 8)])
def test_rho_is_bounded_and_shared_across_hidden_channels(shape: tuple[int, ...]) -> None:
    torch.manual_seed(11)
    h = torch.zeros(*shape)
    h_delta_alt = torch.ones_like(h)
    engine = _engine(noise_sigma=0.0, rho_min=0.2, rho_max=0.8)

    h_off = engine.make_off_state(h, h_delta_alt=h_delta_alt)

    assert h_off.min() >= 0.2
    assert h_off.max() <= 0.8
    assert torch.allclose(h_off, h_off[..., :1].expand_as(h_off))


def test_invalid_state_rank_raises() -> None:
    h = torch.zeros(2, 3, 4, 5)
    h_delta_alt = torch.ones_like(h)
    engine = _engine(noise_sigma=0.0)

    with pytest.raises(ValueError):
        engine.make_off_state(h, h_delta_alt=h_delta_alt)


def test_invalid_rho_range_raises() -> None:
    with pytest.raises(ValueError):
        OffTrajectoryConfig(rho_min=1.1, rho_max=1.0)


def test_nonfinite_config_value_raises() -> None:
    with pytest.raises(ValueError):
        OffTrajectoryConfig(noise_sigma=float("nan"))


def test_negative_noise_sigma_raises() -> None:
    with pytest.raises(ValueError):
        OffTrajectoryConfig(noise_sigma=-0.1)


def test_h_delta_alt_shape_mismatch_raises() -> None:
    h = torch.randn(2, 3, 5)
    h_delta_alt = torch.randn(2, 1, 5)
    engine = _engine(noise_sigma=0.0)

    with pytest.raises(ValueError):
        engine.make_off_state(h, h_delta_alt=h_delta_alt)


def test_h_delta_alt_dtype_mismatch_raises() -> None:
    h = torch.randn(2, 3, 5, dtype=torch.float32)
    h_delta_alt = torch.randn(2, 3, 5, dtype=torch.float64)
    engine = _engine(noise_sigma=0.0)

    with pytest.raises(TypeError):
        engine.make_off_state(h, h_delta_alt=h_delta_alt)
