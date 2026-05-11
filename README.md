# CSDM Mamba KD

This repository implements Continuous-State Distribution Matching (CSDM) for
Transformer-to-Mamba knowledge distillation. The current implementation is
Stage 1 plus Stage 2 mock-state engine pieces: configuration skeletons,
KD/CSDM loss functions, off-trajectory student-state construction, and unit
tests with mock tensors.

No real Llama or Mamba modules are imported in the implemented stages. The
implemented losses operate on logits shaped `[B, T, V]` or `[B, N, V]`. The
Stage 2 engine operates on mock student recurrent states shaped `[B, D]` or
`[B, T, D]`.

## Implemented Files

- `losses/kd_loss.py`: temperature-scaled `KL(teacher || student)` in
  logit space, with teacher logits detached internally.
- `losses/cdm_loss.py`: centered-logit utilities and CSDM off-trajectory loss,
  with teacher and fake-student logits detached internally.
- `models/cdm_engine.py`: delta-perturbation off-state engine for mock Mamba
  student states, with strict alternate-state validation and placeholder
  adapter interface.
- `configs/train_config.yaml`: minimal training/loss defaults for mock mode.
- `configs/model_config.yaml`: model-role placeholders without real imports.
- `tests/`: mock-tensor tests for shapes, finite losses, invalid inputs, and
  gradient-flow behavior.

## Stage 2 Off-State Engine

`DeltaPerturbationEngine.make_off_state(...)` constructs mock off-trajectory
student states:

```python
from models.cdm_engine import DeltaPerturbationEngine, OffTrajectoryConfig

engine = DeltaPerturbationEngine(OffTrajectoryConfig(noise_sigma=0.0))
h_off = engine.make_off_state(h, h_delta_alt=h_delta_alt)
```

`h_delta_alt` is an alternate Mamba student recurrent state, not a teacher
state. When provided, it must have exactly the same shape, device, and dtype as
`h`. If `h_delta_alt` is omitted, the engine is using the Gaussian-noise-only
baseline/fallback rather than the preferred CSDM direction.

`delta_perturb_eps` is reserved for future
`MambaStateAdapter.forward_with_delta_scale(...)` implementations. It is not
consumed by `make_off_state(...)` in this mock Stage 2 implementation.

## Loss Validation

Run:

```bash
python -m compileall .
pytest -q
```

The tests verify that:

- `center_logits(x)` preserves shape and centers the vocab-dimension mean.
- `rms(x)` reduces only the vocab dimension and returns shape `[..., 1]`.
- `kd_kl_loss(...)` returns finite scalar losses and sends gradients only to
  student logits.
- `csdm_loss(...)` returns finite scalar losses and sends gradients only to
  off-trajectory student logits.
- `DeltaPerturbationEngine.make_off_state(...)` preserves mock state shapes,
  samples rho as `h.shape[:-1] + (1,)`, validates alternate student states
  strictly, and respects `detach_direction` gradient behavior.
- shape mismatches and invalid temperatures raise errors.
