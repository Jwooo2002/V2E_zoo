# CSDM Mamba KD

This repository implements Continuous-State Distribution Matching (CSDM) for
Transformer-to-Mamba knowledge distillation. The current implementation is
Stage 1 plus Stage 2 mock-state engine pieces, the Stage 3 minimal mock
training scaffold, and Stage 4 mock evaluation scaffolds: configuration
skeletons, KD/CSDM loss functions, off-trajectory student-state construction,
mock teacher/student modules, token-weighted evaluation metrics, and unit tests
with mock tensors.

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
- `models/teacher_wrapper.py`: frozen mock teacher that consumes only clean
  token IDs and returns token-prefix-aligned logits.
- `models/student_mamba.py`: lightweight mock student that produces
  on-trajectory logits, off-trajectory logits, and detached fake logits.
- `data/dataset.py`: deterministic random-token mock dataset with next-token
  shifted labels and `ignore_index` on the final placeholder token.
- `train.py`: mock-only training loop with gradient accumulation, CUDA bf16
  autocast when available, shared valid-position masking, and JSON console
  metrics.
- `evaluate.py`: mock-only Stage 4 evaluation CLI with JSON metrics.
- `evals/perplexity.py`: token-weighted next-token CE/perplexity evaluation.
- `evals/perturbation_robustness.py`: token-weighted
  `KL(teacher || student)` comparison for clean and off-trajectory student
  logits.
- `evals/needle.py`: deterministic synthetic Needle-in-a-Haystack metric
  scaffold for mock mode only.
- `configs/train_config.yaml`: minimal training/loss defaults for mock mode.
- `configs/ds_config.json`: placeholder future DeepSpeed config; DeepSpeed is
  not a required dependency.
- `configs/model_config.yaml`: model-role placeholders without real imports.
- `tests/`: mock-tensor tests for shapes, finite losses, invalid inputs, and
  gradient-flow behavior.

## Stage 3 Mock Training

Run two optimizer steps without real Llama or Mamba imports:

```bash
python train.py --config configs/train_config.yaml --mock --max_steps 2
```

`MockTextDataset` creates labels by shifting tokens left for next-token
prediction and sets `labels[-1] = -100`. The training loop applies one shared
valid-position mask to CE, on-trajectory KD, and off-trajectory CSDM, so the
last placeholder token is excluded from all three objectives.

Teacher/student alignment is token-prefix based: `teacher_logits[:, t]`
represents `p_phi(y | x_{<=t})` and is aligned with `on_logits[:, t]` and
`off_logits[:, t]`. The teacher never receives `h_t`, `h'_t`, `h_delta_alt`, or
any Mamba state. The mock student's `h_delta_alt` is only a student-side
surrogate used to exercise the off-trajectory engine; it is not real Mamba
delta behavior. Fake logits are detached at the producer boundary before being
passed to `csdm_loss`.

## Stage 4 Mock Evaluation

Run all mock evaluation scaffolds:

```bash
python evaluate.py --config configs/train_config.yaml --mock --mode all --max_batches 2
```

Individual modes are available with `--mode perplexity`, `--mode perturbation`,
and `--mode needle`. Non-mock evaluation is intentionally not implemented yet,
so the CLI exits clearly if `--mock` is omitted.

Perplexity accumulates CE with `reduction="sum"` over the same valid-position
mask used by training, then divides by the number of valid tokens. Perturbation
robustness computes full-vocab mock-mode `KL(teacher || student)` for
on-trajectory and off-trajectory student logits using the configured
temperature. The teacher consumes only clean `input_ids`; it never consumes
student states. The needle scaffold is deterministic synthetic bookkeeping only
and should not be interpreted as real long-context reasoning evaluation.

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
