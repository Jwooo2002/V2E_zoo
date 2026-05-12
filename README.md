# CSDM Mamba KD

This repository implements Continuous-State Distribution Matching (CSDM) for
Transformer-to-Mamba knowledge distillation. The current implementation is
Stage 1 plus Stage 2 mock-state engine pieces, the Stage 3 minimal mock
training scaffold, Stage 4 mock evaluation scaffolds, Stage 5A HuggingFace
teacher wrapper integration, the Stage 5B teacher-logit cache scaffold,
Stage 5C real-HF-teacher smoke training, Stage 5D top-k KD/CSDM support
with a mock student, Stage 5E teacher-cache integration in training,
Stage 6A optional real-Mamba student adapter scaffold, Stage 6B Mamba
dependency diagnostics, Stage 6C real-Mamba forward smoke support, Stage
6D/6E student-side state and approximate off-trajectory scaffolding, and
Stage 6F opt-in HF-teacher/RealMambaStudent smoke training, and Stage 7A
local tokenizer/text data smoke support:
configuration skeletons, KD/CSDM loss functions, off-trajectory student-state
construction, mock teacher/student modules, token-weighted evaluation metrics,
teacher-logit cache utilities, and unit tests with mock tensors.

No real Llama or Mamba modules are imported by default. The HuggingFace teacher
wrapper imports `transformers` only when instantiated, and
`RealMambaStudent` imports `mamba_ssm` only when instantiated. Mock training and
tests do not require Llama weights, HF login, external downloads,
`transformers`, or `mamba-ssm`. The implemented losses operate on logits shaped
`[B, T, V]` or `[B, N, V]`. The Stage 2 engine operates on mock student
recurrent states shaped `[B, D]` or `[B, T, D]`.

## Implemented Files

- `losses/kd_loss.py`: temperature-scaled `KL(teacher || student)` in
  logit space, with teacher logits detached internally and optional top-k
  selected-vocab utilities.
- `losses/cdm_loss.py`: centered-logit utilities and CSDM off-trajectory loss,
  with teacher and fake-student logits detached internally.
- `models/cdm_engine.py`: delta-perturbation off-state engine for mock Mamba
  student states, with strict alternate-state validation and placeholder
  adapter interface.
- `models/teacher_wrapper.py`: frozen mock teacher plus opt-in HuggingFace
  causal-LM teacher wrapper. Both consume only clean token IDs and attention
  masks, never Mamba states, and return token-prefix-aligned logits.
- `utils/logit_cache.py`: optional teacher-logit cache utility for clean
  token-prefix teacher outputs, with full-logit and top-k storage modes.
- `utils/mamba_env.py`: optional real-Mamba dependency diagnostics with lazy
  checks for `mamba_ssm` and `causal-conv1d`.
- `scripts/check_mamba_env.py`: CLI wrapper for the Stage 6B dependency report.
- `scripts/check_mamba_forward.py`: opt-in Stage 6C-6E real-Mamba forward
  smoke check with tiny config-driven dimensions, state/off-state reporting,
  and no downloads.
- `models/student_mamba.py`: lightweight mock student that produces
  on-trajectory logits, off-trajectory logits, and detached fake logits, plus
  an optional `RealMambaStudent` adapter with lazy `mamba_ssm` import, public
  `MambaLMHeadModel` support, and no private-internal assumptions.
- `data/dataset.py`: deterministic random-token mock dataset with next-token
  shifted labels, plus local text/JSONL tokenized datasets with right padding,
  attention masks, and invalid labels set to `ignore_index`.
- `data/tokenizer.py`: lazy HuggingFace tokenizer loader for opt-in local text
  data, with configurable pad-token handling.
- `train.py`: mock training plus opt-in HuggingFace-teacher smoke paths for a
  mock student or `RealMambaStudent`, with gradient accumulation, CUDA-only
  autocast, optional full-logit teacher caching, shared valid-position masking,
  and JSON console metrics.
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
- `configs/model_config.yaml`: mock teacher/student defaults plus an opt-in
  HuggingFace teacher example block.
- `docs/requirements-mamba.txt`: optional real-Mamba dependency notes.
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

## Stage 6C Real Mamba Forward Smoke

`RealMambaStudent` is opt-in and still does not import `mamba_ssm` at module
import time, so mock tests and mock training remain dependency-free:

```bash
python -m compileall .
pytest -q tests/test_mamba_adapter.py tests/test_real_mamba_smoke.py tests/test_train_smoke.py
```

Run the real forward smoke explicitly:

```bash
python scripts/check_mamba_forward.py \
  --device cuda \
  --batch-size 1 \
  --seq-len 16 \
  --vocab-size 128 \
  --hidden-size 64 \
  --num-layers 2
```

Use `--device cpu` for environments without visible CUDA. Some `mamba-ssm`
builds expose CUDA-only fast paths; the Stage 6C adapter switches to public
reference paths for CPU smoke when available, and otherwise the script exits
nonzero with a JSON error.

If CUDA forward fails with a fused `causal_conv1d_fwd` API/signature error,
the adapter/script retries once with a smoke-only reference/non-fused
causal-conv path. You can request that path directly:

```bash
python scripts/check_mamba_forward.py --device cuda --use-reference-forward
```

`mamba_ssm` import success does not guarantee fused CUDA forward compatibility.
When smoke still fails, stderr is a compact parseable JSON object with
truncated error text and tensor argument reprs omitted.
For example, a `mamba-ssm` wheel and `causal-conv1d` wheel can expose
incompatible CUDA extension signatures. Stage 6C treats CPU/reference smoke as
sufficient evidence for import/instantiate/forward plumbing. Full CUDA fused
Mamba training later requires a pinned compatible `mamba-ssm` +
`causal-conv1d` wheel pair; do not downgrade or install random versions
without pinning and validating the pair.

When `mamba_ssm` is available, `RealMambaStudent` uses the public
`MambaLMHeadModel` path. The smoke output is a `StudentOutput` with
`on_logits`, `off_logits`, and detached `fake_logits` shaped `[B, T, V]`.
By default Stage 6E builds an approximate student-side `h_delta_alt`, constructs
`h_off` with `DeltaPerturbationEngine`, projects `h_off` through the LM head,
and sets `fake_logits = off_logits.detach()`. This does not change teacher
behavior or training behavior. The teacher path is unchanged: the teacher
consumes only clean `input_ids` and optional `attention_mask`, never student
recurrent states.

## Stage 6D Real Mamba State Extraction Scaffold

Stage 6D adds an explicit student-side state exposure contract for
`RealMambaStudent` without implementing final off-trajectory construction. The
config field `state_extraction` supports:

- `last_hidden`: expose the public Mamba backbone output used by the LM head.
- `embedding`: expose token embeddings only as provisional shape plumbing.
- `none`: expose no state tensors.

`expose_states=false` suppresses `h`, `h_off`, and `h_delta_alt` in the returned
`StudentOutput`, but the adapter may still build internal student-side
off-trajectory tensors when Stage 6E modes are enabled. The embedding fallback
is not claimed to be the final recurrent Mamba state.

The smoke script reports the state fields explicitly:

```bash
python scripts/check_mamba_forward.py \
  --device cpu \
  --batch-size 1 \
  --seq-len 16 \
  --vocab-size 128 \
  --hidden-size 64 \
  --num-layers 2 \
  --state-extraction last_hidden
```

The JSON output includes `h_shape`, `h_off_shape`, `h_delta_alt_shape`,
`state_extraction`, and `smoke_placeholder_off_logits`.

## Stage 6E Approximate Real-Mamba Off-State Path

Stage 6E uses the Stage 6D student-side `h` representation to build an
approximate off-trajectory student representation and `off_logits`. This is not
true Mamba delta-kernel perturbation or recurrent-state injection. It is a
scaffold for wiring CSDM-compatible tensors without changing teacher behavior or
loss math.

Supported `RealMambaStudent` modes:

- `off_state_mode=projection`: build `h_delta_alt` and pass `h, h_delta_alt`
  through `DeltaPerturbationEngine`.
- `off_state_mode=placeholder`: mirror `h_delta_alt`, `h_off`, and
  `off_logits` from the on-trajectory path for smoke only.
- `off_state_mode=none`: do not build an off-state; real CSDM should not be
  enabled with this mode.
- `delta_alt_mode=delta_projection`: approximate drift as
  `h + eps * tanh(W h)` using a student-side linear layer.
- `delta_alt_mode=noise`: approximate drift with RMS-scaled Gaussian noise.
- `delta_alt_mode=identity`: use `h_delta_alt = h`.
- `off_logits_mode=lm_head`: project `h_off` through the public Mamba LM head.
- `off_logits_mode=projection_head`: project `h_off` through a dedicated
  student-side linear head.
- `off_logits_mode=placeholder`: keep `off_logits == on_logits` for smoke only.
- `off_state_detach_direction=true`: detach `h_delta_alt - h` before applying
  sampled `rho`, matching the intended stop-gradient direction.

`fake_logits` are `off_logits.detach()` for now, matching the early
detached-student placeholder used before an EMA student exists. The teacher
still consumes only clean token prefixes and never receives `h`, `h_off`,
`h_delta_alt`, or Mamba states.

Example:

```bash
python scripts/check_mamba_forward.py \
  --device cpu \
  --batch-size 1 \
  --seq-len 16 \
  --vocab-size 128 \
  --hidden-size 64 \
  --num-layers 2 \
  --off-state-mode projection \
  --delta-alt-mode delta_projection \
  --off-logits-mode lm_head \
  --off-state-detach-direction
```

The JSON output reports `off_state_mode`, `delta_alt_mode`, `off_logits_mode`,
`off_state_detach_direction`, `off_state_source`, `delta_alt_source`,
`off_logits_source`, `off_state_available`, `delta_alt_available`, and whether
`smoke_placeholder_off_logits` / `off_logits_placeholder` are active. True
delta-controlled Mamba-state extraction remains future research and engineering
work.

## Stage 6F HF Teacher + Real Mamba Smoke Training

Stage 6F enables tiny smoke-scale training with `student_type=mamba`. This is
not the final experiment-training path. It exists to verify that frozen teacher
logits, real-Mamba student logits, Stage 6E approximate off-state logits,
CE/KD/CSDM loss wiring, and gradient accumulation can run together on small
synthetic token batches.

Tests use fake `transformers` and fake public Mamba modules for the main HF
teacher + `RealMambaStudent` path, so they do not download weights:

```bash
pytest -q tests/test_hf_teacher_smoke.py tests/test_train_real_mamba_smoke.py
```

The optional real-Mamba subprocess smoke in
`tests/test_train_real_mamba_smoke.py` runs only when `mamba_ssm` is importable;
otherwise it is skipped.

The mathematical boundary is unchanged:

- the teacher consumes only clean `input_ids` and optional `attention_mask`;
- the teacher never receives `h`, `h_off`, `h_delta_alt`, or any Mamba state;
- `on_logits` come from the real Mamba forward path;
- `off_logits` come from the approximate Stage 6E `h_off` projection/LM-head
  path;
- `fake_logits = off_logits.detach()` until an EMA student is added;
- no direct Llama-Mamba hidden-state MSE is introduced.

Safe mock-teacher + real-Mamba smoke command:

```bash
python train.py \
  --config configs/train_config.yaml \
  --teacher-type mock \
  --student-type mamba \
  --max_steps 1 \
  --seq-len 16 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --student-vocab-size 128 \
  --student-hidden-size 64 \
  --student-num-layers 2 \
  --mixed-precision no \
  --csdm-weight 0.03
```

Manual HF-teacher + real-Mamba smoke command. This can download or require
local HF files depending on the model path, so it is not used by tests:

```bash
python train.py \
  --config configs/train_config.yaml \
  --teacher-type hf \
  --student-type mamba \
  --teacher-model-name-or-path sshleifer/tiny-gpt2 \
  --max_steps 1 \
  --seq-len 16 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --student-hidden-size 64 \
  --student-num-layers 2 \
  --mixed-precision no \
  --csdm-weight 0.03 \
  --topk-enabled \
  --top-k 128
```

For HF teacher smoke training, the real-Mamba student vocab size is aligned to
the teacher vocab size unless `--student-vocab-size` is explicitly provided.
If an explicit student vocab override does not match the teacher, training
raises during teacher/student logit validation.

## Stage 7A Tokenizer And Local Text Data

Stage 7A adds an opt-in local text data path for tiny smoke tests. It does not
change CE/KD/CSDM math and does not add large-scale dataset streaming. Mock data
remains the default.

Supported formats:

- `dataset_type=text`: read a local plain-text file.
- `dataset_type=jsonl`: read local JSONL rows from `text_field` (default
  `text`).

The tokenizer is loaded lazily through `data/tokenizer.py`. Tests use fake
tokenizers and do not download models. For HF teacher runs, omit
`--tokenizer-name-or-path` to default to `--teacher-model-name-or-path`.

Label and mask convention:

- `input_ids` are fixed length `[T]`, right-padded if needed.
- `attention_mask` is `1` for real tokens and `0` for padding.
- `labels[t] = input_ids[t + 1]` when both positions are real tokens.
- the final position and all padding-derived positions use `-100`.
- CE, KD, and CSDM continue to use one shared valid-token mask from labels.

Mock teacher/student text smoke:

```bash
python train.py \
  --config configs/train_config.yaml \
  --dataset-type text \
  --data-path data/smoke.txt \
  --tokenizer-name-or-path sshleifer/tiny-gpt2 \
  --teacher-type mock \
  --student-type mock \
  --max_steps 1 \
  --seq-len 16 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --mixed-precision no
```

Manual HF teacher + real-Mamba text smoke. This can download or require local
HF files, so it is not used by tests:

```bash
python train.py \
  --config configs/train_config.yaml \
  --teacher-type hf \
  --student-type mamba \
  --teacher-model-name-or-path sshleifer/tiny-gpt2 \
  --tokenizer-name-or-path sshleifer/tiny-gpt2 \
  --dataset-type text \
  --data-path data/smoke.txt \
  --max_steps 1 \
  --seq-len 16 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --student-hidden-size 64 \
  --student-num-layers 2 \
  --mixed-precision no \
  --csdm-weight 0.03 \
  --topk-enabled \
  --top-k 128
```

## Stage 6B Mamba Dependency Diagnostics

Stage 6B is an environment/import diagnostic only. Real Mamba dependencies are
still optional, and the diagnostics CLI imports `mamba_ssm` and
`causal_conv1d` only inside the explicit dependency check. Repository imports,
mock training, and tests do not require a real Mamba installation:

```bash
python scripts/check_mamba_env.py
python scripts/check_mamba_env.py --json
```

Missing Mamba dependencies are reported but do not fail the command by
default. Use `--require-mamba` for environment-gating scripts that should fail
unless both `mamba_ssm` and `causal_conv1d` can be imported:

```bash
python scripts/check_mamba_env.py --require-mamba
```

The CLI checks only runtime availability and does not change CSDM objectives,
teacher usage, off-trajectory state construction, or training behavior. The
teacher still consumes only clean token prefixes.

Optional install attempts after installing a compatible PyTorch build:

```bash
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

or:

```bash
pip install mamba-ssm[causal-conv1d] --no-build-isolation
```

These packages may compile CUDA extensions and can fail depending on PyTorch,
CUDA, compiler, and driver versions. Optional install notes live in
`docs/requirements-mamba.txt`.

## Stage 5D Top-k KD/CSDM

Top-k KD/CSDM is available but disabled by default. When enabled, `train.py`
builds one shared selected-vocab index tensor from detached raw teacher logits,
optionally appends valid next-token labels, and uses those same indices for
teacher, on-trajectory student, off-trajectory student, and fake-student logits.
CE remains full-vocab.

The selected-vocab losses are approximations. With the default
`renormalize_topk: true`, KD and CSDM renormalize over the selected K entries.
Teacher logits still come only from clean token prefixes, and teacher/fake
terms are detached inside the losses.

Example mock run:

```bash
python train.py --config configs/train_config.yaml --mock --max_steps 2 \
  --topk-enabled --top-k 256
```

## Stage 5E Teacher Cache Integration

Training can optionally cache frozen teacher logits for clean token prefixes:

```bash
python train.py --config configs/train_config.yaml --mock --max_steps 2 \
  --gradient-accumulation-steps 1 \
  --teacher-cache-enabled \
  --teacher-cache-dir /tmp/csdm_teacher_logits
```

Cache keys are derived from `input_ids`, optional `attention_mask`, and
teacher-output metadata such as teacher type, teacher implementation, vocab
size, mock teacher seed, or HuggingFace model identity. They do not include
labels, student logits, `h_t`, `h'_t`, `h_delta_alt`, `rho`, `sigma`, or
adapter state. Cache misses call:

```python
teacher(input_ids, attention_mask=attention_mask)
```

under `torch.no_grad()`, and cached logits are detached. During training,
full cached logits are moved to the student logits device before CE/KD/CSDM
loss computation. If top-k KD/CSDM is enabled, `train.py` still builds selected
indices from the full cached teacher logits:

```bash
python train.py --config configs/train_config.yaml --mock --max_steps 2 \
  --gradient-accumulation-steps 1 \
  --topk-enabled --top-k 256 \
  --teacher-cache-enabled \
  --teacher-cache-dir /tmp/csdm_teacher_logits
```

`--teacher-cache-overwrite` recomputes matching entries. Top-k-only cache
storage is available in `TeacherLogitCache`, but `train.py` raises
`NotImplementedError` for top-k-only cache entries until the cache-to-loss path
can consume cached `topk_values` and `topk_indices` directly.

## Stage 5A HuggingFace Teacher Wrapper

`HuggingFaceTeacherWrapper` is an opt-in frozen causal-LM teacher:

```python
from models.teacher_wrapper import HuggingFaceTeacherConfig, HuggingFaceTeacherWrapper

teacher = HuggingFaceTeacherWrapper(
    HuggingFaceTeacherConfig(
        model_name_or_path="/path/to/local/model",
        torch_dtype="bfloat16",
        device_map="auto",
        local_files_only=True,
    )
)
teacher_logits = teacher(input_ids, attention_mask=attention_mask)
```

The wrapper returns raw causal-LM logits and does not shift labels or compute
loss. `logits[:, t]` is the next-token distribution after prefix `x_{<=t}`.
Real KD requires compatible token indices between teacher and student, or an
explicit cached/top-k teacher-logit path that maps indices correctly. The mock
teacher remains the runnable default in `configs/model_config.yaml`; the
`hf_teacher_example` block is documentation for future real-model runs.

When using `device_map="auto"`, Transformers may shard the teacher across
devices. The wrapper moves clean `input_ids` and optional `attention_mask` to
the teacher input embedding device before calling the frozen teacher, and the
training loop moves returned teacher logits back to the student logits device
before loss computation. If a local HF smoke run still hits a placement issue,
`CUDA_VISIBLE_DEVICES=0` remains the simplest single-device workaround.

By default the HF teacher asks Transformers to load safetensors weights
(`use_safetensors=True`). If a legacy model only ships PyTorch `.bin`
checkpoints, prefer a safetensors variant of the model. Set
`use_safetensors: false` only when the model source is trusted and the runtime
uses `torch>=2.6`, or upgrade to `torch>=2.6` if Transformers reports the
`torch.load` checkpoint vulnerability error. Do not patch Transformers
internals to bypass that safety check.

## Stage 5C HuggingFace Teacher Smoke Training

The real teacher smoke path keeps the student mocked and does not implement
real Mamba. It loads a frozen HuggingFace causal-LM teacher, derives the mock
dataset and student vocab size from `teacher.model.config.vocab_size`, creates
an all-ones `attention_mask`, and calls:

```python
teacher(input_ids, attention_mask=attention_mask)
```

The teacher never receives `h_t`, `h'_t`, `h_delta_alt`, or any other student
state. `max_steps` counts optimizer steps; with
`--gradient-accumulation-steps 2`, one logged optimizer step consumes two
microbatches.

Example local-only smoke command:

```bash
python train.py \
  --config configs/train_config.yaml \
  --teacher-type hf \
  --student-type mock \
  --teacher-model-name-or-path /path/to/local/hf-causal-lm \
  --local-files-only \
  --max_steps 1 \
  --batch-size 1 \
  --seq-len 128 \
  --gradient-accumulation-steps 1 \
  --mixed-precision no \
  --csdm-weight 0.0
```

Tests for this path install a fake `transformers` module, so they run on CPU
without downloads, HF login, or external model weights. The existing mock
command remains unchanged:

```bash
python train.py --config configs/train_config.yaml --mock --max_steps 2
```

## Stage 5B Teacher Logit Cache

`TeacherLogitCache` stores frozen teacher outputs and is used by `train.py`
when `teacher_cache.enabled` or `--teacher-cache-enabled` is set. Mock
training does not require cache usage.

```python
from utils.logit_cache import LogitCacheConfig, TeacherLogitCache

cache = TeacherLogitCache(LogitCacheConfig(enabled=True, cache_dir="/tmp/teacher_logits"))
entry = cache.get_or_compute(
    input_ids,
    compute_fn=lambda input_ids, attention_mask=None: teacher(
        input_ids,
        attention_mask=attention_mask,
    ),
    attention_mask=attention_mask,
    extra={"teacher": "mock", "tokenizer": "mock-v1"},
)
```

Cache entries represent teacher outputs on clean token prefixes:
`teacher(input_ids, attention_mask) -> logits` for
`p_phi(y | x_{<=t})`. Cache keys include `input_ids`, optional
`attention_mask` content and shape, and canonical JSON `extra`. Use `extra`
only for teacher-output-affecting metadata such as teacher version, tokenizer
version, or prompt formatting. Do not include student states, `rho`, `sigma`,
`h_t`, `h_off`, `h_delta_alt`, adapter details, or other off-trajectory
student-side data.

Full logits must have shape `[B, T, V]`. In top-k mode the cache stores
`topk_values` and `topk_indices` and omits full logits; downstream top-k KD
needs explicit loss-side handling and is an approximation when full logits are
not retained.

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
