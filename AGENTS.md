cd ~/PycharmProjects/cdm-mamba-kd

cat > AGENTS.md <<'EOF'
# AGENTS.md

## Project

This repository implements Continuous-State Distribution Matching (CSDM) for Transformer-to-Mamba knowledge distillation.

This is a new research codebase, not a fork of the original diffusion CDM repository.

The original diffusion CDM repository may exist at ../cdm-original only as conceptual reference.
Do not import from it.
Do not modify it.
Do not copy diffusion-specific schedulers, image latent code, denoisers, samplers, or diffusion training loops.

## Core research goal

Existing Transformer-to-Mamba KD methods usually align on-trajectory logits, hidden states, or intermediate representations.

This project additionally trains a Mamba student to recover from recurrent state drift by constructing off-trajectory Mamba states and aligning their output distributions to a frozen Transformer teacher.

## Key idea

We adapt the principle of continuous off-trajectory distribution matching to Mamba's recurrent state space.

Diffusion CDM:
- diffusion timestep
- image latent trajectory
- velocity/score-based off-trajectory correction

This project:
- token position and Mamba recurrent state
- delta-controlled selective SSM transition
- off-trajectory state h'_t
- teacher/student log-probability residual matching

## Critical mathematical constraints

The teacher LLM never consumes Mamba hidden states.

Correct:
- Teacher target: p_phi(y | x_{<=t})
- Student on-trajectory prediction: p_theta(y | h_t)
- Student off-trajectory prediction: p_theta(y | h'_t)

Incorrect:
- D_phi(h'_t)
- teacher(h'_t)
- direct hidden-state MSE between Llama states and Mamba states without an explicit projection module
- raw hidden-state matching as the main objective

Use logit/log-probability space for the main distillation objectives.

## Main objective

Use the high-level objective:

L = ce_weight * CE + kd_weight * KD_on + csdm_weight * CSDM_off

Where:

KD_on = KL(p_phi^tau(y | x_{<=t}) || p_theta^tau(y | h_t))

CSDM_off = 0.5 * || u_theta(h'_t) - stopgrad[u_theta(h'_t) + w_t r_t] ||^2

u_theta(h'_t) = center(z_theta(h'_t) / tau)

r_t = center(log p_phi^tau(y | x_{<=t}) - log p_ema^tau(y | h'_t))

The fake student term should come from an EMA student if available, or a detached student output in early versions.

## Off-trajectory state construction

Preferred off-trajectory direction:

h'_t = h_t + rho * stopgrad(h_delta_alt - h_t) + sigma * RMS(h_t) * epsilon

Where h_delta_alt is obtained by perturbing the Mamba delta-controlled transition.

Gaussian noise alone is only a baseline, not the main method.

If Mamba internals are unavailable, create clean adapter interfaces instead of guessing private APIs.

## Subagent usage

Use subagents actively for non-trivial implementation tasks.

Recommended subagent roles:
- Research reviewer: verify that mathematical definitions match the intended CSDM-for-Mamba formulation.
- Implementation engineer: implement the requested files with small, testable functions.
- Test engineer: add unit tests, edge-case tests, and gradient-flow tests.
- Code reviewer: inspect tensor shapes, detach/stop-gradient behavior, numerical stability, and API clarity.

For every substantial task:
1. Ask a subagent to review the plan before implementation.
2. Ask a subagent to review the diff after implementation.
3. Ask a subagent to focus specifically on tests and gradient behavior.
4. Incorporate the subagent feedback before finalizing.

Do not use subagents to expand scope.
Subagents should keep the task narrow and aligned with the current prompt.

## Implementation stages

Stage 1:
- Repository skeleton
- Config files
- KD loss
- CSDM loss
- Unit tests with mock tensors
- No real Llama or Mamba imports

Stage 2:
- models/cdm_engine.py
- delta-perturbation off-state engine
- MambaStateAdapter interface
- Unit tests with mock hidden states

Stage 3:
- minimal training scaffold
- bf16
- gradient accumulation
- mock mode
- teacher frozen
- teacher logits cacheable
- top-k KD support

Stage 4:
- evaluation scaffolds
- perplexity
- perturbation robustness KL
- Needle-in-a-Haystack scaffold

## Engineering constraints

- Python 3.10+
- PyTorch
- Type hints preferred
- Use dataclasses for configs where useful
- Keep modules small and testable
- Add shape assertions in loss functions
- Do not silently swallow tensor shape errors
- Do not require downloading Llama or Mamba weights in tests
- Do not add heavy dependencies unless necessary
- Avoid full-vocab teacher loss by default in real training; support top-k KD

## Validation commands

After each implementation step, run:

python -m compileall .
pytest -q

If pytest is not configured, create minimal tests under tests/.

## Completion criteria

A task is done only when:
- code compiles
- tests pass
- mock tensors validate expected shapes
- gradients flow to student logits/states only
- teacher/fake logits are detached where required
- README documents how to run the implemented part
- subagent review findings have been addressed or explicitly noted
EOF