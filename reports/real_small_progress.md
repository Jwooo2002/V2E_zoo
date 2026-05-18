# Real-Small Progress Report

Generated: 2026-05-18T02:55:46.837790+09:00

## Artifact Roots Inspected

- `/mnt/sda2/csdm_tmp_archive_20260514/`
- `/mnt/sda2/csdm_main20k_kd/`
- `/mnt/sda2/csdm_main20k_csdm_w01/`
- `/mnt/sda2/csdm_main20k_csdm_topk_w003/`
- `/tmp fallback paths (not preferred)`

## Git

- commit: `d71f365905e88aaae776080c68c8e74023e5669c`
- branch: `master`
- dirty: `True`

## Environment Summary

- causal_conv1d_version: `1.6.2.post1`
- cuda_available: `True`
- gpu_count: `1`
- mamba_ssm_version: `2.2.4`
- python: `3.10.20`
- torch_cuda: `12.1`
- torch_version: `2.5.1+cu121`
- transformers_version: `4.48.3`

## Main Summary Table

| run_group | variant | status | step | total | ce | kd | csdm | full_delta_kl | full_kl_on | full_kl_off | topk_delta_kl | topk_kl_on | topk_kl_off | checkpoint_loaded | artifact_health_ok | checkpoint_path | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| csdm_real_small_5k_kd | kd | complete | 5000 | 0.7627075166 | 3.379812524 | 0.08674499718 | 0 | 0.0001658591209 | 0.02349716536 | 0.02366302448 | 0.001726378687 | 0.3364105923 | 0.338136971 | True |  | /mnt/sda2/csdm_tmp_archive_20260514/csdm_real_small_5k_kd/stage10b_kd_1778652890203950132/checkpoints/checkpoint_step_5000_opt_5000.pt | /mnt/sda2/csdm_tmp_archive_20260514/csdm_real_small_5k_kd/stage10b_kd_1778652890203950132 |
| csdm_real_small_5k_csdm | csdm_w01 | complete | 5000 | 0.7615649588 | 3.37462759 | 0.08659723355 | 0.0004219775929 | 0.0001571374596 | 0.02330325957 | 0.02346039703 | 0.001773662865 | 0.3366387049 | 0.3384123677 | True |  | /mnt/sda2/csdm_tmp_archive_20260514/csdm_real_small_5k_csdm/stage10b_csdm_w01_1778661581959230886/checkpoints/checkpoint_step_5000_opt_5000.pt | /mnt/sda2/csdm_tmp_archive_20260514/csdm_real_small_5k_csdm/stage10b_csdm_w01_1778661581959230886 |
| csdm_real_small_5k_topk | csdm_topk_w003 | user_observed_artifact_missing | 5000 |  |  |  |  | 0.0001141112298 | 0.08375097928 | 0.08386509051 | 5.568517372e-05 | 0.01514217284 | 0.01519785801 | observed_true_but_artifact_missing |  |  | missing |
| csdm_main20k_kd | kd | complete | 20000 | 0.6674558148 | 2.913377106 | 0.08478038106 | 0 | 4.722467727e-05 | 0.02133544224 | 0.02138266692 | 0.001017759244 | 0.3662943376 | 0.3673120969 | True |  | /mnt/sda2/csdm_main20k_kd/run_20260514_110827_16747f/checkpoints/checkpoint_step_20000_opt_20000.pt | /mnt/sda2/csdm_main20k_kd/run_20260514_110827_16747f |
| csdm_main20k_csdm_w01 | csdm_w01 | complete | 20000 | 0.6675064936 | 2.911852181 | 0.08509506797 | 0.0004097290221 | 4.250763191e-05 | 0.02135313965 | 0.02139564728 | 0.0009979142083 | 0.3667226036 | 0.3677205178 | True |  | /mnt/sda2/csdm_main20k_csdm_w01/run_20260514_110827_cc658a/checkpoints/checkpoint_step_20000_opt_20000.pt | /mnt/sda2/csdm_main20k_csdm_w01/run_20260514_110827_cc658a |
| csdm_main20k_csdm_topk_w003 | csdm_topk_w003 | complete | 20000 | 0.7840148285 | 3.588365495 | 0.06633061776 | 0.0003699012559 | 0.0003313397368 | 0.08040751848 | 0.08073885822 | 9.612304469e-05 | 0.01630736618 | 0.01640348923 | True | True | /mnt/sda2/csdm_main20k_csdm_topk_w003/run_20260517_094626_ec7bbd/checkpoints/checkpoint_step_20000_opt_20000.pt | /mnt/sda2/csdm_main20k_csdm_topk_w003/run_20260517_094626_ec7bbd |
| csdm_real_small_20k_kd | kd | failed_legacy_superseded | 138 | 1.550740354 | 7.656205297 | 0.01949927164 | 0 |  |  |  |  |  |  |  |  |  | /mnt/sda2/csdm_tmp_archive_20260514/csdm_real_small_20k_kd/stage10b_kd_1778668793934459842 |
| csdm_real_small_20k_csdm_w01 | csdm_w01 | failed_legacy_superseded | 115 | 1.609092571 | 7.961152583 | 0.0168461951 | 0.0001585916716 |  |  |  |  |  |  |  |  |  | /mnt/sda2/csdm_tmp_archive_20260514/csdm_real_small_20k_csdm_w01/stage10b_csdm_w01_1778668808148763482 |

## 1k Comparison
- No current 1k artifact-backed row is included in this refresh; use prior pilot notes if needed, but do not mix non-identical setups without labeling them.

## 5k Comparison

- KD artifact-backed: full_vocab delta_kl `0.0001658591209`, kl_on `0.02349716536`, kl_off `0.02366302448`.
- CSDM w=0.1 artifact-backed: full_vocab delta_kl `0.0001571374596`, kl_on `0.02330325957`, kl_off `0.02346039703`.
- CSDM+top-k w=0.03 user-observed only: topk delta_kl `5.568517372e-05`, kl_on `0.01514217284`, kl_off `0.01519785801`. The artifact is currently missing; do not treat this as artifact-backed.

## 20k Comparison

- KD 20k artifact-backed: full_vocab delta_kl `4.722467727e-05`, kl_on `0.02133544224`, kl_off `0.02138266692`; topk delta_kl `0.001017759244`, topk kl_on `0.3662943376`, topk kl_off `0.3673120969`.
- CSDM w=0.1 20k artifact-backed: full_vocab delta_kl `4.250763191e-05`, kl_on `0.02135313965`, kl_off `0.02139564728`; topk delta_kl `0.0009979142083`, topk kl_on `0.3667226036`, topk kl_off `0.3677205178`.
- CSDM+top-k w=0.03 20k artifact-backed: full_vocab delta_kl `0.0003313397368`, kl_on `0.08040751848`, kl_off `0.08073885822`; topk delta_kl `9.612304469e-05`, topk kl_on `0.01630736618`, topk kl_off `0.01640348923`.

## Interpretation

- Artifact-backed 20k full-vocab comparison: CSDM w=0.1 has slightly lower full-vocab delta_kl than KD, but KD remains slightly better on full-vocab kl_on and kl_off. Treat this as a robustness-delta improvement, not a broad full-vocab win.
- Artifact-backed 20k top-k comparison: CSDM+top-k w=0.03 is much stronger on selected-vocab/top-k KL than KD and CSDM w=0.1.
- CSDM+top-k w=0.03 sacrifices full-vocab KL: its full_vocab kl_on/kl_off are much worse than KD and CSDM w=0.1, so report it as a selected-vocab/top-k objective win, not a full-vocab win.
- Artifact-backed 5k full-vocab comparison remains favorable to CSDM w=0.1 over KD on kl_on, kl_off, and delta_kl.
- The 5k CSDM+top-k w=0.03 metrics are user-observed only; the artifact is missing, so those numbers are not reproducible from current disk and should not be used in paper tables until rerun or recovered.
- No run artifact was erased: the completed top-k 20k run is useful for the top-k objective, artifact health passed, and no .pth cleanup target exists in the tracked run roots.

## Recommended Next Action

- Use the 20k top-k result as the selected-vocab/top-k candidate; keep the full-vocab KD/CSDM w=0.1 comparison separate.
- Before moving to a much larger teacher, decide whether the next experiment is optimizing full-vocab behavior or top-k behavior; the evidence points in different directions.
- If a paper table needs the 5k top-k result, rerun CSDM+top-k w=0.03 5k and archive it properly, or recover the missing run directory from backup.
