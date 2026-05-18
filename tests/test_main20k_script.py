from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_main20k_script_has_long_run_guardrails() -> None:
    script = ROOT / "scripts" / "run_main20k.sh"
    text = script.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script)], cwd=ROOT, check=True)

    assert "/mnt/sda2/csdm_main20k_kd" in text
    assert "/mnt/sda2/csdm_main20k_csdm_w01" in text
    assert "/mnt/sda2/csdm_main20k_csdm_topk_w003" in text
    assert "CSDM_STORAGE_MIN_FREE_GB=${CSDM_STORAGE_MIN_FREE_GB:-20}" in text
    assert "CSDM_ARTIFACT_HEALTH_CACHE_SAMPLE_SIZE=${CSDM_ARTIFACT_HEALTH_CACHE_SAMPLE_SIZE:-64}" in text
    assert text.count("--override storage_min_free_gb=") == 3
    assert text.count("--artifact-health-check") == 3
    assert text.count("--artifact-health-cache-sample-size") == 3
    assert text.count("--no-timeout") == 3
    assert "CUDA_VISIBLE_DEVICES=0" in text
    assert "CUDA_VISIBLE_DEVICES=1" in text
    assert 'wait "$PID_KD"' in text
    assert 'wait "$PID_CSDM"' in text
    assert "RC_KD=$?" in text
    assert "RC_CSDM=$?" in text
    assert "stopping before top-k" in text
    assert "RC_TOPK=$?" in text
