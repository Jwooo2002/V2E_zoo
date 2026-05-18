from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "kd": ROOT / "configs" / "experiments" / "train_real_small_5k_kd.yaml",
    "csdm_w01": ROOT / "configs" / "experiments" / "train_real_small_5k_csdm_w01.yaml",
    "csdm_topk_w003": ROOT / "configs" / "experiments" / "train_real_small_5k_csdm_topk_w003.yaml",
}
MATRIX_PATH = ROOT / "configs" / "ablations" / "real_small_5k_selected.yaml"


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def test_real_small_5k_experiment_configs_exist_and_parse() -> None:
    for name, path in EXPERIMENTS.items():
        assert path.is_file(), name
        config = _load_yaml(path)
        assert config["config"] == "configs/train_config.yaml"
        assert config["teacher_type"] == "hf"
        assert config["student_type"] == "mamba"
        assert config["teacher_model_name_or_path"] == "sshleifer/tiny-gpt2"
        assert config["tokenizer_name_or_path"] == "sshleifer/tiny-gpt2"
        assert config["dataset_type"] == "text"
        assert config["data_path"] == "${CSDM_DATA_PATH}"
        assert config["data_path"] != "data/smoke.txt"
        assert config["seq_len"] == 128
        assert config["batch_size"] == 1
        assert config["gradient_accumulation_steps"] == 16
        assert config["max_steps"] == 5000
        assert config["student_hidden_size"] == 128
        assert config["student_num_layers"] == 4
        assert config["mixed_precision"] == "bf16"
        assert config["teacher_cache_enabled"] is True
        assert "teacher_cache_use_top_k" not in config
        assert config["save_every_steps"] == 500
        assert config["save_at_end"] is True
        assert config["local_files_only"] is True


def test_real_small_5k_variant_weights_are_selected_candidates() -> None:
    kd = _load_yaml(EXPERIMENTS["kd"])
    csdm = _load_yaml(EXPERIMENTS["csdm_w01"])
    topk = _load_yaml(EXPERIMENTS["csdm_topk_w003"])

    assert kd["ce_weight"] == 0.2
    assert kd["kd_weight"] == 1.0
    assert kd["csdm_weight"] == 0.0
    assert kd["topk_enabled"] is False

    assert csdm["ce_weight"] == 0.2
    assert csdm["kd_weight"] == 1.0
    assert csdm["csdm_weight"] == 0.1
    assert csdm["topk_enabled"] is False

    assert topk["ce_weight"] == 0.2
    assert topk["kd_weight"] == 1.0
    assert topk["csdm_weight"] == 0.03
    assert topk["topk_enabled"] is True
    assert topk["top_k"] == 128


def test_real_small_5k_ablation_matrix_contains_selected_variants() -> None:
    matrix = _load_yaml(MATRIX_PATH)
    base = matrix["base"]
    variants = {variant["name"]: variant for variant in matrix["variants"]}

    assert set(variants) == {"kd", "csdm_w01", "csdm_topk_w003"}
    assert base["dataset_type"] == "text"
    assert base["data_path"] == "${CSDM_DATA_PATH}"
    assert base["max_steps"] == 5000
    assert base["gradient_accumulation_steps"] == 16
    assert base["local_files_only"] is True
    assert base["save_every_steps"] == 500
    assert variants["kd"]["csdm_weight"] == 0.0
    assert variants["csdm_w01"]["csdm_weight"] == 0.1
    assert variants["csdm_topk_w003"]["csdm_weight"] == 0.03
    assert variants["csdm_topk_w003"]["topk_enabled"] is True


def _run_5k(*args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_real_small_5k.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=timeout,
    )


def test_run_real_small_5k_kd_dry_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CSDM_DATA_PATH", raising=False)

    result = _run_5k(
        "--variant",
        "kd",
        "--base-output-dir",
        str(tmp_path),
        "--dry-run",
    )

    assert "run_registered_experiment.py" in result.stdout
    assert "train_real_small_5k_kd.yaml" in result.stdout
    assert "train.py" in result.stdout
    assert "--max_steps 5000" in result.stdout
    assert "--local-files-only" in result.stdout
    assert "--data-path '${CSDM_DATA_PATH}'" in result.stdout or "--data-path \\$\\{CSDM_DATA_PATH\\}" in result.stdout
    assert "--no-topk-enabled" in result.stdout
    run_dir = Path(result.stdout.strip().splitlines()[-1])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "10B"
    assert manifest["metadata"]["status"] == "planned"


def test_run_real_small_5k_all_dry_run_and_cuda_visible_devices(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CSDM_DATA_PATH", raising=False)

    result = _run_5k(
        "--variant",
        "all",
        "--base-output-dir",
        str(tmp_path),
        "--dry-run",
        "--cuda-visible-devices",
        "0",
    )

    assert "CUDA_VISIBLE_DEVICES=0" in result.stdout
    assert result.stdout.count("run_registered_experiment.py") == 3
    assert result.stdout.count("train.py") == 3
    assert "train_real_small_5k_kd.yaml" in result.stdout
    assert "train_real_small_5k_csdm_w01.yaml" in result.stdout
    assert "train_real_small_5k_csdm_topk_w003.yaml" in result.stdout
    assert "--csdm-weight 0.1" in result.stdout
    assert "--csdm-weight 0.03" in result.stdout
    assert "--topk-enabled" in result.stdout


def test_run_real_small_5k_overrides_and_no_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CSDM_DATA_PATH", raising=False)

    result = _run_5k(
        "--variant",
        "csdm_topk_w003",
        "--base-output-dir",
        str(tmp_path),
        "--dry-run",
        "--max-steps",
        "12",
        "--no-timeout",
        "--storage-min-free-gb",
        "20",
        "--artifact-health-check",
        "--artifact-health-max-files",
        "10",
        "--artifact-health-cache-sample-size",
        "5",
        "--override",
        "teacher_cache_dir=/tmp/custom_cache",
    )

    assert "--no-timeout" in result.stdout
    assert "--max_steps 12" in result.stdout
    assert "--storage-min-free-gb 20" in result.stdout
    assert "--artifact-health-check" in result.stdout
    assert "--artifact-health-max-files 10" in result.stdout
    assert "--artifact-health-cache-sample-size 5" in result.stdout
    assert "--teacher-cache-dir /tmp/custom_cache" in result.stdout
    run_dir = Path(result.stdout.strip().splitlines()[-1])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["timeout_disabled"] is True


def test_real_small_5k_ablation_matrix_dry_run(monkeypatch) -> None:
    monkeypatch.delenv("CSDM_DATA_PATH", raising=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ablation_matrix.py",
            "--matrix",
            "configs/ablations/real_small_5k_selected.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    assert "kd:" in result.stdout
    assert "csdm_w01:" in result.stdout
    assert "csdm_topk_w003:" in result.stdout
    assert "--max_steps 5000" in result.stdout
    assert "--save-every-steps 500" in result.stdout
    assert "--local-files-only" in result.stdout
    assert "--data-path '${CSDM_DATA_PATH}'" in result.stdout or "--data-path \\$\\{CSDM_DATA_PATH\\}" in result.stdout
