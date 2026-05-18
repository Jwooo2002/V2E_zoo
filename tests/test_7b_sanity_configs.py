from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "kd_topk": ROOT / "configs" / "experiments" / "train_7b_sanity_kd_topk.yaml",
    "csdm_topk": ROOT / "configs" / "experiments" / "train_7b_sanity_csdm_topk.yaml",
}
MATRIX_PATH = ROOT / "configs" / "ablations" / "7b_sanity_topk.yaml"


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def test_7b_sanity_experiment_configs_exist_and_parse() -> None:
    for name, path in EXPERIMENTS.items():
        assert path.is_file(), name
        config = _load_yaml(path)
        assert config["config"] == "configs/train_config.yaml"
        assert config["teacher_type"] == "hf"
        assert config["student_type"] == "mamba"
        assert config["teacher_model_name_or_path"] == "${CSDM_TEACHER_PATH}"
        assert config["tokenizer_name_or_path"] == "${CSDM_TOKENIZER_PATH}"
        assert config["dataset_type"] == "text"
        assert config["data_path"] == "${CSDM_DATA_PATH}"
        assert config["data_path"] != "data/smoke.txt"
        assert config["seq_len"] == 128
        assert config["batch_size"] == 1
        assert config["gradient_accumulation_steps"] == 16
        assert config["max_steps"] == 100
        assert config["student_hidden_size"] == 128
        assert config["student_num_layers"] == 4
        assert config["mixed_precision"] == "bf16"
        assert config["topk_enabled"] is True
        assert config["top_k"] == 256
        assert config["teacher_cache_enabled"] is False
        assert config["save_every_steps"] == 25
        assert config["save_at_end"] is True
        assert config["local_files_only"] is True
        assert config["use_safetensors"] is True
        assert config["trust_remote_code"] is False
        assert config["device_map"] == "auto"


def test_7b_sanity_variant_weights() -> None:
    kd = _load_yaml(EXPERIMENTS["kd_topk"])
    csdm = _load_yaml(EXPERIMENTS["csdm_topk"])

    assert kd["ce_weight"] == 0.2
    assert kd["kd_weight"] == 1.0
    assert kd["csdm_weight"] == 0.0
    assert kd["topk_enabled"] is True

    assert csdm["ce_weight"] == 0.2
    assert csdm["kd_weight"] == 1.0
    assert csdm["csdm_weight"] == 0.03
    assert csdm["topk_enabled"] is True


def test_7b_sanity_ablation_matrix_contains_topk_variants() -> None:
    matrix = _load_yaml(MATRIX_PATH)
    base = matrix["base"]
    variants = {variant["name"]: variant for variant in matrix["variants"]}

    assert set(variants) == {"kd_topk", "csdm_topk"}
    assert base["teacher_type"] == "hf"
    assert base["student_type"] == "mamba"
    assert base["teacher_model_name_or_path"] == "${CSDM_TEACHER_PATH}"
    assert base["tokenizer_name_or_path"] == "${CSDM_TOKENIZER_PATH}"
    assert base["data_path"] == "${CSDM_DATA_PATH}"
    assert base["dataset_type"] == "text"
    assert base["local_files_only"] is True
    assert base["teacher_cache_enabled"] is False
    assert base["max_steps"] == 100
    assert base["seq_len"] == 128
    assert base["gradient_accumulation_steps"] == 16
    assert base["topk_enabled"] is True
    assert base["top_k"] == 256
    assert variants["kd_topk"]["csdm_weight"] == 0.0
    assert variants["csdm_topk"]["csdm_weight"] == 0.03


def _run_7b(*args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_7b_sanity.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=timeout,
    )


def test_run_7b_sanity_csdm_topk_dry_run(monkeypatch, tmp_path: Path) -> None:
    for name in ("CSDM_TEACHER_PATH", "CSDM_TOKENIZER_PATH", "CSDM_DATA_PATH"):
        monkeypatch.delenv(name, raising=False)

    result = _run_7b(
        "--variant",
        "csdm_topk",
        "--base-output-dir",
        str(tmp_path),
        "--dry-run",
    )

    assert "CSDM_TEACHER_PATH=<unset>" in result.stdout
    assert "CSDM_TOKENIZER_PATH=<unset>" in result.stdout
    assert "CSDM_DATA_PATH=<unset>" in result.stdout
    assert "run_registered_experiment.py" in result.stdout
    assert "train_7b_sanity_csdm_topk.yaml" in result.stdout
    assert "train.py" in result.stdout
    assert "--teacher-type hf" in result.stdout
    assert "--student-type mamba" in result.stdout
    assert "CSDM_TEACHER_PATH" in result.stdout
    assert "CSDM_TOKENIZER_PATH" in result.stdout
    assert "CSDM_DATA_PATH" in result.stdout
    assert "--local-files-only" in result.stdout
    assert "--teacher-cache-enabled" not in result.stdout
    assert "--no-teacher-cache-enabled" in result.stdout
    assert "--max_steps 100" in result.stdout
    assert "--top-k 256" in result.stdout
    assert "--hf-device-map auto" in result.stdout
    assert "--use-safetensors" in result.stdout
    assert "--no-trust-remote-code" in result.stdout
    run_dir = Path(result.stdout.strip().splitlines()[-1])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "11A"
    assert manifest["metadata"]["status"] == "planned"


def test_run_7b_sanity_all_dry_run_and_cuda_visible_devices(monkeypatch, tmp_path: Path) -> None:
    for name in ("CSDM_TEACHER_PATH", "CSDM_TOKENIZER_PATH", "CSDM_DATA_PATH"):
        monkeypatch.delenv(name, raising=False)

    result = _run_7b(
        "--variant",
        "all",
        "--base-output-dir",
        str(tmp_path),
        "--dry-run",
        "--cuda-visible-devices",
        "0,1",
        "--seq-len",
        "64",
        "--top-k",
        "128",
        "--student-hidden-size",
        "96",
        "--student-num-layers",
        "3",
    )

    assert "CUDA_VISIBLE_DEVICES=0,1" in result.stdout
    assert result.stdout.count("run_registered_experiment.py") == 2
    assert result.stdout.count("train.py") == 2
    assert "train_7b_sanity_kd_topk.yaml" in result.stdout
    assert "train_7b_sanity_csdm_topk.yaml" in result.stdout
    assert "--seq-len 64" in result.stdout
    assert "--top-k 128" in result.stdout
    assert "--student-hidden-size 96" in result.stdout
    assert "--student-num-layers 3" in result.stdout


def test_run_7b_sanity_allow_downloads_removes_local_files_only(monkeypatch, tmp_path: Path) -> None:
    for name in ("CSDM_TEACHER_PATH", "CSDM_TOKENIZER_PATH", "CSDM_DATA_PATH"):
        monkeypatch.delenv(name, raising=False)

    result = _run_7b(
        "--variant",
        "kd_topk",
        "--base-output-dir",
        str(tmp_path),
        "--dry-run",
        "--allow-downloads",
    )

    train_line = [line for line in result.stdout.splitlines() if "train.py" in line][-1]
    assert "--local-files-only" not in train_line


def test_run_7b_sanity_requires_env_for_real_execution(monkeypatch, tmp_path: Path) -> None:
    for name in ("CSDM_TEACHER_PATH", "CSDM_TOKENIZER_PATH", "CSDM_DATA_PATH"):
        monkeypatch.delenv(name, raising=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_7b_sanity.py",
            "--variant",
            "kd_topk",
            "--base-output-dir",
            str(tmp_path),
            "--no-with-perturbation",
            "--no-with-report",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 2
    assert "Missing required environment variable" in result.stderr
    assert "train.py" not in result.stdout


def test_7b_sanity_ablation_matrix_dry_run(monkeypatch, tmp_path: Path) -> None:
    for name in ("CSDM_TEACHER_PATH", "CSDM_TOKENIZER_PATH", "CSDM_DATA_PATH"):
        monkeypatch.delenv(name, raising=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ablation_matrix.py",
            "--matrix",
            "configs/ablations/7b_sanity_topk.yaml",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    assert "kd_topk:" in result.stdout
    assert "csdm_topk:" in result.stdout
    assert "--teacher-type hf" in result.stdout
    assert "--student-type mamba" in result.stdout
    assert "--local-files-only" in result.stdout
    assert "--no-teacher-cache-enabled" in result.stdout
    assert "--top-k 256" in result.stdout
    assert "CSDM_TEACHER_PATH" in result.stdout
    assert not (tmp_path / "kd_topk").exists()
    assert not (tmp_path / "csdm_topk").exists()
