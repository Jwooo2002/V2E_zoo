from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.run_ablation_matrix import load_matrix
from scripts.run_small_experiment import load_experiment, validate_execution_paths


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "kd": ROOT / "configs" / "experiments" / "train_real_small_kd.yaml",
    "csdm": ROOT / "configs" / "experiments" / "train_real_small_csdm.yaml",
    "csdm_topk": ROOT / "configs" / "experiments" / "train_real_small_csdm_topk.yaml",
}
MATRIX_PATH = ROOT / "configs" / "ablations" / "real_small_training.yaml"


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def test_real_small_experiment_configs_exist_and_parse() -> None:
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
        assert config["max_steps"] == 1000
        assert config["student_hidden_size"] == 128
        assert config["student_num_layers"] == 4
        assert config["teacher_cache_enabled"] is True
        assert config["save_every_steps"] == 100
        assert config["save_at_end"] is True
        assert config["local_files_only"] is True


def test_real_small_variant_weights_are_correct() -> None:
    kd = _load_yaml(EXPERIMENTS["kd"])
    csdm = _load_yaml(EXPERIMENTS["csdm"])
    topk = _load_yaml(EXPERIMENTS["csdm_topk"])

    assert kd["ce_weight"] == 0.2
    assert kd["kd_weight"] == 1.0
    assert kd["csdm_weight"] == 0.0
    assert kd["topk_enabled"] is False

    assert csdm["ce_weight"] == 0.2
    assert csdm["kd_weight"] == 1.0
    assert csdm["csdm_weight"] == 0.03
    assert csdm["topk_enabled"] is False

    assert topk["ce_weight"] == 0.2
    assert topk["kd_weight"] == 1.0
    assert topk["csdm_weight"] == 0.03
    assert topk["topk_enabled"] is True
    assert topk["top_k"] == 128


def test_real_small_ablation_matrix_contains_expected_variants() -> None:
    matrix = load_matrix(MATRIX_PATH)
    base = matrix["base"]
    variants = {variant["name"]: variant for variant in matrix["variants"]}

    assert set(variants) == {"kd", "csdm", "csdm_topk"}
    assert base["dataset_type"] == "text"
    assert base["data_path"] == "${CSDM_DATA_PATH}"
    assert base["data_path"] != "data/smoke.txt"
    assert base["local_files_only"] is True
    assert base["student_hidden_size"] == 128
    assert base["student_num_layers"] == 4
    assert variants["kd"]["csdm_weight"] == 0.0
    assert variants["csdm"]["csdm_weight"] == 0.03
    assert variants["csdm_topk"]["topk_enabled"] is True


def test_environment_variable_expansion_is_path_scoped(monkeypatch, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n", encoding="utf-8")
    monkeypatch.setenv("CSDM_DATA_PATH", str(corpus))

    config = load_experiment(EXPERIMENTS["kd"], [])

    assert config["data_path"] == str(corpus)
    assert config["teacher_model_name_or_path"] == "sshleifer/tiny-gpt2"
    assert config["tokenizer_name_or_path"] == "sshleifer/tiny-gpt2"
    validate_execution_paths(config)


def test_ablation_matrix_expands_data_path_env_var(monkeypatch, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n", encoding="utf-8")
    monkeypatch.setenv("CSDM_DATA_PATH", str(corpus))

    matrix = load_matrix(MATRIX_PATH)

    assert matrix["base"]["data_path"] == str(corpus)


def test_unresolved_data_path_fails_only_for_execution(monkeypatch) -> None:
    monkeypatch.delenv("CSDM_DATA_PATH", raising=False)
    config = load_experiment(EXPERIMENTS["kd"], [])

    assert config["data_path"] == "${CSDM_DATA_PATH}"
    try:
        validate_execution_paths(config)
    except ValueError as exc:
        assert "CSDM_DATA_PATH" in str(exc)
    else:
        raise AssertionError("validate_execution_paths should reject unresolved data path for execution.")


def test_missing_data_path_file_fails_for_execution(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    config = {
        "dataset_type": "text",
        "data_path": str(missing),
    }

    try:
        validate_execution_paths(config)
    except ValueError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("validate_execution_paths should reject missing data file for execution.")


def test_run_small_experiment_non_dry_rejects_unresolved_data_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("CSDM_DATA_PATH", raising=False)
    config_path = tmp_path / "real_text.yaml"
    config_path.write_text(
        "\n".join(
            [
                "config: configs/train_config.yaml",
                "teacher_type: mock",
                "student_type: mock",
                "dataset_type: text",
                "data_path: ${CSDM_DATA_PATH}",
                "max_steps: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/run_small_experiment.py", "--experiment", str(config_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode != 0
    assert "CSDM_DATA_PATH" in result.stderr


def test_run_small_experiment_dry_run_allows_unresolved_data_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("CSDM_DATA_PATH", raising=False)
    config_path = tmp_path / "real_text.yaml"
    config_path.write_text(
        "\n".join(
            [
                "config: configs/train_config.yaml",
                "teacher_type: mock",
                "student_type: mock",
                "dataset_type: text",
                "data_path: ${CSDM_DATA_PATH}",
                "max_steps: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/run_small_experiment.py", "--experiment", str(config_path), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    assert "--data-path '${CSDM_DATA_PATH}'" in result.stdout or "--data-path \\$\\{CSDM_DATA_PATH\\}" in result.stdout


def test_registered_experiment_dry_run_real_small_configs_without_data_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("CSDM_DATA_PATH", raising=False)
    for path in EXPERIMENTS.values():
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_registered_experiment.py",
                "--experiment",
                str(path.relative_to(ROOT)),
                "--base-output-dir",
                str(tmp_path),
                "--dry-run",
                "--override",
                "max_steps=10",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
        assert "--teacher-type hf" in result.stdout
        assert "--student-type mamba" in result.stdout
        assert "--data-path '${CSDM_DATA_PATH}'" in result.stdout or "--data-path \\$\\{CSDM_DATA_PATH\\}" in result.stdout
        assert "--local-files-only" in result.stdout
        assert "--max_steps 10" in result.stdout


def test_real_small_ablation_matrix_dry_run_without_data_path(monkeypatch) -> None:
    monkeypatch.delenv("CSDM_DATA_PATH", raising=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ablation_matrix.py",
            "--matrix",
            "configs/ablations/real_small_training.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    assert "kd:" in result.stdout
    assert "csdm:" in result.stdout
    assert "csdm_topk:" in result.stdout
    assert "--data-path '${CSDM_DATA_PATH}'" in result.stdout or "--data-path \\$\\{CSDM_DATA_PATH\\}" in result.stdout
    assert "--student-hidden-size 128" in result.stdout
    assert "--gradient-accumulation-steps 16" in result.stdout
