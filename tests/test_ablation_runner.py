from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "cdm_mamba_kd_ablation_runner",
    ROOT / "scripts" / "run_ablation_matrix.py",
)
assert RUNNER_SPEC is not None
assert RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)


def _write_matrix(path: Path, *, include_failure: bool = False, include_mamba: bool = False) -> None:
    variants = [
        """
  - name: ce_only
    ce_weight: 1.0
    kd_weight: 0.0
    csdm_weight: 0.0
    topk_enabled: false
""",
        """
  - name: ce_kd
    ce_weight: 0.2
    kd_weight: 1.0
    csdm_weight: 0.0
    topk_enabled: false
""",
    ]
    if include_failure:
        variants.insert(
            0,
            """
  - name: bad_max_steps
    max_steps: 0
    ce_weight: 1.0
    kd_weight: 0.0
    csdm_weight: 0.0
""",
        )
    if include_mamba:
        variants.append(
            """
  - name: optional_mamba
    requires_mamba: true
    mock: false
    student_type: mamba
    teacher_type: mock
    student_vocab_size: 128
    student_hidden_size: 64
    student_num_layers: 2
    ce_weight: 0.2
    kd_weight: 1.0
    csdm_weight: 0.0
""",
        )
    path.write_text(
        """
experiment_name: test_ablation
base:
  config: configs/train_config.yaml
  mock: true
  teacher_type: mock
  student_type: mock
  dataset_type: mock
  max_steps: 1
  seq_len: 8
  batch_size: 1
  gradient_accumulation_steps: 1
  mixed_precision: "no"
  teacher_cache_enabled: false
  save_at_end: false
variants:
"""
        + "".join(variants),
        encoding="utf-8",
    )


def test_dry_run_parses_matrix_and_prints_commands() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ablation_matrix.py",
            "--matrix",
            "configs/ablations/csdm_mamba_smoke.yaml",
            "--dry-run",
            "--only",
            "ce_only",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )

    assert result.stderr == ""
    assert "ce_only:" in result.stdout
    assert "train.py" in result.stdout
    assert "--ce-weight 1.0" in result.stdout


def test_only_skip_and_override_are_applied(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path)
    matrix = runner.load_matrix(matrix_path)

    only_configs = runner.build_variant_configs(
        matrix,
        output_dir=tmp_path / "out",
        only=["ce_kd"],
        overrides=["max_steps=3", "topk_enabled=true"],
    )
    skipped_configs = runner.build_variant_configs(matrix, output_dir=tmp_path / "out", skip=["ce_only"])

    assert [config["name"] for config in only_configs] == ["ce_kd"]
    assert only_configs[0]["max_steps"] == 3
    assert only_configs[0]["topk_enabled"] is True
    assert [config["name"] for config in skipped_configs] == ["ce_kd"]


def test_unknown_variant_selection_raises(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path)
    matrix = runner.load_matrix(matrix_path)

    with pytest.raises(ValueError, match="Unknown --only"):
        runner.build_variant_configs(matrix, output_dir=tmp_path / "out", only=["missing"])


def test_parse_json_lines_extracts_final_metrics() -> None:
    stdout = """
not-json
{"step": 1, "total": 3.0, "ce": 2.0, "kd": 1.0, "csdm": 0.0, "optimizer_step": 1}
{"step": 2, "total": 2.5, "ce": 1.5, "kd": 1.0, "csdm": 0.0, "grad_norm": 0.7, "optimizer_step": 2}
"""
    metrics = runner.extract_final_metrics(stdout)

    assert metrics["total"] == 2.5
    assert metrics["optimizer_step"] == 2
    assert metrics["grad_norm"] == 0.7


def test_mock_matrix_executes_and_writes_summary(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    output_dir = tmp_path / "ablation_out"
    _write_matrix(matrix_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ablation_matrix.py",
            "--matrix",
            str(matrix_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )

    assert "summary_json=" in result.stdout
    summary_json = output_dir / "ablation_summary.json"
    summary_csv = output_dir / "ablation_summary.csv"
    assert summary_json.is_file()
    assert summary_csv.is_file()

    records = json.loads(summary_json.read_text(encoding="utf-8"))
    assert [record["name"] for record in records] == ["ce_only", "ce_kd"]
    assert all(record["status"] == "success" for record in records)
    assert all(record["metrics"]["optimizer_step"] == 1 for record in records)
    assert (output_dir / "logs" / "ce_only.stdout").is_file()
    assert (output_dir / "logs" / "ce_kd.stderr").is_file()

    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["name"] for row in rows] == ["ce_only", "ce_kd"]
    assert rows[0]["status"] == "success"


def test_continue_on_error_records_failed_and_successful_variants(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    output_dir = tmp_path / "ablation_out"
    _write_matrix(matrix_path, include_failure=True)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ablation_matrix.py",
            "--matrix",
            str(matrix_path),
            "--output-dir",
            str(output_dir),
            "--continue-on-error",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert result.returncode != 0
    records = json.loads((output_dir / "ablation_summary.json").read_text(encoding="utf-8"))
    assert [record["name"] for record in records] == ["bad_max_steps", "ce_only", "ce_kd"]
    assert records[0]["status"] == "failed"
    assert records[1]["status"] == "success"
    assert records[2]["status"] == "success"


def test_mamba_variant_is_skipped_when_dependency_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, include_mamba=True)
    matrix = runner.load_matrix(matrix_path)
    monkeypatch.setattr(runner, "mamba_ssm_available", lambda: False)

    records = runner.run_matrix(
        matrix,
        output_dir=tmp_path / "ablation_out",
        only=["optional_mamba"],
        continue_on_error=True,
    )

    assert records[0]["status"] == "skipped"
    assert "mamba_ssm is unavailable" in records[0]["reason"]


def test_skip_reason_marks_variant_skipped_without_execution(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        """
experiment_name: skip_reason
base:
  config: configs/train_config.yaml
  mock: true
  max_steps: 1
variants:
  - name: declared_future
    skip_reason: not implemented in this stage
    ce_weight: 1.0
    kd_weight: 0.0
    csdm_weight: 0.0
""",
        encoding="utf-8",
    )
    matrix = runner.load_matrix(matrix_path)

    records = runner.run_matrix(matrix, output_dir=tmp_path / "out")

    assert records[0]["status"] == "skipped"
    assert records[0]["returncode"] is None
    assert records[0]["reason"] == "not implemented in this stage"


def test_runner_rejects_unknown_matrix_keys(tmp_path: Path) -> None:
    matrix_path = tmp_path / "bad.yaml"
    matrix_path.write_text(
        """
base:
  config: configs/train_config.yaml
variants:
  - name: bad
    surprise: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported variant"):
        runner.load_matrix(matrix_path)
