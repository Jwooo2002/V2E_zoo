from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

PILOT_CONFIGS = {
    "ce": ROOT / "configs" / "pilots" / "tiny_real_ce.yaml",
    "kd": ROOT / "configs" / "pilots" / "tiny_real_kd.yaml",
    "csdm": ROOT / "configs" / "pilots" / "tiny_real_csdm.yaml",
    "csdm_topk": ROOT / "configs" / "pilots" / "tiny_real_csdm_topk.yaml",
}

REQUIRED_KEYS = {
    "config",
    "teacher_type",
    "student_type",
    "teacher_model_name_or_path",
    "tokenizer_name_or_path",
    "local_files_only",
    "dataset_type",
    "data_path",
    "seq_len",
    "batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "student_hidden_size",
    "student_num_layers",
    "mixed_precision",
    "ce_weight",
    "kd_weight",
    "csdm_weight",
    "topk_enabled",
    "teacher_cache_enabled",
    "checkpoint_output_dir",
    "teacher_cache_dir",
    "save_at_end",
}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def test_all_pilot_configs_exist_and_have_required_keys() -> None:
    for path in PILOT_CONFIGS.values():
        assert path.is_file()
        config = _load_yaml(path)
        assert REQUIRED_KEYS.issubset(config)
        assert config["teacher_type"] == "hf"
        assert config["student_type"] == "mamba"
        assert config["dataset_type"] == "text"
        assert config["data_path"] == "data/smoke.txt"
        assert config["local_files_only"] is True
        assert config["teacher_cache_enabled"] is True
        assert config["save_at_end"] is True


def test_pilot_variant_weights_match_expected_ablation_roles() -> None:
    ce = _load_yaml(PILOT_CONFIGS["ce"])
    kd = _load_yaml(PILOT_CONFIGS["kd"])
    csdm = _load_yaml(PILOT_CONFIGS["csdm"])
    csdm_topk = _load_yaml(PILOT_CONFIGS["csdm_topk"])

    assert ce["ce_weight"] == 1.0
    assert ce["kd_weight"] == 0.0
    assert ce["csdm_weight"] == 0.0
    assert ce["topk_enabled"] is False

    assert kd["kd_weight"] > 0.0
    assert kd["csdm_weight"] == 0.0
    assert kd["topk_enabled"] is False

    assert csdm["kd_weight"] > 0.0
    assert csdm["csdm_weight"] > 0.0
    assert csdm["topk_enabled"] is False

    assert csdm_topk["kd_weight"] > 0.0
    assert csdm_topk["csdm_weight"] > 0.0
    assert csdm_topk["topk_enabled"] is True
    assert csdm_topk["top_k"] == 128


def test_tiny_real_pilot_ablation_matrix_contains_all_variants() -> None:
    matrix = _load_yaml(ROOT / "configs" / "ablations" / "tiny_real_pilot.yaml")
    variants = {variant["name"]: variant for variant in matrix["variants"]}

    assert {"ce_only", "ce_kd", "ce_kd_csdm", "ce_kd_csdm_topk"} == set(variants)
    assert variants["ce_only"]["kd_weight"] == 0.0
    assert variants["ce_kd"]["kd_weight"] > 0.0
    assert variants["ce_kd"]["csdm_weight"] == 0.0
    assert variants["ce_kd_csdm"]["csdm_weight"] > 0.0
    assert variants["ce_kd_csdm"]["topk_enabled"] is False
    assert variants["ce_kd_csdm_topk"]["topk_enabled"] is True
    assert all(variant["requires_mamba"] is True for variant in variants.values())


def test_tiny_real_pilot_ablation_dry_run_is_offline(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ablation_matrix.py",
            "--matrix",
            "configs/ablations/tiny_real_pilot.yaml",
            "--dry-run",
            "--only",
            "ce_kd_csdm_topk",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )

    assert result.stderr == ""
    assert "ce_kd_csdm_topk:" in result.stdout
    assert "train.py" in result.stdout
    assert "--teacher-type hf" in result.stdout
    assert "--student-type mamba" in result.stdout
    assert "--local-files-only" in result.stdout
    assert str(tmp_path / "ce_kd_csdm_topk" / "teacher_cache") in result.stdout
    assert str(tmp_path / "ce_kd_csdm_topk" / "checkpoints") in result.stdout
    assert not (tmp_path / "ce_kd_csdm_topk").exists()


def test_tiny_real_pilot_ablation_summary_is_report_compatible(tmp_path: Path) -> None:
    summary_path = tmp_path / "ablation_summary.json"
    report_dir = tmp_path / "report"
    records = [
        {
            "name": name,
            "status": "skipped",
            "metrics": {"total": index + 1.0, "optimizer_step": 0},
            "command": [sys.executable, "train.py"],
            "returncode": None,
            "reason": "dry-run compatibility fixture",
        }
        for index, name in enumerate(("ce_only", "ce_kd", "ce_kd_csdm", "ce_kd_csdm_topk"))
    ]
    summary_path.write_text(json.dumps(records), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            "--ablation-summary",
            str(summary_path),
            "--output-dir",
            str(report_dir),
            "--columns",
            "name,status,total,optimizer_step,reason",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )

    rows = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in rows] == ["ce_only", "ce_kd", "ce_kd_csdm", "ce_kd_csdm_topk"]
    assert (report_dir / "report.csv").is_file()
    assert (report_dir / "report.md").is_file()


def test_run_tiny_pilot_ce_dry_run_is_offline(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_tiny_pilot.py",
            "--variant",
            "ce",
            "--base-output-dir",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    assert "tiny_real_ce.yaml" in result.stdout
    assert "train.py" in result.stdout
    assert "--teacher-type hf" in result.stdout
    assert "--student-type mamba" in result.stdout
    assert "--local-files-only" in result.stdout
    assert "run_" in result.stdout
    run_dir = Path(result.stdout.strip().splitlines()[-1])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "9A"


def test_run_tiny_pilot_all_dry_run_and_override(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_tiny_pilot.py",
            "--variant",
            "all",
            "--base-output-dir",
            str(tmp_path),
            "--dry-run",
            "--override",
            "max_steps=1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )

    assert result.stdout.count("run_registered_experiment.py") == 4
    assert result.stdout.count("train.py") == 4
    assert "--max_steps 1" in result.stdout
    assert "tiny_real_ce.yaml" in result.stdout
    assert "tiny_real_kd.yaml" in result.stdout
    assert "tiny_real_csdm.yaml" in result.stdout
    assert "tiny_real_csdm_topk.yaml" in result.stdout
