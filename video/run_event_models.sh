#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_event_models.sh DATA_DIR [RESULTS_DIR]

Environment overrides:
  CONDA_ENV=V2E_zoo_video
  MODELS=all                         # all,dvs,rpg,v2e,senpi,v2ce or comma list
  TMP_ROOT=/tmp/v2e_zoo_batch
  MAX_FRAME_NUM=0                    # 0 means all frames where supported

  RPG_CONTRAST_NEG=0.2
  RPG_CONTRAST_POS=0.2
  RPG_REFRACTORY_NS=0

  V2E_POS_THRES=0.2
  V2E_NEG_THRES=0.2
  V2E_SIGMA_THRES=0.03
  V2E_CUTOFF_HZ=300
  V2E_LEAK_RATE_HZ=0.01
  V2E_SHOT_NOISE_RATE_HZ=0.001
  V2E_REFRACTORY_PERIOD=0.0005
  V2E_DISABLE_SLOMO=1
  V2E_EXTRA_ARGS=""

  V2CE_INFER_TYPE=pano
  V2CE_HEIGHT=260
  V2CE_WIDTH=346
  V2CE_CEIL=10
  V2CE_BATCH_SIZE=1
  V2CE_STAGE2_BATCH_SIZE=24
  V2CE_EXTRA_ARGS=""

  SENPI_DEVICE=auto                  # auto,cuda,cpu
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(cd "$1" && pwd)"
RESULTS_DIR="${2:-"$ROOT_DIR/results"}"
mkdir -p "$RESULTS_DIR"

CONDA_ENV="${CONDA_ENV:-V2E_zoo_video}"
MODELS="${MODELS:-all}"
TMP_ROOT="${TMP_ROOT:-/tmp/v2e_zoo_batch}"
MAX_FRAME_NUM="${MAX_FRAME_NUM:-0}"

RPG_CONTRAST_NEG="${RPG_CONTRAST_NEG:-0.2}"
RPG_CONTRAST_POS="${RPG_CONTRAST_POS:-0.2}"
RPG_REFRACTORY_NS="${RPG_REFRACTORY_NS:-0}"

V2E_POS_THRES="${V2E_POS_THRES:-0.2}"
V2E_NEG_THRES="${V2E_NEG_THRES:-0.2}"
V2E_SIGMA_THRES="${V2E_SIGMA_THRES:-0.03}"
V2E_CUTOFF_HZ="${V2E_CUTOFF_HZ:-300}"
V2E_LEAK_RATE_HZ="${V2E_LEAK_RATE_HZ:-0.01}"
V2E_SHOT_NOISE_RATE_HZ="${V2E_SHOT_NOISE_RATE_HZ:-0.001}"
V2E_REFRACTORY_PERIOD="${V2E_REFRACTORY_PERIOD:-0.0005}"
V2E_DISABLE_SLOMO="${V2E_DISABLE_SLOMO:-1}"
V2E_EXTRA_ARGS="${V2E_EXTRA_ARGS:-}"

V2CE_INFER_TYPE="${V2CE_INFER_TYPE:-pano}"
V2CE_HEIGHT="${V2CE_HEIGHT:-260}"
V2CE_WIDTH="${V2CE_WIDTH:-346}"
V2CE_CEIL="${V2CE_CEIL:-10}"
V2CE_BATCH_SIZE="${V2CE_BATCH_SIZE:-1}"
V2CE_STAGE2_BATCH_SIZE="${V2CE_STAGE2_BATCH_SIZE:-24}"
V2CE_EXTRA_ARGS="${V2CE_EXTRA_ARGS:-}"

SENPI_DEVICE="${SENPI_DEVICE:-auto}"

run_py() {
  conda run -n "$CONDA_ENV" python "$@"
}

has_model() {
  local model="$1"
  [[ "$MODELS" == "all" || ",$MODELS," == *",$model,"* ]]
}

video_size_args() {
  local video="$1"
  run_py "$ROOT_DIR/video_metadata.py" "$video" --v2e-args
}

process_video() {
  local video="$1"
  local name
  name="$(basename "$video")"
  name="${name%.*}"

  local work_root="$TMP_ROOT/$name"
  local rpg_input="$work_root/rpg_input"
  local dvs_input="$work_root/dvs_input"
  rm -rf "$work_root"
  mkdir -p "$rpg_input" "$dvs_input"

  echo "==> preparing $name"
  run_py "$ROOT_DIR/prepare_video_inputs.py" "$video" --name "$name" --rpg-root "$rpg_input" --dvs-root "$dvs_input"

  if has_model dvs; then
    echo "==> DVS-Voltmeter: $name"
    (
      cd "$ROOT_DIR/DVS-Voltmeter"
      conda run -n "$CONDA_ENV" python main.py --input_dir "$dvs_input" --output_dir "$RESULTS_DIR"
    )
    if [[ -f "$RESULTS_DIR/$name.txt" ]]; then
      mv "$RESULTS_DIR/$name.txt" "$RESULTS_DIR/DVS-Voltmeter_${name}.txt"
    fi
  fi

  if has_model rpg; then
    echo "==> rpg_vid2e: $name"
    (
      cd "$ROOT_DIR/rpg_vid2e"
      conda run -n "$CONDA_ENV" python esim_torch/scripts/generate_events.py \
        -i "$rpg_input" \
        -o "$RESULTS_DIR/rpg_vid2e_${name}" \
        -cn "$RPG_CONTRAST_NEG" \
        -cp "$RPG_CONTRAST_POS" \
        -rp "$RPG_REFRACTORY_NS"
    )
  fi

  if has_model v2e; then
    echo "==> v2e: $name"
    local slomo_args=()
    if [[ "$V2E_DISABLE_SLOMO" == "1" ]]; then
      slomo_args+=(--disable_slomo)
    fi
    # shellcheck disable=SC2207
    local size_args=($(video_size_args "$video"))
    (
      cd "$ROOT_DIR/v2e"
      conda run -n "$CONDA_ENV" python v2e.py \
        -i "$video" \
        -o "$RESULTS_DIR/v2e_${name}" \
        --overwrite \
        --no_preview \
        "${slomo_args[@]}" \
        "${size_args[@]}" \
        --pos_thres "$V2E_POS_THRES" \
        --neg_thres "$V2E_NEG_THRES" \
        --sigma_thres "$V2E_SIGMA_THRES" \
        --cutoff_hz "$V2E_CUTOFF_HZ" \
        --leak_rate_hz "$V2E_LEAK_RATE_HZ" \
        --shot_noise_rate_hz "$V2E_SHOT_NOISE_RATE_HZ" \
        --refractory_period "$V2E_REFRACTORY_PERIOD" \
        --dvs_text "v2e_${name}.txt" \
        --dvs_h5 "v2e_${name}.h5" \
        --dvs_vid "v2e_${name}.mp4" \
        --vid_orig None \
        --vid_slomo None \
        $V2E_EXTRA_ARGS
    )
  fi

  if has_model senpi; then
    echo "==> SENPI: $name"
    local senpi_args=()
    if [[ "$SENPI_DEVICE" != "auto" ]]; then
      senpi_args+=(--device "$SENPI_DEVICE")
    fi
    run_py "$ROOT_DIR/run_senpi_video.py" \
      --frames-dir "$dvs_input/$name" \
      --output "$RESULTS_DIR/senpi_ebi_${name}.npz" \
      "${senpi_args[@]}"
  fi

  if has_model v2ce; then
    echo "==> V2CE-Toolbox: $name"
    if [[ ! -f "$ROOT_DIR/V2CE-Toolbox/weights/v2ce_3d.pt" ]]; then
      echo "missing V2CE-Toolbox/weights/v2ce_3d.pt; skipping V2CE for $name" >&2
    else
      local max_args=()
      if [[ "$MAX_FRAME_NUM" != "0" ]]; then
        max_args+=(--max_frame_num "$MAX_FRAME_NUM")
      fi
      (
        cd "$ROOT_DIR/V2CE-Toolbox"
        conda run -n "$CONDA_ENV" python v2ce.py \
          -i "$video" \
          -o "$RESULTS_DIR/V2CE-Toolbox_${name}" \
          --out_name_suffix "$name" \
          -t "$V2CE_INFER_TYPE" \
          --height "$V2CE_HEIGHT" \
          --width "$V2CE_WIDTH" \
          --ceil "$V2CE_CEIL" \
          -b "$V2CE_BATCH_SIZE" \
          --stage2_batch_size "$V2CE_STAGE2_BATCH_SIZE" \
          --write_event_frame_video true \
          -l info \
          "${max_args[@]}" \
          $V2CE_EXTRA_ARGS
      )
    fi
  fi
}

shopt -s nullglob
videos=("$DATA_DIR"/*.mp4 "$DATA_DIR"/*.mov "$DATA_DIR"/*.avi "$DATA_DIR"/*.mkv)
if [[ ${#videos[@]} -eq 0 ]]; then
  echo "No video files found in $DATA_DIR" >&2
  exit 1
fi

for video in "${videos[@]}"; do
  process_video "$video"
done

echo "done: $RESULTS_DIR"
