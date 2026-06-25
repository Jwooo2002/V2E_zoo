# V2E_zoo 전체 사용법

이 프로젝트는 video-to-event 모델들과 image-to-event 모델들을 한 환경에서 테스트하기 위해 정리한 workspace입니다.

크게 두 영역으로 나뉩니다.

- `video/`: 비디오 입력을 event stream 또는 event frame으로 변환하는 모델들
- `image/`: 정적 이미지 또는 이미지 pair를 event 표현으로 변환하는 모델들

## 디렉토리 구조

```text
V2E_zoo/
├── requirements.txt
├── video/
│   ├── run_event_models.py
│   ├── DVS-Voltmeter/
│   ├── rpg_vid2e/
│   ├── v2e/
│   ├── senpi_ebi/
│   └── V2CE-Toolbox/
└── image/
    ├── README.md
    ├── run_i2e.py
    ├── run_eventgan.py
    ├── run_tetci_byeongjun.py
    ├── I2E/
    ├── EventGAN/
    └── TETCI_byeongjun/
```

## 환경

공통 conda 환경 이름은 `V2E_zoo_video`입니다.

검증한 Python 버전:

```text
Python 3.9.25
```

권장 생성 방식:

```bash
conda create -n V2E_zoo_video python=3.9
conda activate V2E_zoo_video
```

```bash
conda activate V2E_zoo_video
```

또는 `conda run`으로 실행합니다.

```bash
conda run -n V2E_zoo_video python video/run_event_models.py --help
```

requirements 파일:

```text
requirements.txt
video/V2E_zoo_requirements.txt
image/requirements.txt
```

`requirements.txt`는 video 모델 세팅 때 정리한 공통 requirements입니다. `image/requirements.txt`에는 image 쪽에서 추가로 필요한 패키지까지 포함되어 있습니다.

주의:

- I2E 원본 README는 Python 3.10을 권장하지만, 이 zoo runner의 smoke test는 Python 3.9.25에서 통과했습니다.
- EventGAN은 오래된 코드라 너무 최신 Python보다 Python 3.9 환경에서 맞추는 편이 안전합니다.
- 따라서 이 repo 전체 실행 기준 권장 Python은 `3.9`입니다.

## Weight / Checkpoint

이 GitHub repo에는 Git LFS를 쓰지 않기 위해 weight, checkpoint, generated results, 입력 영상 파일을 포함하지 않습니다.

필요한 weight는 원본 repo에서 직접 받아서 아래 위치에 두면 됩니다.

```text
EventGAN checkpoint:
  원본: https://github.com/alexzzhu/EventGAN
  위치: image/EventGAN/logs/EventGAN/checkpoints/

V2CE checkpoint:
  원본: https://github.com/ucsd-hdsi-dvs/V2CE-Toolbox
  위치: video/V2CE-Toolbox/weights/v2ce_3d.pt
```

weight 없이도 runner의 `--help`와 일부 smoke/import 확인은 가능하지만, EventGAN/V2CE의 실제 weight inference는 checkpoint가 필요합니다.

## Video Models

video 쪽은 [video/run_event_models.py](video/run_event_models.py) 하나로 실행합니다.

포함 모델:

```text
dvs    -> DVS-Voltmeter
rpg    -> rpg_vid2e
v2e    -> v2e
senpi  -> senpi_ebi
v2ce   -> V2CE-Toolbox
```

### 기본 실행

입력은 비디오 파일들이 들어있는 디렉토리입니다.

```bash
conda run -n V2E_zoo_video python video/run_event_models.py video \
  --results-dir video/results \
  --models all
```

특정 모델만 실행:

```bash
conda run -n V2E_zoo_video python video/run_event_models.py video \
  --results-dir video/results \
  --models dvs,rpg,v2e
```

이미 결과가 있는 모델은 건너뛰고 preview만 다시 만들기:

```bash
conda run -n V2E_zoo_video python video/run_event_models.py video \
  --results-dir video/results \
  --models all \
  --skip-existing
```

### 출력

각 모델별 원본 결과와 함께 33ms 누적 event frame preview를 저장합니다.

예시:

```text
video/results/DVS-Voltmeter_d3.txt
video/results/DVS-Voltmeter_d3_event_frame_33ms.png

video/results/rpg_vid2e_d3/
video/results/rpg_vid2e_d3_event_frame_33ms.png

video/results/v2e_d3/
video/results/v2e_d3_event_frame_33ms.png

video/results/senpi_ebi_d3.npz
video/results/senpi_ebi_d3_event_frame_33ms.png

video/results/V2CE-Toolbox_d3/
video/results/V2CE-Toolbox_d3_event_frame_33ms.png
```

현재 `d3.mp4` 기준으로 생성 확인된 preview:

```text
video/results/DVS-Voltmeter_d3_event_frame_33ms.png
video/results/rpg_vid2e_d3_event_frame_33ms.png
video/results/v2e_d3_event_frame_33ms.png
video/results/senpi_ebi_d3_event_frame_33ms.png
video/results/V2CE-Toolbox_d3_event_frame_33ms.png
```

### 주요 파라미터

공통:

```text
--models          all 또는 dvs,rpg,v2e,senpi,v2ce
--results-dir     결과 저장 디렉토리
--preview-ms      preview event frame 누적 시간. 기본 33ms
--skip-existing   기존 결과가 있으면 재실행 생략
--max-frame-num   지원 모델에서 처리할 최대 프레임 수
```

rpg_vid2e:

```text
--rpg-contrast-neg
--rpg-contrast-pos
--rpg-refractory-ns
```

v2e:

```text
--v2e-pos-thres
--v2e-neg-thres
--v2e-sigma-thres
--v2e-cutoff-hz
--v2e-leak-rate-hz
--v2e-shot-noise-rate-hz
--v2e-refractory-period
--v2e-disable-slomo / --no-v2e-disable-slomo
--v2e-extra-arg
```

V2CE:

```text
--v2ce-infer-type center|pano
--v2ce-height
--v2ce-width
--v2ce-ceil
--v2ce-batch-size
--v2ce-stage2-batch-size
--v2ce-extra-arg
```

SENPI:

```text
--senpi-device auto|cuda|cpu
```

### Video 모델별 주의점

- `DVS-Voltmeter`: frame directory 형식이 필요해서 runner가 내부적으로 비디오를 프레임으로 변환합니다.
- `rpg_vid2e`: CUDA extension 기반입니다. shell에서 nested `conda run`으로 돌리면 CUDA 인식 문제가 있었고, 현재 Python runner에서는 정상 실행됐습니다.
- `v2e`: threshold/noise/filter 파라미터가 많습니다. 이벤트 수가 과하거나 적으면 threshold와 noise 계열부터 조절합니다.
- `SENPI`: wrapper인 `run_senpi_video.py`를 통해 frame directory 입력을 받도록 정리했습니다.
- `V2CE`: checkpoint `V2CE-Toolbox/weights/v2ce_3d.pt`가 필요합니다. square 입력에서는 `--v2ce-infer-type pano`가 안전했습니다.

## Image Models

image 쪽 상세 사용법은 [image/README.md](image/README.md)에 따로 정리되어 있습니다. 여기서는 전체 요약만 적습니다.

포함 모델:

```text
I2E              -> 정적 이미지 1장 to event frame
EventGAN         -> 연속 이미지 2장 to event volume
TETCI_byeongjun  -> LIF repo의 neuron processing 단계
```

### I2E

정적 이미지 1장을 바로 event frame으로 변환합니다.

```bash
conda run -n V2E_zoo_video python image/run_i2e.py image/I2E/assets --limit 1
```

출력:

```text
{stem}_i2e_events.npz
{stem}_i2e_event_frame.png
i2e_summary.json
```

`event_frames` shape:

```text
T x 2 x H x W
```

### EventGAN

EventGAN은 이미지 1장이 아니라 frame pair를 입력으로 받습니다.

```bash
conda run -n V2E_zoo_video python image/run_eventgan.py \
  --prev-image /path/to/frame_000.png \
  --next-image /path/to/frame_001.png \
  --results-dir image/results/eventgan
```

기본값은 공식 demo에 맞춰 resize를 `861 x 260`으로 둡니다.

출력:

```text
{prev}_{next}_eventgan_event_volume.npz
{prev}_{next}_eventgan_event_frame.png
{prev}_{next}_eventgan_event_time_image.png
eventgan_summary.json
```

`event_volume` shape:

```text
1 x 18 x H x W
```

`18`은 `9 positive time bins + 9 negative time bins`입니다.

주의:

- 서로 다른 unrelated 이미지 2장을 넣으면 논문 그림과 다르게 보입니다.
- 같은 장면의 실제 연속 프레임을 넣어야 의미가 있습니다.
- `event_time_image.png`가 공식 demo의 time image 표현에 더 가깝습니다.

### TETCI_byeongjun

원래 clone repo 이름은 `LIF`였지만 여기서는 `TETCI_byeongjun`으로 통일했습니다.

```bash
conda run -n V2E_zoo_video python image/run_tetci_byeongjun.py image/I2E/assets --limit 1
```

threshold sweep:

```bash
conda run -n V2E_zoo_video python image/run_tetci_byeongjun.py \
  image/I2E/assets \
  --limit 1 \
  --thresholds 0.45,0.55,0.65,0.75,0.85 \
  --results-dir image/results/TETCI_byeongjun_thresh_sweep
```

확인된 threshold sweep:

```text
0.45 -> spike_count 30028
0.55 -> spike_count 26171
0.65 -> spike_count 21943
0.75 -> spike_count 16630
0.85 -> spike_count 12867
```

주의:

TETCI_byeongjun README에는 전체 pipeline이 설명되어 있지만, 현재 repo만으로 end-to-end 실행한 것은 아닙니다. 지금 runner는 LIF neuron processing만 단독 실행합니다.

전체 pipeline에는 별도로 다음이 필요합니다.

```text
CelebA 이미지
Adaptive StyleFlow event-style 변환
face-parsing.PyTorch + segmentation checkpoint
face cropping
background removal
region polarity/refinement
```

## 빠른 확인 명령

각 runner의 옵션 확인:

```bash
conda run -n V2E_zoo_video python video/run_event_models.py --help
conda run -n V2E_zoo_video python image/run_i2e.py --help
conda run -n V2E_zoo_video python image/run_eventgan.py --help
conda run -n V2E_zoo_video python image/run_tetci_byeongjun.py --help
```

## 현재 smoke test 요약

Video `d3.mp4`:

```text
DVS-Voltmeter: 결과 생성 및 33ms preview 확인
rpg_vid2e: 결과 생성 및 33ms preview 확인
v2e: 결과 생성 및 33ms preview 확인
SENPI: 결과 생성 및 33ms preview 확인
V2CE: checkpoint 사용, pano mode로 결과 생성 및 33ms preview 확인
```

Image:

```text
I2E: event_frames_shape [8, 2, 128, 128]
EventGAN: checkpoint 사용, event_shape [1, 18, 260, 861]
TETCI_byeongjun: threshold sweep 확인
```

## GitHub 업로드 주의사항

이 workspace는 private GitHub repo로 올릴 수 있지만, 그대로 전체를 `git add -A` 하면 안 됩니다.

주의할 점:

- 현재 생성된 결과 파일 중 100MB가 넘는 파일이 있습니다.
- GitHub 일반 git push는 100MB 초과 파일을 거부합니다.
- cloned repo 안의 `.git/` 디렉토리와 generated results는 publish 대상에서 제외하는 것이 안전합니다.
- weight/checkpoint를 같이 공유해야 하면 Git LFS 또는 별도 storage를 써야 합니다.

현재 100MB 초과 파일 예시:

```text
image/EventGAN/logs-20260625T053306Z-3-001.zip
video/V2CE-Toolbox/weights/v2ce_3d.pt
video/results/DVS-Voltmeter_d3.txt
video/results/v2e_d3/v2e_d3.txt
```

권장 방식:

```text
1. 코드, runner, README, requirements만 private repo에 push
2. results/ 디렉토리는 push하지 않음
3. *.pt, *.zip, *.h5 같은 weight/checkpoint는 Git LFS나 외부 storage 사용
4. 업로드 전에 clean publish directory를 만들어 기존 루트 git 상태와 섞이지 않게 처리
```

현재 루트 git에는 이전 프로젝트 파일 삭제 상태가 많이 섞여 있으므로, 이 디렉토리에서 바로 `git add -A` 하는 것은 피해야 합니다.
