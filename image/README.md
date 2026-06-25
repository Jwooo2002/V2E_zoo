# V2E_zoo Image Models 사용법

이 디렉토리는 이미지 기반 event 변환 repo들을 한 환경에서 테스트하기 위해 정리한 공간입니다.

현재 포함된 모델은 세 가지입니다.

- `I2E`: 정적 이미지 1장을 입력으로 받아 `T x 2 x H x W` 이벤트 프레임을 생성합니다.
- `EventGAN`: 연속된 프레임 2장을 입력으로 받아 `1 x 18 x H x W` 이벤트 volume을 생성합니다.
- `TETCI_byeongjun`: `LIF` repo를 이름 통일한 버전입니다. 원 논문 pipeline 전체가 아니라, 현재 repo에 들어있는 LIF neuron processing 단계만 실행합니다.

## 디렉토리 구조

```text
image/
├── I2E/
├── EventGAN/
├── TETCI_byeongjun/
├── requirements.txt
├── run_i2e.py
├── run_eventgan.py
└── run_tetci_byeongjun.py
```

## 환경

기존 video 쪽에서 만든 conda 환경을 그대로 사용합니다.

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

또는 매번 `conda run`으로 실행할 수 있습니다.

```bash
conda run -n V2E_zoo_video python image/run_i2e.py --help
```

필요 패키지는 [requirements.txt](requirements.txt)에 정리되어 있습니다.

```bash
conda run -n V2E_zoo_video python -m pip install -r image/requirements.txt
```

주의:

- I2E 원본 README는 Python 3.10을 권장하지만, 여기의 `run_i2e.py` smoke test는 Python 3.9.25에서 통과했습니다.
- EventGAN은 2019년 코드라 최신 Python보다는 Python 3.9 환경이 안전합니다.
- 이 image runner 묶음의 권장 Python은 `3.9`입니다.

## Weight / Checkpoint

이 repo에는 Git LFS를 쓰지 않기 위해 weight/checkpoint 파일을 포함하지 않습니다.

EventGAN checkpoint는 원본 repo 안내를 따라 받아서 아래 위치에 두면 됩니다.

```text
image/EventGAN/logs/EventGAN/checkpoints/
```

원본:

```text
https://github.com/alexzzhu/EventGAN
```

## 1. I2E

I2E는 이미지 1장을 바로 event frame으로 변환합니다. SNN 학습이나 dataset loader까지 신경 쓰지 않고, 변환 결과만 저장하도록 정리했습니다.

기본 실행:

```bash
conda run -n V2E_zoo_video python image/run_i2e.py image/I2E/assets --limit 1
```

주요 옵션:

```text
--results-dir   결과 저장 디렉토리
--image-size    입력 이미지를 정사각형으로 resize할 크기
--ratio         이벤트 threshold ratio. 낮을수록 이벤트가 많아짐
--limit         처리할 이미지 개수. 0이면 전체
--device        auto, cuda, cpu
```

출력:

```text
{stem}_i2e_events.npz
{stem}_i2e_event_frame.png
i2e_summary.json
```

`npz` 안에는 `event_frames`가 들어 있으며 shape은 다음과 같습니다.

```text
T x 2 x H x W
```

여기서 `2`는 polarity channel입니다.

## 2. EventGAN

EventGAN은 이미지 1장 변환 모델이 아닙니다. 이전 프레임과 다음 프레임, 즉 frame pair를 넣어 그 사이 event volume을 예측합니다.

기본 실행:

```bash
conda run -n V2E_zoo_video python image/run_eventgan.py image/I2E/assets
```

위 명령은 디렉토리 안의 앞 2개 이미지를 pair로 사용합니다. 제대로 쓰려면 같은 장면의 연속 프레임을 직접 지정하는 것이 좋습니다.

```bash
conda run -n V2E_zoo_video python image/run_eventgan.py \
  --prev-image /path/to/frame_000.png \
  --next-image /path/to/frame_001.png \
  --results-dir image/results/eventgan
```

주요 옵션:

```text
--prev-image       이전 프레임
--next-image       다음 프레임
--width            입력 resize 너비. 기본 861
--height           입력 resize 높이. 기본 260
--time-bins        time bin 수. checkpoint 기본 9
--checkpoint-dir   checkpoint 디렉토리
--synthetic-shift  next frame이 없을 때 같은 이미지를 x축으로 shift해서 synthetic pair 생성
```

출력:

```text
{prev}_{next}_eventgan_event_volume.npz
{prev}_{next}_eventgan_event_frame.png
{prev}_{next}_eventgan_event_time_image.png
eventgan_summary.json
```

`event_volume.npz` 안에는 `event_volume`이 들어 있으며 shape은 보통 다음과 같습니다.

```text
1 x 18 x H x W
```

`18`은 `9 positive time bins + 9 negative time bins`입니다.

주의:

- 논문 그림과 비슷하게 보려면 같은 장면의 실제 연속 프레임을 넣어야 합니다.
- 서로 다른 이미지 2장을 넣으면 결과가 어색한 것이 정상입니다.
- `event_time_image.png`가 공식 demo의 time image 표현에 더 가깝습니다.

## 3. TETCI_byeongjun

원래 clone한 repo 이름은 `LIF`였지만, 여기서는 `TETCI_byeongjun`으로 통일했습니다.

실행:

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

주요 옵션:

```text
--beta              LIF membrane decay beta
--threshold         단일 threshold
--thresholds        여러 threshold를 한 번에 sweep
--no-spike-on-prob  non-spike 위치에 랜덤 white noise를 넣는 확률
--spike-value       spike 위치에 저장할 grayscale 값
--limit             처리할 이미지 개수
```

출력:

```text
{stem}_TETCI_byeongjun_th{threshold}.png
TETCI_byeongjun_summary.json
```

주의:

이 repo의 README에는 전체 pipeline이 설명되어 있지만, 현재 여기서 실행 가능한 것은 LIF neuron processing 부분입니다.

전체 pipeline에는 다음이 필요합니다.

- CelebA 이미지
- Adaptive StyleFlow 기반 Event-Style Dataset Creation
- face-parsing.PyTorch와 segmentation checkpoint
- face cropping
- background removal
- region polarity/refinement

현재 repo에는 위 전체 pipeline을 end-to-end로 실행할 수 있는 코드와 weight가 모두 들어있지는 않습니다. 따라서 `run_tetci_byeongjun.py` 결과는 최종 synthetic-event dataset이 아니라, LIF 처리 단독 결과입니다.

또한 원본 구현은 픽셀을 Python 이중 loop로 처리하므로 느립니다.

## Smoke Test 기록

이미 확인한 실행 결과입니다.

I2E:

```text
event_frames_shape: [8, 2, 128, 128]
event_count: 17732
```

EventGAN:

```text
event_shape: [1, 18, 260, 861]
checkpoint 사용: true
```

TETCI_byeongjun threshold sweep:

```text
0.45 -> spike_count 30028
0.55 -> spike_count 26171
0.65 -> spike_count 21943
0.75 -> spike_count 16630
0.85 -> spike_count 12867
```

## 결과 파일 위치

테스트 결과는 다음 디렉토리에 생성되어 있습니다.

```text
image/results_i2e_test/
image/results_eventgan_test/
image/results_TETCI_byeongjun_thresh_sweep/
```

## 권장 사용 구분

- 정적 이미지 1장을 event frame으로 빠르게 바꾸고 싶으면 `run_i2e.py`
- 연속 프레임 2장 사이의 event volume을 만들고 싶으면 `run_eventgan.py`
- TETCI_byeongjun/LIF repo의 일부 처리만 확인하고 싶으면 `run_tetci_byeongjun.py`
