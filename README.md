# V2E_zoo

`V2E_zoo`는 image-to-event와 video-to-event 모델들을 한 workspace에서 실행해 비교하기 위한 정리 repo입니다.

포함된 상위 runner는 두 묶음입니다.

- `video/run_event_models.py`: 비디오를 event stream 또는 event frame preview로 변환
- `image/run_image_models.py`, `image/run_i2e.py`, `image/run_eventgan.py`, `image/run_tetci_byeongjun.py`: 이미지 또는 이미지 pair를 event 표현으로 변환

## 포함 모델

### Video

| 이름 | 폴더 | 설명 |
| --- | --- | --- |
| `dvs` | `video/DVS-Voltmeter/` | frame sequence 기반 DVS simulator |
| `rpg` | `video/rpg_vid2e/` | ESIM/CUDA 기반 video-to-events |
| `v2e` | `video/v2e/` | v2e event camera emulator |
| `senpi` | `video/senpi_ebi/` | SENPI event processing wrapper |
| `v2ce` | `video/V2CE-Toolbox/` | pretrained V2CE inference |

### Image

| 이름 | 폴더 | 설명 |
| --- | --- | --- |
| `i2e` | `image/I2E/` | 단일 이미지를 `T x 2 x H x W` event frame으로 변환 |
| `eventgan` | `image/EventGAN/` | 연속 프레임 2장 사이의 event volume 예측 |
| `tetci` | `image/TETCI_byeongjun/` | LIF neuron processing 단계 실행 |

## 새 PC 설치

권장 환경은 Linux + Python 3.9입니다. CUDA를 쓰려면 먼저 새 PC의 CUDA/드라이버에 맞는 PyTorch 설치 조합을 확인하세요.

```bash
git clone <YOUR_REPOSITORY_URL> V2E_zoo
cd V2E_zoo

conda create -n V2E_zoo python=3.9
conda activate V2E_zoo

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`rpg_vid2e`를 실행하려면 로컬 CUDA extension을 추가로 빌드합니다.

```bash
python -m pip install -e ./video/rpg_vid2e/esim_torch
```

PyTorch를 CUDA 버전에 맞춰 직접 설치해야 하는 PC라면, `requirements.txt` 설치 전에 PyTorch 공식 명령으로 `torch`, `torchvision`, `torchaudio`를 먼저 설치한 뒤 진행하세요.

## 빠른 검증

Image smoke test:

```bash
python image/run_image_models.py image/I2E/assets \
  --results-dir image/results \
  --models all
```

Video smoke test:

```bash
python video/run_event_models.py video \
  --results-dir video/results \
  --models dvs,v2e,senpi \
  --skip-existing
```

전체 video 모델 실행:

```bash
python video/run_event_models.py video \
  --results-dir video/results \
  --models all
```

`rpg`와 `v2ce`는 각각 CUDA extension과 checkpoint가 필요합니다. 처음 환경을 옮긴 직후에는 `dvs,v2e,senpi`로 먼저 검증한 뒤 `rpg,v2ce`를 추가하는 편이 문제 원인을 분리하기 쉽습니다.

## Video Runner 사용법

입력은 비디오 파일들이 들어 있는 디렉토리입니다. 지원 확장자는 `mp4`, `mov`, `avi`, `mkv`입니다.

```bash
python video/run_event_models.py <VIDEO_DIR> \
  --results-dir video/results \
  --models all
```

특정 모델만 실행:

```bash
python video/run_event_models.py video \
  --results-dir video/results \
  --models dvs,rpg,v2e
```

기존 결과는 재사용하고 preview만 다시 만들기:

```bash
python video/run_event_models.py video \
  --results-dir video/results \
  --models all \
  --skip-existing
```

주요 옵션:

```text
--models          all 또는 dvs,rpg,v2e,senpi,v2ce
--results-dir     결과 저장 디렉토리
--preview-ms      preview event frame 누적 시간. 기본 33ms
--skip-existing   기존 결과가 있으면 모델 재실행 생략
--max-frame-num   지원 모델에서 처리할 최대 프레임 수
```

모델별 튜닝 옵션은 `python video/run_event_models.py --help`에서 확인할 수 있습니다.

## Image Runner 사용법

I2E:

```bash
python image/run_i2e.py image/I2E/assets \
  --results-dir image/results/i2e \
  --limit 1
```

EventGAN:

```bash
python image/run_eventgan.py \
  --prev-image image/I2E/assets/original_1.jpg \
  --next-image image/I2E/assets/original_2.jpg \
  --results-dir image/results/eventgan
```

TETCI_byeongjun threshold sweep:

```bash
python image/run_tetci_byeongjun.py image/I2E/assets \
  --limit 1 \
  --thresholds 0.45,0.55,0.65,0.75,0.85 \
  --results-dir image/results/TETCI_byeongjun_thresh_sweep
```

Image 쪽 상세 설명은 `image/README.md`에 있습니다.

## 결과 파일

생성 결과는 기본적으로 다음 위치에 저장합니다.

```text
video/results/
image/results/
image/results_*/
```

대표 출력 예:

```text
video/results/DVS-Voltmeter_d3.txt
video/results/DVS-Voltmeter_d3_event_frame_33ms.png
video/results/v2e_d3/v2e_d3.txt
video/results/v2e_d3_event_frame_33ms.png
image/results/i2e/<image>_i2e_events.npz
image/results/i2e/<image>_i2e_event_frame.png
```

## GitHub에 올릴 때 주의할 점

현재 `image/EventGAN`, `image/I2E`, `image/TETCI_byeongjun`, `video/DVS-Voltmeter`, `video/V2CE-Toolbox`, `video/rpg_vid2e`, `video/senpi_ebi`, `video/v2e`는 각각 원본 repo에서 가져온 코드입니다. 폴더 안에 `.git` 디렉토리가 남아 있으면 GitHub에 올릴 때 중첩 repo로 처리되어 내용이 제대로 올라가지 않을 수 있습니다.

단순히 전체 코드를 한 repo로 옮길 계획이면 하위 `.git` 디렉토리를 제거한 뒤 commit하세요.

```bash
find image video -mindepth 2 -name .git -type d -prune
```

위 명령으로 하위 `.git` 위치를 확인한 다음, 의도한 경우에만 삭제합니다.

```bash
rm -rf image/EventGAN/.git image/I2E/.git image/TETCI_byeongjun/.git
rm -rf video/DVS-Voltmeter/.git video/V2CE-Toolbox/.git video/rpg_vid2e/.git video/senpi_ebi/.git video/v2e/.git
```

대용량 파일은 `.gitignore`에서 제외하고 있습니다. checkpoint, zip, mp4, h5 같은 파일을 반드시 공유해야 한다면 Git LFS 또는 별도 다운로드 링크를 쓰는 것이 좋습니다.

```bash
git status --short
git add README.md requirements.txt image/requirements.txt video/V2E_zoo_requirements.txt
git add image video .gitignore
git commit -m "Prepare V2E zoo for transfer"
git push
```

## 체크포인트와 데이터

- `video/V2CE-Toolbox/weights/v2ce_3d.pt`가 없으면 `v2ce` runner는 해당 모델을 건너뜁니다.
- `image/EventGAN/logs/EventGAN/checkpoints`에 checkpoint가 없으면 `run_eventgan.py`는 실행을 중단합니다.
- `rpg_vid2e`는 CUDA extension 빌드가 필요합니다.
- 로컬 테스트용 비디오와 생성 결과는 repo에 올리지 않는 것을 권장합니다.
