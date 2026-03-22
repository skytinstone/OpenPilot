# Open Pilot

**AI-Powered Hands-Free Computer Control**

눈 추적(Eye Tracking)으로 마우스 커서를 제어하는 핸즈프리 컴퓨터 제어 플랫폼.
현재 **Phase 1** — macOS 기반 시선 추적 커서 제어를 구현 중입니다.

---

## 개요

| 항목 | 내용 |
|------|------|
| 플랫폼 | macOS |
| 언어 | Python 3.10+ |
| 핵심 기술 | MediaPipe FaceMesh, OpenCV, pyobjc |
| 현재 단계 | Phase 1 — 눈 추적 커서 제어 |
| 버전 | v0.1.0 |

---

## 로드맵

```
Phase 1  ✅  눈 추적 커서 제어 (macOS)
Phase 2  🔲  클릭 / 제스처 인식
Phase 3  🔲  음성 명령 통합
```

---

## 아키텍처

```
OpenPilot/
├── main.py                      # 진입점 (CLI, 배너, 권한 설정)
├── openpilot                    # 실행 스크립트
├── requirements.txt
├── config/
│   └── settings.yaml            # 카메라·시선 추적·캘리브레이션 설정
├── core/
│   └── vision/
│       ├── camera_capture.py    # 카메라 프레임 캡처
│       ├── eye_tracker.py       # MediaPipe FaceMesh 홍채 추적
│       ├── gaze_estimator.py    # 홍채 오프셋 → 화면 좌표 변환 (EMA + 캘리브레이션)
│       └── hover_detector.py    # 시선 호버 감지 (Dwell Time)
├── action/
│   └── mouse_controller.py      # macOS 커서 이동 (Quartz CGEvent)
├── feedback/
│   ├── screen_overlay.py        # 화면 테두리 오버레이 (NSWindow)
│   └── hover_overlay.py         # 호버 피드백 오버레이
└── step1_targeting_test.py      # Phase 1 카메라 뷰 HUD·타겟 그리드
```

### 데이터 흐름

```
카메라 프레임
    │
    ▼
EyeTracker (MediaPipe FaceMesh)
    │  홍채 중심 좌표 (EyeData)
    ▼
GazeEstimator (EMA 스무딩 + 5포인트 캘리브레이션)
    │  화면 좌표 (ScreenPoint)
    ▼
MouseController ──→ 커서 이동 (CGWarpMouseCursorPosition)
    │
HoverDetector ──→ Dwell Time 감지 → HoverOverlay 피드백
```

---

## 설치

### 요구사항

- macOS 12 Monterey 이상
- Python 3.10+
- 웹캠 (내장 카메라 가능)

### 패키지 설치

```bash
pip install -r requirements.txt
```

requirements.txt에 포함된 패키지:

| 패키지 | 용도 |
|--------|------|
| `opencv-python>=4.8.0` | 카메라 캡처 · 프레임 처리 |
| `mediapipe>=0.10.0` | FaceMesh 478 랜드마크 (홍채 포함) |
| `numpy>=1.24.0` | 좌표 계산 |
| `pyobjc-framework-Quartz>=9.2` | 커서 이동 · 접근성 권한 확인 |
| `pyobjc-framework-Cocoa>=9.2` | 화면 오버레이 (NSWindow) |
| `pyobjc-framework-ApplicationServices>=9.2` | macOS 이벤트 |
| `pyyaml>=6.0` | 설정 파일 로드 |

---

## 권한 설정 (최초 1회)

Open Pilot은 두 가지 macOS 권한이 필요합니다.

| 권한 | 용도 |
|------|------|
| **접근성** | 커서 이동 (CGWarpMouseCursorPosition) |
| **카메라** | 눈 추적 (FaceMesh) |

자동 설정 (권장):

```bash
./openpilot --setup
```

터미널 앱을 자동 감지하고 시스템 설정을 직접 열어 안내합니다.

---

## 실행

```bash
# 기본 실행
./openpilot

# 마우스 이동 없이 눈 추적만 확인
./openpilot --no-mouse

# 랜드마크 시각화 포함 (디버그)
./openpilot --debug

# 환경 및 권한 상태 확인
./openpilot --check

# 권한 대화형 자동 설정
./openpilot --setup
```

---

## 조작키

실행 중 카메라 창이 열리면 아래 키로 제어합니다.

| 키 | 동작 |
|----|------|
| `c` | 캘리브레이션 시작 (5포인트) |
| `m` | 마우스 제어 ON / OFF 토글 |
| `d` | 랜드마크 시각화 ON / OFF 토글 |
| `r` | 캘리브레이션 리셋 |
| `q` / `ESC` | 종료 |

---

## 캘리브레이션

처음 실행 시 `c` 키를 눌러 5포인트 캘리브레이션을 진행하면 시선 정확도가 크게 향상됩니다.

1. 화면에 순서대로 나타나는 포인트를 **1.5초 동안** 응시
2. 4 모서리 + 중앙 총 5포인트 완료
3. 이후 캘리브레이션 보정 매핑 자동 적용

> 캘리브레이션 없이도 동작하나, 정확도가 낮을 수 있습니다.

---

## 설정

[config/settings.yaml](config/settings.yaml)에서 세부 파라미터를 조정할 수 있습니다.

```yaml
eye_tracking:
  smoothing_alpha: 0.2      # EMA 스무딩 (낮을수록 부드럽고 느림)
  dead_zone_px: 4           # 이 픽셀 이하 이동 무시 (떨림 방지)
  gaze_scale_x: 1.6         # 수평 시선 민감도
  gaze_scale_y: 1.6         # 수직 시선 민감도
  calibration_dwell_ms: 1500  # 캘리브레이션 포인트 응시 유지 시간
```

---

## 개발 환경

```bash
# 가상환경 생성 (권장)
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# 환경 체크
./openpilot --check
```

---

## 라이선스

MIT License
