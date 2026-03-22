<div align="center">

# ✈️ OpenPilot

### Computer Pilot Agent — Control Your Computer with Eyes, Hands & Voice

**by [Horcrux Technologies](https://github.com/skytinstone)**

---

*마우스도, 키보드도 필요 없다.*
*당신의 눈빛, 손짓, 목소리가 곧 인터페이스다.*

</div>

---

## What is OpenPilot?

**OpenPilot**은 사람의 자연스러운 신체 신호만으로 컴퓨터를 완전히 제어하는 **Computer Pilot Agent**입니다.

기존 마우스·키보드 중심의 인터페이스를 넘어, 눈(Eye) · 손(Hand) · 목소리(Voice) 세 가지 채널을 통해 운영체제와 애플리케이션을 자유롭게 조종할 수 있는 새로운 Human-Computer Interaction 패러다임을 목표로 합니다.

> "The next interface is you."

---

## Vision & Roadmap

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenPilot Roadmap                           │
├──────────┬──────────────────────────────────┬───────────────────┤
│  Phase   │  기능                             │  상태              │
├──────────┼──────────────────────────────────┼───────────────────┤
│  Phase 1 │     Eye Tracking Cursor Control  │  ✅ 개발 완료       │
│  Phase 2 │     Hand Gesture Control         │  ✅ 개발 완료       │
│  Phase 3 │     Voice Command Integration    │  ✅ 개발 완료       │
│  Phase 4 │     Unified Pilot Agent (AI)     │  🔲 개발 예정       │
└──────────┴──────────────────────────────────┴───────────────────┘
```

### Phase 1 — Eye Tracking `✅ Current`
웹캠 하나로 홍채를 추적해 마우스 커서를 제어합니다. 시선이 곧 포인터입니다.

### Phase 2 — Hand Gesture `✅ Current`
카메라로 손 제스처를 인식해 클릭, 드래그, 스크롤 등 세밀한 조작을 지원합니다.

### Phase 3 — Voice Command `✅ Current`
음성 명령으로 앱 실행, 텍스트 입력, 단축키 실행 등 고수준 제어를 수행합니다.

### Phase 4 — Unified Pilot Agent *(Coming Soon)*
Eye · Hand · Voice 세 채널을 AI가 통합 해석해 사용자 의도를 자동으로 판단하고 실행하는 완전 자율 에이전트를 구현합니다.

---

## Phase 1 — Eye Tracking Cursor Control

> 현재 릴리즈: **v0.1.0** | 지원 플랫폼: **macOS**

### 핵심 기술

| 기술 | 역할 |
|------|------|
| **MediaPipe FaceMesh** | 478개 얼굴 랜드마크 + 홍채 중심 좌표 추출 |
| **EMA Smoothing** | 시선 떨림 제거 (지수 이동 평균) |
| **5-Point Calibration** | 개인 시선 특성 보정으로 정확도 향상 |
| **Dwell Detection** | 일정 시간 응시로 클릭 대체 |
| **Quartz CGEvent** | macOS 네이티브 커서 제어 |

### 아키텍처

```
OpenPilot/
├── main.py                      # 진입점 (CLI, 배너, 권한 설정)
├── openpilot                    # 실행 스크립트
├── requirements.txt
├── config/
│   └── settings.yaml            # 카메라·시선·캘리브레이션 파라미터
├── core/
│   └── vision/
│       ├── camera_capture.py    # 카메라 프레임 캡처
│       ├── eye_tracker.py       # MediaPipe 홍채 추적
│       ├── gaze_estimator.py    # 시선 → 화면 좌표 변환
│       └── hover_detector.py    # 시선 호버 감지 (Dwell Time)
├── action/
│   └── mouse_controller.py      # macOS 커서 이동 (Quartz)
└── feedback/
    ├── screen_overlay.py        # 화면 테두리 피드백 (NSWindow)
    └── hover_overlay.py         # 호버 상태 피드백
```

### 데이터 흐름

```
웹캠 프레임
    │
    ▼
EyeTracker — MediaPipe FaceMesh로 홍채 중심 좌표 추출
    │
    ▼
GazeEstimator — EMA 스무딩 + 캘리브레이션 보정 → 화면 좌표 변환
    │
    ├──▶  MouseController — CGWarpMouseCursorPosition으로 커서 이동
    │
    └──▶  HoverDetector — Dwell Time 감지 → 호버/클릭 피드백
```

---

## 플랫폼 지원

| 플랫폼 | Phase 1 (Eye) | Phase 2 (Hand) | Phase 3 (Voice) | 비고 |
|--------|:-------------:|:--------------:|:---------------:|------|
| 🍎 **macOS** | ✅ 지원 | ✅ 지원 | ✅ 지원 | 현재 개발 중 |
| 🪟 **Windows** | 🔲 예정 | 🔲 예정 | 🔲 예정 | 추후 지원 예정 |
| 🐧 **Linux** | 🔲 예정 | 🔲 예정 | 🔲 예정 | 추후 지원 예정 |

---

## Quick Start

```bash
# 1. 저장소 클론
git clone https://github.com/skytinstone/OpenPilot.git
cd OpenPilot

# 2. 패키지 설치
pip install -r requirements.txt

# 3. macOS 권한 설정 (최초 1회 — 접근성 · 카메라 · 마이크)
./openpilot --setup

# 4. Phase 1 — 눈 트래킹 실행
./openpilot

# 5. Phase 2 — 손 제스처 실행
./openpilot --step 2

# 6. Phase 3 — 음성 제어 실행 (ANTHROPIC_API_KEY 설정 시 AI 모드 활성화)
export ANTHROPIC_API_KEY=your_key_here
./openpilot --step 3
```

처음 실행이라면 `c` 키를 눌러 **5포인트 캘리브레이션**을 먼저 진행하세요.
시선 정확도가 크게 향상됩니다.

---

## Tech Stack

### Phase 1 — Eye Tracking

| 기능 | 내부 모듈 | 라이브러리 | 역할 |
|------|-----------|-----------|------|
| 카메라 캡처 | `core/vision/camera_capture.py` | `opencv-python` | 웹캠 프레임 읽기 |
| 홍채 추적 | `core/vision/eye_tracker.py` | `mediapipe` | FaceMesh 478 랜드마크 + 홍채 중심 |
| 시선 추정 | `core/vision/gaze_estimator.py` | `numpy` | EMA 스무딩 · 5포인트 캘리브레이션 |
| 호버 감지 | `core/vision/hover_detector.py` | — | Dwell Time 기반 호버 판정 |
| 커서 이동 | `action/mouse_controller.py` | `pyobjc-Quartz` | `CGWarpMouseCursorPosition` |
| 화면 오버레이 | `feedback/screen_overlay.py` | `pyobjc-Cocoa` | `NSWindow` 테두리 피드백 |
| 호버 오버레이 | `feedback/hover_overlay.py` | `pyobjc-Cocoa` | 시선 호버 시각 피드백 |
| 설정 관리 | `config/__init__.py` | `pyyaml` | `settings.yaml` 로드 / 저장 |

### Phase 2 — Hand Gesture

| 기능 | 내부 모듈 | 라이브러리 | 역할 |
|------|-----------|-----------|------|
| 손 추적 | `core/vision/hand_tracker.py` | `mediapipe` | HandLandmarker 21 랜드마크 + 제스처 분류 |
| 클릭 | `action/click_controller.py` | `pyobjc-Quartz` | `CGEventCreateMouseEvent` 좌/우 클릭 |
| 스크롤 | `action/scroll_controller.py` | `pyobjc-Quartz` | `CGScrollWheelEvent` + 관성(inertia) |
| 줌 | `action/zoom_controller.py` | `pyobjc-Quartz` | 양손 핀치 거리 → `Ctrl+Scroll` 줌 |

### Phase 3 — Voice + AI

| 기능 | 내부 모듈 | 라이브러리 | 역할 |
|------|-----------|-----------|------|
| 마이크 녹음 | `core/audio/audio_capture.py` | `sounddevice` | 16kHz mono float32 실시간 캡처 |
| 음성 → 텍스트 | `core/audio/whisper_transcriber.py` | `faster-whisper` | Whisper STT (로컬 추론) |
| AI 명령 해석 | `core/ai/command_processor.py` | `anthropic` | Claude API 음성 → 액션 변환 |
| 텍스트 입력 | `action/keyboard_controller.py` | `pyobjc-Quartz` · `subprocess` | 클립보드 + `Cmd+V` 붙여넣기 |

### 공통 인프라

| 역할 | 라이브러리 | 용도 |
|------|-----------|------|
| 영상 처리 | `opencv-python` | 카메라 프레임 캡처 · 시각화 렌더링 |
| 수치 계산 | `numpy` | 좌표 변환 · 신호 처리 |
| macOS 시스템 | `pyobjc-framework-Quartz` | 커서 · 클릭 · 스크롤 · 줌 이벤트 |
| macOS UI | `pyobjc-framework-Cocoa` | NSWindow 오버레이 |
| 설정 파일 | `pyyaml` | YAML 읽기 / 쓰기 |

---

## 설치 (Phase 1)

### 요구사항

- macOS 12 Monterey 이상
- Python 3.10+
- 웹캠 (내장 카메라 가능)

### 패키지 설치

```bash
pip install -r requirements.txt
```

| 패키지 | 용도 |
|--------|------|
| `opencv-python` | 카메라 캡처 · 프레임 처리 |
| `mediapipe` | FaceMesh 478 랜드마크 (홍채 포함) |
| `numpy` | 좌표 계산 |
| `pyobjc-framework-Quartz` | 커서 이동 · 접근성 권한 확인 |
| `pyobjc-framework-Cocoa` | 화면 오버레이 (NSWindow) |
| `pyyaml` | 설정 파일 로드 |

---

## macOS 권한 설정 (최초 1회)

OpenPilot은 두 가지 macOS 권한이 필요합니다.

| 권한 | 용도 |
|------|------|
| **접근성** | 커서 이동 (CGWarpMouseCursorPosition) |
| **카메라** | 홍채 추적 (MediaPipe FaceMesh) |

```bash
./openpilot --setup
```

터미널 앱을 자동 감지하고 시스템 설정을 열어 단계별로 안내합니다.

---

## 실행

```bash
# 기본 실행
./openpilot

# 마우스 이동 없이 눈 추적만 확인
./openpilot --no-mouse

# 랜드마크 시각화 포함 (디버그 모드)
./openpilot --debug

# 환경 및 권한 상태 확인
./openpilot --check

# 권한 대화형 자동 설정
./openpilot --setup
```

---

## 조작키

| 키 | 동작 |
|----|------|
| `c` | 캘리브레이션 시작 (5포인트) |
| `m` | 마우스 제어 ON / OFF |
| `d` | 랜드마크 디버그 시각화 ON / OFF |
| `r` | 캘리브레이션 리셋 |
| `q` / `ESC` | 종료 |

---

## 캘리브레이션

첫 실행 시 `c`를 눌러 5포인트 캘리브레이션을 진행하면 시선 정확도가 크게 향상됩니다.

1. 화면에 포인트가 순서대로 나타남
2. 각 포인트를 **1.5초 동안 응시**
3. 4 모서리 + 중앙 총 5포인트 완료
4. 개인 시선 오프셋 자동 보정 적용

> 캘리브레이션 없이도 동작하나 정확도가 낮을 수 있습니다.

---

## 설정

[config/settings.yaml](config/settings.yaml)에서 세부 파라미터를 조정할 수 있습니다.

```yaml
eye_tracking:
  smoothing_alpha: 0.2        # EMA 스무딩 (낮을수록 부드럽고 느림)
  dead_zone_px: 4             # 이 픽셀 이하 이동 무시 (떨림 방지)
  gaze_scale_x: 1.6           # 수평 시선 민감도
  gaze_scale_y: 1.6           # 수직 시선 민감도
  calibration_dwell_ms: 1500  # 캘리브레이션 포인트 응시 유지 시간 (ms)
```

---

## About

**Horcrux Technologies**는 신체 신호 기반의 차세대 Human-Computer Interaction을 연구·개발하는 회사입니다.

OpenPilot은 신체적 제약 없이 누구나 컴퓨터를 자유롭게 사용할 수 있는 세상을 만들기 위한 첫 번째 프로젝트입니다.

---

<div align="center">

MIT License · © 2025 Horcrux Technologies

</div>
