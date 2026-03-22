"""
Open Pilot — 메인 진입점
사용법:
  ./openpilot               # 기본 실행
  ./openpilot --no-mouse    # 마우스 이동 없이 실행
  ./openpilot --debug       # 랜드마크 시각화 포함
  ./openpilot --check       # 환경/권한 체크만 실행
"""
import argparse
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── ANSI 색상 코드 ──────────────────────────────────────────────
R  = "\033[31m"   # 빨강
G  = "\033[32m"   # 초록
Y  = "\033[33m"   # 노랑
B  = "\033[34m"   # 파랑
M  = "\033[35m"   # 마젠타
C  = "\033[36m"   # 시안
W  = "\033[97m"   # 흰색 (밝음)
DG = "\033[90m"   # 진한 회색
BO = "\033[1m"    # 볼드
DIM= "\033[2m"    # 어둡게
NC = "\033[0m"    # 리셋

# ── ASCII 아트 배너 (OPEN / PILOT 2행 분리 — 최대 40자, 80컬럼 안전) ───
ASCII_LOGO = f"""
{DG}  ┌──────────────────────────────────────────┐{NC}
{DG}  │{NC}{R}{BO}  ██████╗ ██████╗ ███████╗███╗  ██╗       {NC}{DG}│{NC}
{DG}  │{NC}{R}{BO} ██╔═══██╗██╔══██╗██╔════╝████╗ ██║       {NC}{DG}│{NC}
{DG}  │{NC}{R}{BO} ██║   ██║██████╔╝█████╗  ██╔██╗██║       {NC}{DG}│{NC}
{DG}  │{NC}{R}{BO} ██║   ██║██╔═══╝ ██╔══╝  ██║╚████║       {NC}{DG}│{NC}
{DG}  │{NC}{R}{BO} ╚██████╔╝██║     ███████╗██║ ╚███║       {NC}{DG}│{NC}
{DG}  │{NC}{R}{BO}  ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚══╝       {NC}{DG}│{NC}
{DG}  ├──────────────────────────────────────────┤{NC}
{DG}  │{NC}{W}{BO} ██████╗ ██╗██╗      ██████╗ ████████╗    {NC}{DG}│{NC}
{DG}  │{NC}{W}{BO} ██╔══██╗██║██║     ██╔═══██╗╚══██╔══╝    {NC}{DG}│{NC}
{DG}  │{NC}{W}{BO} ██████╔╝██║██║     ██║   ██║   ██║        {NC}{DG}│{NC}
{DG}  │{NC}{W}{BO} ██╔═══╝ ██║██║     ██║   ██║   ██║        {NC}{DG}│{NC}
{DG}  │{NC}{W}{BO} ██║     ██║███████╗╚██████╔╝   ██║        {NC}{DG}│{NC}
{DG}  │{NC}{W}{BO} ╚═╝     ╚═╝╚══════╝ ╚═════╝    ╚═╝        {NC}{DG}│{NC}
{DG}  └──────────────────────────────────────────┘{NC}"""

SUBTITLE = f"""
{DG}         AI-Powered Hands-Free Computer Control{NC}
{DG}                v0.1.0  │  Phase 1  │  macOS{NC}
"""


def clear_line():
    sys.stdout.write("\033[2K\r")
    sys.stdout.flush()


def print_step(icon, msg, status=None):
    """로딩 단계 출력"""
    if status == "ok":
        print(f"  {G}✔{NC}  {W}{msg}{NC}  {G}완료{NC}")
    elif status == "warn":
        print(f"  {Y}⚠{NC}  {W}{msg}{NC}  {Y}경고{NC}")
    elif status == "fail":
        print(f"  {R}✘{NC}  {W}{msg}{NC}  {R}실패{NC}")
    else:
        print(f"  {C}{icon}{NC}  {msg}", end="", flush=True)


def animate_loading(msg: str, stop_event: threading.Event):
    """스피너 애니메이션"""
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r  {C}{frames[i % len(frames)]}{NC}  {msg}  ")
        sys.stdout.flush()
        time.sleep(0.08)
        i += 1
    sys.stdout.write("\r")
    sys.stdout.flush()


def loading_step(msg: str, fn, warn_ok=False):
    """
    로딩 단계 실행 + 스피너 표시
    fn() 실행 → 성공/실패 표시
    """
    stop = threading.Event()
    t = threading.Thread(target=animate_loading, args=(msg, stop), daemon=True)
    t.start()

    try:
        result = fn()
        stop.set()
        t.join()
        clear_line()
        if result is False and warn_ok:
            print_step("", msg, "warn")
        elif result is False:
            print_step("", msg, "fail")
        else:
            print_step("", msg, "ok")
        return result
    except Exception as e:
        stop.set()
        t.join()
        clear_line()
        print_step("", f"{msg}  ({e})", "fail")
        return False


def print_banner():
    os.system("clear")
    print(ASCII_LOGO)
    print(SUBTITLE)


def print_ready():
    print(f"""
{DG}  ─────────────────────────────────────────────────────────────────────{NC}
{G}{BO}  🛩️   Open Pilot 준비 완료! 카메라 창이 열립니다.{NC}
{DG}  ─────────────────────────────────────────────────────────────────────{NC}
{W}  조작키{NC}
{DG}  ┌──────────────────────────────────────────────────────────────────┐{NC}
{DG}  │{NC}  {Y}c{NC} = 캘리브레이션   {Y}m{NC} = 마우스 ON/OFF   {Y}d{NC} = 디버그   {Y}r{NC} = 리셋   {Y}q{NC} = 종료  {DG}│{NC}
{DG}  └──────────────────────────────────────────────────────────────────┘{NC}
""")


def print_goodbye():
    print(f"""
{DG}  ─────────────────────────────────────────────────────────────────────{NC}
{M}  Open Pilot 종료됨. 다음에 또 만나요! 👋{NC}
{DG}  ─────────────────────────────────────────────────────────────────────{NC}
""")


# ── 권한 설정 헬퍼 ──────────────────────────────────────────────

# macOS 시스템 설정 딥링크
_PREF_URLS = {
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "camera":        "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera",
    "microphone":    "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
}


def _get_terminal_app() -> str:
    """현재 실행 중인 터미널 앱 이름 반환"""
    try:
        import subprocess
        result = subprocess.check_output(
            ["osascript", "-e",
             'tell application "System Events" to get name of first process '
             'whose frontmost is true'],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        return result if result else "터미널 앱"
    except Exception:
        return "터미널 앱"


def _ask_yn(prompt: str) -> bool:
    """y/n 입력 받기. y=True, n=False"""
    while True:
        try:
            ans = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if ans in ("y", "yes", "ㅛ"):
            return True
        if ans in ("n", "no", "ㅜ"):
            return False
        print(f"  {Y}  y 또는 n 으로 입력해주세요.{NC}")


def _open_pref(key: str):
    """시스템 설정 해당 권한 패널 열기"""
    os.system(f'open "{_PREF_URLS[key]}"')


def _check_accessibility() -> bool:
    try:
        from Quartz import CGPreflightPostEventAccess
        return bool(CGPreflightPostEventAccess())
    except Exception:
        return False


def _check_camera() -> bool:
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ok = cap.isOpened()
        cap.release()
        return ok
    except Exception:
        return False


def _check_microphone() -> bool:
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        ok = pa.get_default_input_device_info() is not None
        pa.terminate()
        return ok
    except Exception:
        # pyaudio 미설치 시 패스 (Phase 1은 마이크 불필요)
        return True


def _setup_one_permission(name: str, label: str, check_fn, pref_key: str,
                           terminal_app: str) -> bool:
    """
    권한 하나를 대화형으로 설정
    - 이미 허용: 그냥 통과
    - 미허용: y/n 물어보고 y면 시스템 설정 열기 → 확인 대기
    """
    if check_fn():
        print_step("", label, "ok")
        return True

    print(f"\n  {R}✘{NC}  {W}{label}{NC}  {DG}— 권한이 없습니다{NC}")
    print(f"  {DG}  터미널 앱: {W}{terminal_app}{NC}")

    if not _ask_yn(f"  {Y}  시스템 설정을 열어 권한을 부여하시겠습니까? (y/n) ▶ {NC}"):
        print(f"  {DG}  건너뜀{NC}")
        return False

    _open_pref(pref_key)
    print(f"\n  {C}  시스템 설정이 열렸습니다.{NC}")
    print(f"  {W}  [{terminal_app}] 을 목록에 추가하고 토글을 켠 후{NC}")
    input(f"  {Y}  완료되면 Enter 키를 누르세요... {NC}")

    if check_fn():
        print_step("", label, "ok")
        return True
    else:
        print(f"  {R}✘  아직 권한이 확인되지 않습니다. 설정 후 다시 시도해주세요.{NC}")
        return False


# ── 체크 / 셋업 커맨드 ───────────────────────────────────────────

def run_check():
    """패키지 + 권한 상태만 출력 (변경 없음)"""
    print_banner()
    print(f"{W}{BO}  환경 및 권한 체크{NC}\n")
    all_ok = True

    def chk(label, pkg, import_name):
        nonlocal all_ok
        try:
            __import__(import_name)
            print_step("", label, "ok")
            return True
        except ImportError:
            print(f"  {R}✘{NC}  {label}  {DG}→  pip install {pkg}{NC}")
            all_ok = False
            return False

    print(f"  {DG}[ 필수 패키지 ]{NC}")
    chk("opencv-python", "opencv-python", "cv2")
    chk("mediapipe",     "mediapipe",     "mediapipe")
    chk("numpy",         "numpy",         "numpy")
    chk("pyyaml",        "pyyaml",        "yaml")

    print(f"\n  {DG}[ macOS 시스템 패키지 ]{NC}")
    chk("pyobjc-Cocoa",  "pyobjc-framework-Cocoa",               "AppKit")
    chk("pyobjc-Quartz", "pyobjc-framework-Quartz",              "Quartz")
    chk("pyobjc-AppSvc", "pyobjc-framework-ApplicationServices", "ApplicationServices")

    print(f"\n  {DG}[ macOS 권한 ]{NC}")
    if _check_accessibility():
        print_step("", "접근성 권한", "ok")
    else:
        print(f"  {R}✘{NC}  접근성 권한  {DG}→  ./openpilot --setup 으로 자동 설정{NC}")
        all_ok = False

    if _check_camera():
        print_step("", "카메라 권한", "ok")
    else:
        print(f"  {R}✘{NC}  카메라 권한  {DG}→  ./openpilot --setup 으로 자동 설정{NC}")
        all_ok = False

    print()
    if all_ok:
        print(f"  {G}{BO}✔  모든 체크 통과! 실행 준비 완료{NC}\n")
    else:
        print(f"  {Y}⚠   권한 미설정 항목이 있습니다.{NC}")
        print(f"  {DG}  → {W}./openpilot --setup{NC}{DG} 으로 자동 설정하세요.\n{NC}")


def run_setup():
    """대화형 권한 자동 설정 (y/n)"""
    print_banner()
    print(f"{W}{BO}  macOS 권한 자동 설정{NC}")
    print(f"  {DG}권한이 없는 항목만 물어봅니다.{NC}\n")

    terminal_app = _get_terminal_app()
    print(f"  {DG}감지된 터미널: {W}{terminal_app}{NC}\n")
    print(f"  {DG}{'─'*44}{NC}\n")

    results = {}

    # 1. 접근성
    results["accessibility"] = _setup_one_permission(
        "accessibility", "접근성 권한  (커서/키보드 제어 필수)",
        _check_accessibility, "accessibility", terminal_app,
    )

    # 2. 카메라
    results["camera"] = _setup_one_permission(
        "camera", "카메라 권한  (눈 트래킹 필수)",
        _check_camera, "camera", terminal_app,
    )

    # 3. 마이크 (Phase 1 선택사항)
    print(f"\n  {DG}마이크 권한은 Phase 1에서 선택사항입니다.{NC}")
    if _ask_yn(f"  {Y}  마이크 권한도 설정하시겠습니까? (y/n) ▶ {NC}"):
        results["microphone"] = _setup_one_permission(
            "microphone", "마이크 권한  (음성 명령 — Phase 3)",
            _check_microphone, "microphone", terminal_app,
        )
    else:
        print(f"  {DG}  건너뜀{NC}")
        results["microphone"] = None

    # 4. 화면 해상도
    print(f"\n  {DG}{'─'*44}{NC}")
    _setup_screen()

    # ── 최종 결과 요약
    print(f"\n  {DG}{'─'*44}{NC}")
    print(f"  {W}{BO}설정 결과 요약{NC}\n")

    labels = {
        "accessibility": "접근성",
        "camera":        "카메라",
        "microphone":    "마이크",
    }
    all_required_ok = True
    for key, ok in results.items():
        if ok is None:
            print(f"  {DG}─{NC}  {labels[key]}  {DG}건너뜀{NC}")
        elif ok:
            print_step("", labels[key], "ok")
        else:
            print_step("", labels[key], "fail")
            if key != "microphone":
                all_required_ok = False

    print()
    if all_required_ok:
        print(f"  {G}{BO}✔  필수 권한 설정 완료! 이제 실행하세요:{NC}")
        print(f"  {Y}     ./openpilot\n{NC}")
    else:
        print(f"  {R}  일부 권한이 설정되지 않았습니다.{NC}")
        print(f"  {DG}  다시 시도: {W}./openpilot --setup\n{NC}")


# ── 화면 해상도 설정 ─────────────────────────────────────────────

# MacBook/Mac 주요 프리셋 (논리 해상도 기준 — NSScreen 반환값과 동일 단위)
_SCREEN_PRESETS = [
    # ── MacBook ───────────────────────────────────────────────────
    ("MacBook 14\"  기본 (1512×982)",      1512,  982),
    ("MacBook 14\"  더 넓게 (1800×1169)",  1800, 1169),
    ("MacBook 16\"  기본 (1728×1117)",     1728, 1117),
    ("MacBook 16\"  더 넓게 (2056×1329)",  2056, 1329),
    ("MacBook Air 13\"  기본 (1280×832)",  1280,  832),
    ("MacBook Air 15\"  기본 (1440×932)",  1440,  932),
    ("MacBook 13\"  기본 (1280×800)",      1280,  800),
    # ── iMac ──────────────────────────────────────────────────────
    ("iMac 24\"  기본 (2240×1260)",        2240, 1260),
    # ── 외장 모니터 27\" ──────────────────────────────────────────
    ("27\"  iMac / Studio Display (2560×1440)", 2560, 1440),
    ("27\"  4K UHD (3840×2160)",           3840, 2160),
    ("27\"  QHD (2560×1440)",              2560, 1440),
    ("27\"  FHD (1920×1080)",              1920, 1080),
    # ── 외장 모니터 32\" ──────────────────────────────────────────
    ("32\"  Pro Display XDR (3008×1692)",  3008, 1692),
    ("32\"  4K UHD (3840×2160)",           3840, 2160),
    ("32\"  QHD (2560×1440)",              2560, 1440),
    ("32\"  FHD (1920×1080)",              1920, 1080),
    # ── 기타 ─────────────────────────────────────────────────────
    ("직접 입력",                           0,     0),
]


def _detect_screen() -> tuple:
    """NSScreen으로 현재 화면 논리 해상도 반환"""
    try:
        from AppKit import NSScreen
        frame = NSScreen.mainScreen().frame()
        return int(frame.size.width), int(frame.size.height)
    except Exception:
        return 0, 0


def _setup_screen():
    """대화형 화면 해상도 선택 → settings.yaml 저장"""
    from config import load_config, save_config

    print(f"\n  {W}{BO}화면 해상도 설정{NC}")
    print(f"  {DG}시선과 커서의 좌표 매핑에 사용됩니다.{NC}\n")

    # 자동 감지
    detected_w, detected_h = _detect_screen()
    if detected_w:
        print(f"  {C}자동 감지된 해상도{NC}  →  {W}{BO}{detected_w} × {detected_h}{NC}")
        if _ask_yn(f"  {Y}  이 해상도로 설정하시겠습니까? (y/n) ▶ {NC}"):
            _write_screen_config(detected_w, detected_h)
            return
    else:
        print(f"  {Y}⚠  자동 감지 실패 — 직접 선택해주세요.{NC}")

    # 프리셋 목록
    print(f"\n  {DG}[ 화면 프리셋 선택 ]{NC}\n")
    for i, (label, w, h) in enumerate(_SCREEN_PRESETS):
        suffix = f"  {DG}({w}×{h}){NC}" if w else ""
        print(f"  {Y}{i + 1}{NC}. {label}{suffix}")

    while True:
        try:
            ans = input(f"\n  {Y}번호를 입력하세요 ▶ {NC}").strip()
            idx = int(ans) - 1
            if not (0 <= idx < len(_SCREEN_PRESETS)):
                raise ValueError
            break
        except (ValueError, EOFError):
            print(f"  {R}  올바른 번호를 입력해주세요.{NC}")

    label, w, h = _SCREEN_PRESETS[idx]

    # 직접 입력
    if w == 0:
        while True:
            try:
                raw = input(f"  {Y}해상도 입력 (예: 1512 982) ▶ {NC}").strip().split()
                w, h = int(raw[0]), int(raw[1])
                if w > 0 and h > 0:
                    break
            except (ValueError, IndexError, EOFError):
                pass
            print(f"  {R}  올바른 형식으로 입력해주세요 (가로 세로).{NC}")

    _write_screen_config(w, h)


def _write_screen_config(w: int, h: int):
    from config import load_config, save_config
    config = load_config()
    config.setdefault("screen", {})
    config["screen"]["width"]  = w
    config["screen"]["height"] = h
    save_config(config)
    print(f"\n  {G}✔{NC}  화면 해상도 저장 완료  →  {W}{BO}{w} × {h}{NC}")
    print(f"  {DG}  settings.yaml 에 반영되었습니다.{NC}\n")


# ── Step 1 실행 ─────────────────────────────────────────────────

def run_step1(no_mouse: bool, debug: bool):
    import cv2
    import types

    from config import load_config
    from core.vision.camera_capture import CameraCapture
    from core.vision.eye_tracker import EyeTracker
    from core.vision.gaze_estimator import GazeEstimator
    from core.vision.hover_detector import HoverDetector
    from feedback.screen_overlay import ScreenBorderOverlayV2
    from feedback.hover_overlay import HoverOverlay, hover_state
    from step1_targeting_test import (
        get_screen_size, draw_gaze_pointer, draw_calibration_screen,
        draw_hover_badge, draw_hud, draw_target_grid,
    )

    print_banner()
    print(f"{W}{BO}  Phase 1  —  Eye Tracking Targeting Test{NC}\n")
    print(f"  {DG}[ 모듈 초기화 중 ]{NC}\n")

    config     = loading_step("설정 파일 로드",       lambda: load_config() or True)
    screen_w, screen_h = get_screen_size(config)

    camera = [None]
    def init_camera():
        cam_cfg = config.get("camera", {})
        c = CameraCapture(
            device_index=cam_cfg.get("device_index", 0),
            width=cam_cfg.get("width", 1280),
            height=cam_cfg.get("height", 720),
            fps=cam_cfg.get("fps", 30),
        )
        if not c.start():
            return False
        camera[0] = c
        return True

    cam_ok = loading_step("카메라 초기화", init_camera)
    if not cam_ok:
        print(f"\n  {R}카메라를 시작할 수 없습니다. 권한을 확인하세요.{NC}\n")
        sys.exit(1)

    eye_tracker    = [None]
    gaze_estimator = [None]
    hover_detector = [None]

    def init_mediapipe():
        eye_tracker[0]    = EyeTracker(config)
        gaze_estimator[0] = GazeEstimator(screen_w, screen_h, config)
        hover_detector[0] = HoverDetector(dwell_threshold=0.8)
        return True

    loading_step("MediaPipe FaceMesh 로드", init_mediapipe)

    border = [None]
    def init_border():
        border[0] = ScreenBorderOverlayV2(border_width=5)
        border[0].start()
        return True

    loading_step("화면 테두리 오버레이 시작", init_border, warn_ok=True)

    hover_ov = [None]
    def init_hover():
        hover_ov[0] = HoverOverlay(dwell_threshold=0.8)
        hover_ov[0].start()
        return True

    loading_step("호버 오버레이 시작", init_hover, warn_ok=True)

    mouse = [None]
    mouse_enabled = [not no_mouse]
    if not no_mouse:
        def init_mouse():
            try:
                from action.mouse_controller import MouseController
                mouse[0] = MouseController()
                return True
            except Exception:
                mouse_enabled[0] = False
                return False
        loading_step("마우스 컨트롤러 초기화", init_mouse, warn_ok=True)

    print_ready()

    # ── 메인 루프 ────────────────────────────────────────────────
    show_debug = debug
    fps_counter, fps_start = 0, time.time()
    current_fps = 0.0
    eye_detected = False
    gaze_point   = None

    cam          = camera[0]
    et           = eye_tracker[0]
    ge           = gaze_estimator[0]
    hd           = hover_detector[0]

    while True:
        frame = cam.read()
        if frame is None:
            continue

        cam_h, cam_w = frame.shape[:2]
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 0.5:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start   = time.time()

        eye_data     = et.process(frame)
        eye_detected = eye_data is not None

        if eye_data is not None:
            if ge.is_calibrating:
                cal = ge.update_calibration(eye_data)
                if cal["done"]:
                    print(f"\r  {G}✔{NC}  캘리브레이션 완료!          ")
            else:
                gaze_point = ge.estimate(eye_data)
                if gaze_point:
                    if mouse_enabled[0] and mouse[0]:
                        mouse[0].move(gaze_point.x, gaze_point.y)
                    hovered = hd.update(gaze_point.x, gaze_point.y)
                    hover_state.set(hovered)

        if show_debug and eye_data is not None:
            gx = gaze_point.x if gaze_point else -1
            gy = gaze_point.y if gaze_point else -1
            frame = et.draw_debug(frame, eye_data,
                                  gaze_x=gx, gaze_y=gy,
                                  screen_w=screen_w, screen_h=screen_h)

        if ge.is_calibrating:
            cal_pt = ge.current_calibration_point
            if cal_pt:
                cal = ge.update_calibration(eye_data) if eye_data else \
                    {"done": False, "current_idx": 0, "total": 5, "progress": 0.0}
                frame = draw_calibration_screen(
                    frame, cal_pt, cal["progress"], cal["current_idx"],
                    cal["total"], cam_w, cam_h, screen_w, screen_h,
                )
        else:
            frame = draw_target_grid(frame, cam_w, cam_h)
            if gaze_point is not None:
                frame = draw_gaze_pointer(
                    frame, gaze_point.x, gaze_point.y,
                    screen_w, screen_h, cam_w, cam_h,
                )

        frame = draw_hover_badge(frame, hd.current_element, cam_w, cam_h)
        frame = draw_hud(frame, current_fps, ge.is_calibrated, mouse_enabled[0], eye_detected)

        cv2.imshow("Open Pilot — Eye Targeting Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            ge.start_calibration()
        elif key == ord('r'):
            ge.reset()
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"\r  {C}●{NC}  디버그 시각화: {'ON' if show_debug else 'OFF'}          ")
        elif key == ord('m'):
            mouse_enabled[0] = not mouse_enabled[0]
            if mouse_enabled[0] and mouse[0] is None:
                try:
                    from action.mouse_controller import MouseController
                    mouse[0] = MouseController()
                except Exception:
                    mouse_enabled[0] = False
            print(f"\r  {C}●{NC}  마우스 제어: {'ON' if mouse_enabled[0] else 'OFF'}          ")

    # ── 종료 ────────────────────────────────────────────────────
    hover_state.set(None)
    if hover_ov[0]:  hover_ov[0].stop()
    if border[0]:    border[0].stop()
    cam.stop()
    et.close()
    cv2.destroyAllWindows()
    print_goodbye()


# ── Step 2 실행 ─────────────────────────────────────────────────

def run_step2(debug: bool):
    from step2_hand_control import run as hand_run
    print_banner()
    print(f"{W}{BO}  Phase 2  —  Hand Gesture Control{NC}\n")
    hand_run(debug=debug)
    print_goodbye()


def run_step3(model: str):
    from step3_voice_control import run as voice_run
    print_banner()
    print(f"{W}{BO}  Phase 3  —  Voice Control  (Whisper + Claude AI){NC}\n")
    voice_run(model_size=model)
    print_goodbye()


# ── 진입점 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="openpilot",
        description="Open Pilot — AI 기반 핸즈프리 컴퓨터 제어",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    parser.add_argument("--step",     type=int, default=1, choices=[1, 2, 3],
                        help="실행할 Phase (1: 눈 트래킹  2: 손 제스처  3: 음성 제어)")
    parser.add_argument("--model",    type=str, default="base",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper 모델 크기 (Step 3 전용)")
    parser.add_argument("--no-mouse", action="store_true",
                        help="마우스 커서 이동 비활성화 (Step 1 전용)")
    parser.add_argument("--debug",    action="store_true",
                        help="랜드마크 시각화 활성화")
    parser.add_argument("--check",    action="store_true",
                        help="환경 및 권한 상태 확인")
    parser.add_argument("--setup",    action="store_true",
                        help="macOS 권한 대화형 자동 설정 (y/n)")
    parser.add_argument("-h", "--help", action="store_true",
                        help="도움말")
    args = parser.parse_args()

    if args.help:
        print_banner()
        print(f"""  {W}{BO}사용법{NC}
  {DG}┌──────────────────────────────────────────────────────────────────┐{NC}
  {DG}│{NC}  {Y}./openpilot{NC}                    Phase 1 (눈 트래킹)               {DG}│{NC}
  {DG}│{NC}  {Y}./openpilot --step 2{NC}           Phase 2 (손 제스처 클릭/스크롤/줌) {DG}│{NC}
  {DG}│{NC}  {Y}./openpilot --step 3{NC}           Phase 3 (음성 Whisper + Claude AI) {DG}│{NC}
  {DG}│{NC}  {Y}./openpilot --step 3 --model small{NC}  더 정확한 Whisper 모델 사용   {DG}│{NC}
  {DG}│{NC}  {Y}./openpilot --setup{NC}            권한 자동 설정 {R}← 처음 실행시{NC}        {DG}│{NC}
  {DG}│{NC}  {Y}./openpilot --check{NC}            환경 및 권한 상태 확인             {DG}│{NC}
  {DG}│{NC}  {Y}./openpilot --no-mouse{NC}         마우스 이동 없이 눈 추적만 확인    {DG}│{NC}
  {DG}│{NC}  {Y}./openpilot --debug{NC}            랜드마크 시각화 포함               {DG}│{NC}
  {DG}└──────────────────────────────────────────────────────────────────┘{NC}
""")
        return

    if args.setup:
        run_setup()
        return

    if args.check:
        run_check()
        return

    if args.step == 2:
        run_step2(debug=args.debug)
    elif args.step == 3:
        run_step3(model=args.model)
    else:
        run_step1(no_mouse=args.no_mouse, debug=args.debug)


if __name__ == "__main__":
    main()
