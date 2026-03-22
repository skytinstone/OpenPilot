"""
호버 감지 모듈
macOS Accessibility API(AXUIElement)로 시선 위치의 UI 요소 감지

감지 정보:
  - 요소 이름 (앱 아이콘명, 버튼 레이블 등)
  - 요소 역할 (AXButton, AXIcon, AXMenuItem 등)
  - 화면 내 위치/크기 (bounding box)
  - 드웰 타임 (얼마나 오래 보고 있는지)
"""
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class HoverElement:
    title: str                          # 요소 이름
    role: str                           # AXButton, AXIcon, AXImage 등
    frame: Tuple[float, float, float, float]  # (x, y, w, h) 화면 좌표
    dwell_start: float = field(default_factory=time.time)

    @property
    def dwell_time(self) -> float:
        """현재까지 응시한 시간 (초)"""
        return time.time() - self.dwell_start

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.frame
        return (x + w / 2, y + h / 2)

    def is_same_element(self, other: "HoverElement") -> bool:
        if other is None:
            return False
        # 같은 위치 + 같은 이름이면 동일 요소로 판단
        x1, y1, w1, h1 = self.frame
        x2, y2, w2, h2 = other.frame
        return (abs(x1 - x2) < 10 and abs(y1 - y2) < 10 and
                abs(w1 - w2) < 10 and abs(h1 - h2) < 10)


# 호버 대상으로 인식할 AX Role 목록
HOVERABLE_ROLES = {
    "AXButton",
    "AXMenuItem",
    "AXMenuBarItem",
    "AXImage",
    "AXIcon",
    "AXLink",
    "AXTab",
    "AXRadioButton",
    "AXCheckBox",
    "AXPopUpButton",
    "AXCell",
    "AXStaticText",
}


class HoverDetector:
    """
    시선 좌표 → macOS UI 요소 감지
    AXUIElementCopyElementAtPosition 사용
    """

    def __init__(self, dwell_threshold: float = 0.8):
        """
        dwell_threshold: 호버 활성화까지 필요한 응시 시간 (초)
        """
        self._dwell_threshold = dwell_threshold
        self._current: Optional[HoverElement] = None
        self._lock = threading.Lock()
        self._ax_available = self._check_ax()

    def _check_ax(self) -> bool:
        try:
            from ApplicationServices import AXUIElementCreateSystemWide
            return True
        except ImportError:
            print("[HoverDetector] ApplicationServices 미설치 — AX 호버 감지 비활성화")
            print("               설치: pip install pyobjc-framework-ApplicationServices")
            return False

    def get_element_at(self, screen_x: float, screen_y: float) -> Optional[HoverElement]:
        """
        화면 좌표 (screen_x, screen_y)의 UI 요소 반환
        macOS Accessibility API 사용
        """
        if not self._ax_available:
            return None

        try:
            from ApplicationServices import (
                AXUIElementCreateSystemWide,
                AXUIElementCopyElementAtPosition,
                AXUIElementCopyAttributeValue,
                kAXErrorSuccess,
            )

            system = AXUIElementCreateSystemWide()
            err, element = AXUIElementCopyElementAtPosition(
                system, float(screen_x), float(screen_y), None
            )

            if err != kAXErrorSuccess or element is None:
                return None

            # 요소 속성 읽기
            def ax_attr(el, attr):
                err, val = AXUIElementCopyAttributeValue(el, attr, None)
                return val if err == kAXErrorSuccess else None

            role  = ax_attr(element, "AXRole") or ""
            title = ax_attr(element, "AXTitle") or ax_attr(element, "AXDescription") or ax_attr(element, "AXLabel") or ""
            pos   = ax_attr(element, "AXPosition")
            size  = ax_attr(element, "AXSize")

            # hoverable 역할 필터
            if role not in HOVERABLE_ROLES:
                return None

            # 위치/크기 파싱
            if pos and size:
                try:
                    from Foundation import NSValue
                    px = pos.pointValue().x if hasattr(pos, 'pointValue') else float(str(pos).split('x:')[1].split(',')[0].strip('{').strip())
                    py = pos.pointValue().y if hasattr(pos, 'pointValue') else float(str(pos).split('y:')[1].strip('}').strip())
                    pw = size.sizeValue().width if hasattr(size, 'sizeValue') else 60.0
                    ph = size.sizeValue().height if hasattr(size, 'sizeValue') else 60.0
                    frame = (px, py, pw, ph)
                except Exception:
                    frame = (screen_x - 30, screen_y - 30, 60, 60)
            else:
                frame = (screen_x - 30, screen_y - 30, 60, 60)

            return HoverElement(title=str(title), role=str(role), frame=frame)

        except Exception as e:
            return None

    def update(self, screen_x: float, screen_y: float) -> Optional[HoverElement]:
        """
        시선 좌표 업데이트 → 드웰 타임 누적 후 호버 요소 반환
        같은 요소를 계속 보면 dwell_time 증가
        다른 요소로 이동하면 드웰 초기화
        """
        new_element = self.get_element_at(screen_x, screen_y)

        with self._lock:
            if new_element is None:
                self._current = None
                return None

            if self._current and self._current.is_same_element(new_element):
                # 같은 요소 → 드웰 타임 유지 (dwell_start 갱신 X)
                return self._current
            else:
                # 새 요소 → 드웰 초기화
                new_element.dwell_start = time.time()
                self._current = new_element
                return self._current

    @property
    def current_element(self) -> Optional[HoverElement]:
        with self._lock:
            return self._current

    @property
    def dwell_threshold(self) -> float:
        return self._dwell_threshold
