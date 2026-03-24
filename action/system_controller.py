"""
시스템 제어 — macOS 볼륨, 밝기, 삭제 등

음성 명령에서 "system" 액션으로 호출되는 OS 수준 제어 함수들.
"""
import subprocess
import time


def execute_system_command(command: str):
    """시스템 명령 실행"""
    handler = _COMMANDS.get(command)
    if handler:
        handler()
    else:
        print(f"[System] 알 수 없는 명령: {command}")


def _volume_up():
    _osascript("set volume output volume ((output volume of (get volume settings)) + 10)")


def _volume_down():
    _osascript("set volume output volume ((output volume of (get volume settings)) - 10)")


def _mute():
    _osascript("set volume output muted (not (output muted of (get volume settings)))")


def _brightness_up():
    # macOS 밝기: 미디어 키 시뮬레이션
    try:
        from Quartz import (CGEventCreateKeyboardEvent, CGEventPost,
                            kCGHIDEventTap, CGEventSetIntegerValueField)
        # F2 (brightness up) = NX keycode 0x90
        e = CGEventCreateKeyboardEvent(None, 144, True)
        CGEventPost(kCGHIDEventTap, e)
        time.sleep(0.05)
        e = CGEventCreateKeyboardEvent(None, 144, False)
        CGEventPost(kCGHIDEventTap, e)
    except Exception:
        _osascript('tell application "System Events" to key code 144')


def _brightness_down():
    try:
        from Quartz import (CGEventCreateKeyboardEvent, CGEventPost,
                            kCGHIDEventTap)
        # F1 (brightness down) = NX keycode 0x91
        e = CGEventCreateKeyboardEvent(None, 145, True)
        CGEventPost(kCGHIDEventTap, e)
        time.sleep(0.05)
        e = CGEventCreateKeyboardEvent(None, 145, False)
        CGEventPost(kCGHIDEventTap, e)
    except Exception:
        _osascript('tell application "System Events" to key code 145')


def _delete():
    """백스페이스 키"""
    try:
        from Quartz import (CGEventCreateKeyboardEvent, CGEventPost,
                            kCGHIDEventTap)
        e = CGEventCreateKeyboardEvent(None, 51, True)  # 51 = delete/backspace
        CGEventPost(kCGHIDEventTap, e)
        time.sleep(0.03)
        e = CGEventCreateKeyboardEvent(None, 51, False)
        CGEventPost(kCGHIDEventTap, e)
    except Exception:
        pass


def _osascript(script: str):
    """AppleScript 실행"""
    try:
        subprocess.run(["osascript", "-e", script],
                       capture_output=True, timeout=5)
    except Exception as e:
        print(f"[System] osascript 실패: {e}")


_COMMANDS = {
    "volume_up":       _volume_up,
    "volume_down":     _volume_down,
    "mute":            _mute,
    "brightness_up":   _brightness_up,
    "brightness_down": _brightness_down,
    "delete":          _delete,
}
