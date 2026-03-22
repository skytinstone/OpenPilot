"""
마이크 오디오 캡처 — sounddevice 기반

Whisper 는 16kHz mono float32 를 요구함.
start_recording() / stop_recording() 으로 녹음 구간 제어.
"""
import threading
import numpy as np
from typing import Optional

SAMPLE_RATE = 16_000   # Whisper 요구 샘플레이트


class AudioCapture:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self._sr       = sample_rate
        self._recording = False
        self._buffer   = []
        self._lock     = threading.Lock()
        self._stream   = None
        self._level    = 0.0   # 실시간 음량 (0~1, UI 시각화용)

    # ── 녹음 제어 ────────────────────────────────────────────────

    def start_recording(self):
        """녹음 시작"""
        import sounddevice as sd
        self._buffer    = []
        self._recording = True

        self._stream = sd.InputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        )
        self._stream.start()

    def stop_recording(self) -> np.ndarray:
        """
        녹음 종료 → 녹음된 오디오를 float32 numpy 배열로 반환.
        배열이 비어 있으면 길이 0 배열 반환.
        """
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if self._buffer:
                audio = np.concatenate(self._buffer, axis=0).flatten()
            else:
                audio = np.array([], dtype=np.float32)

        self._level = 0.0
        return audio

    # ── 내부 콜백 ────────────────────────────────────────────────

    def _callback(self, indata, frames, time_info, status):
        if not self._recording:
            return
        chunk = indata.copy()
        with self._lock:
            self._buffer.append(chunk)
        # RMS 음량 계산 (0~1)
        self._level = float(np.sqrt(np.mean(chunk ** 2)) * 6)
        self._level = min(self._level, 1.0)

    # ── 상태 조회 ────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def level(self) -> float:
        """현재 마이크 입력 음량 (0~1)"""
        return self._level

    @property
    def recorded_seconds(self) -> float:
        with self._lock:
            total = sum(len(b) for b in self._buffer)
        return total / self._sr
