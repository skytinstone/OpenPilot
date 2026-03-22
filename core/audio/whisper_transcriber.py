"""
Whisper STT 모듈

지원 백엔드 (자동 감지):
  1. faster-whisper  (pip install faster-whisper)  — 권장, 4배 빠름
  2. openai-whisper  (pip install openai-whisper)  — 폴백

모델 크기별 속도/정확도:
  tiny   ~39MB  — 매우 빠름, 낮은 정확도
  base   ~74MB  — 빠름, 준수한 정확도       ← 기본값
  small  ~244MB — 보통, 좋은 정확도
  medium ~769MB — 느림, 높은 정확도
"""
import os
import tempfile
import numpy as np
from typing import Optional


class WhisperTranscriber:
    def __init__(self, model_size: str = "base"):
        self._model_size  = model_size
        self._model       = None
        self._use_faster  = False
        self._load()

    # ── 모델 로드 ────────────────────────────────────────────────

    def _load(self):
        # faster-whisper 우선 시도
        try:
            from faster_whisper import WhisperModel
            print(f"[Whisper] faster-whisper 로드 중: {self._model_size} ...")
            self._model      = WhisperModel(self._model_size, device="cpu",
                                            compute_type="int8")
            self._use_faster = True
            print("[Whisper] faster-whisper 준비 완료")
            return
        except ImportError:
            pass

        # openai-whisper 폴백
        try:
            import whisper
            print(f"[Whisper] openai-whisper 로드 중: {self._model_size} ...")
            self._model = whisper.load_model(self._model_size)
            print("[Whisper] openai-whisper 준비 완료")
        except ImportError:
            raise RuntimeError(
                "Whisper 패키지가 없습니다.\n"
                "  pip install faster-whisper  (권장)\n"
                "  pip install openai-whisper  (대안)"
            )

    # ── 변환 ────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray,
                   language: Optional[str] = None) -> str:
        """
        float32 numpy 배열(16kHz mono) → 텍스트 반환.
        language=None 이면 자동 감지.
        """
        if audio is None or len(audio) < 3200:   # < 0.2초
            return ""

        if self._use_faster:
            return self._transcribe_faster(audio, language)
        else:
            return self._transcribe_openai(audio, language)

    def _transcribe_faster(self, audio: np.ndarray,
                           language: Optional[str]) -> str:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, audio, 16_000)
            kwargs = {"language": language} if language else {}
            segments, _ = self._model.transcribe(tmp, **kwargs)
            return " ".join(s.text for s in segments).strip()
        finally:
            os.unlink(tmp)

    def _transcribe_openai(self, audio: np.ndarray,
                           language: Optional[str]) -> str:
        kwargs = {"language": language, "fp16": False} if language else {"fp16": False}
        result = self._model.transcribe(audio, **kwargs)
        return result.get("text", "").strip()
