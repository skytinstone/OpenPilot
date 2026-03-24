"""
EyePreprocessor — 눈 영역 ROI Crop + CLAHE 전처리
====================================================
동양인(아시아인) 사용자의 작은 눈 영역에서 홍채 랜드마크 신뢰도를 높이기 위한
전처리 파이프라인.

파이프라인:
  1. 얼굴 바운딩 박스 → 양안 영역 ROI 계산
  2. ROI Crop + 확대 (2~3배 Zoom)
  3. CLAHE 적용 (대비 극대화 → 홍채/동공 경계 선명화)
  4. 확대된 프레임을 MediaPipe에 재입력

효과:
  - 홍채 영역 픽셀 수 2~3배 증가 → 랜드마크 정밀도 향상
  - CLAHE로 조명 변화에 강건 → 어두운 환경에서도 홍채 검출
"""
import cv2
import numpy as np
from typing import Optional, Tuple


class EyePreprocessor:
    """얼굴 랜드마크 기반 눈 영역 전처리기"""

    def __init__(self,
                 zoom_factor: float = 2.5,
                 clahe_clip: float = 3.0,
                 clahe_grid: int = 8,
                 padding_ratio: float = 0.6):
        """
        Args:
            zoom_factor:   ROI 확대 배율 (2.0~3.0 권장)
            clahe_clip:    CLAHE clipLimit (2.0~4.0, 높을수록 대비 강함)
            clahe_grid:    CLAHE tileGridSize (8이 표준)
            padding_ratio: 눈 주변 여백 비율 (0.5~0.8, 눈 크기 대비)
        """
        self._zoom = zoom_factor
        self._padding = padding_ratio
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid),
        )
        self._last_roi: Optional[Tuple[int, int, int, int]] = None

    def preprocess(self, frame: np.ndarray,
                   face_landmarks: list) -> Tuple[np.ndarray, dict]:
        """
        원본 프레임 + 얼굴 랜드마크 → 전처리된 프레임 + ROI 메타데이터.

        Args:
            frame:          BGR 원본 프레임
            face_landmarks: MediaPipe 478 랜드마크 리스트

        Returns:
            (enhanced_frame, roi_meta)
            roi_meta: {
                "applied": bool,        # 전처리 적용 여부
                "roi": (x, y, w, h),    # 원본 프레임 기준 ROI
                "scale": float,         # 확대 비율
            }
        """
        h, w = frame.shape[:2]

        # 양안 영역 바운딩 박스 계산
        roi = self._compute_eye_roi(face_landmarks, w, h)
        if roi is None:
            return frame, {"applied": False}

        rx, ry, rw, rh = roi
        self._last_roi = roi

        # ROI Crop
        eye_crop = frame[ry:ry + rh, rx:rx + rw]
        if eye_crop.size == 0:
            return frame, {"applied": False}

        # CLAHE 적용 (LAB 색공간에서 L 채널만)
        enhanced = self._apply_clahe(eye_crop)

        # 확대: ROI를 원본 프레임 크기로 리사이즈 후 원본에 합성
        # → MediaPipe에 전체 프레임을 넣되, 눈 영역이 더 크게 보이도록
        zoom_w = int(rw * self._zoom)
        zoom_h = int(rh * self._zoom)
        zoomed = cv2.resize(enhanced, (zoom_w, zoom_h),
                            interpolation=cv2.INTER_CUBIC)

        # 확대된 눈 영역을 원본 프레임 중앙에 합성
        out = frame.copy()
        # 중앙 배치
        cx = (w - zoom_w) // 2
        cy = (h - zoom_h) // 2
        # 프레임 범위 클리핑
        src_x1 = max(0, -cx)
        src_y1 = max(0, -cy)
        dst_x1 = max(0, cx)
        dst_y1 = max(0, cy)
        copy_w = min(zoom_w - src_x1, w - dst_x1)
        copy_h = min(zoom_h - src_y1, h - dst_y1)

        if copy_w > 0 and copy_h > 0:
            out[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = \
                zoomed[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]

        return out, {
            "applied": True,
            "roi": roi,
            "scale": self._zoom,
        }

    def enhance_roi_only(self, frame: np.ndarray,
                         face_landmarks: list) -> np.ndarray:
        """
        ROI 영역만 CLAHE 적용하여 원본 프레임에 덮어씌우기.
        (MediaPipe 입력용이 아닌, 단순 대비 향상)
        """
        h, w = frame.shape[:2]
        roi = self._compute_eye_roi(face_landmarks, w, h)
        if roi is None:
            return frame

        rx, ry, rw, rh = roi
        eye_crop = frame[ry:ry + rh, rx:rx + rw]
        if eye_crop.size == 0:
            return frame

        enhanced = self._apply_clahe(eye_crop)
        out = frame.copy()
        out[ry:ry + rh, rx:rx + rw] = enhanced
        return out

    def _compute_eye_roi(self, landmarks, w: int, h: int) \
            -> Optional[Tuple[int, int, int, int]]:
        """
        양안 영역 바운딩 박스 (padding 포함).
        MediaPipe 랜드마크 기준:
          왼눈: #263 (좌), #362 (우), #386 (상), #374 (하)
          오른눈: #33 (좌), #133 (우), #159 (상), #145 (하)
        """
        try:
            # 주요 눈 랜드마크에서 bounding box 추출
            eye_indices = [
                # 왼눈
                263, 362, 386, 374,
                # 오른눈
                33, 133, 159, 145,
                # 홍채
                468, 473,
            ]
            xs, ys = [], []
            for idx in eye_indices:
                if hasattr(landmarks[idx], 'x'):
                    xs.append(landmarks[idx].x)
                    ys.append(landmarks[idx].y)
                else:
                    xs.append(landmarks[idx][0])
                    ys.append(landmarks[idx][1])

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # 패딩 추가 (눈 영역 주변 여백)
            eye_w = max_x - min_x
            eye_h = max_y - min_y
            pad_x = eye_w * self._padding
            pad_y = eye_h * self._padding * 1.5  # 세로 패딩 더 크게 (눈썹 포함)

            rx = int(max(0, (min_x - pad_x) * w))
            ry = int(max(0, (min_y - pad_y) * h))
            rw = int(min(w - rx, (eye_w + 2 * pad_x) * w))
            rh = int(min(h - ry, (eye_h + 2 * pad_y) * h))

            if rw < 30 or rh < 15:  # 너무 작은 ROI 무시
                return None

            return (rx, ry, rw, rh)

        except (IndexError, AttributeError):
            return None

    def _apply_clahe(self, bgr_crop: np.ndarray) -> np.ndarray:
        """
        CLAHE 적용 (LAB 색공간의 L 채널).
        조명 변화에 강건하면서 색상은 보존.
        """
        lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self._clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    @property
    def last_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._last_roi
