"""
눈 추적 모듈 — MediaPipe Tasks API (v0.10.20+)
mp.solutions 가 제거된 최신 버전 대응

FaceLandmarker 478 랜드마크:
  0~467  : 얼굴 윤곽
  468~472: 오른쪽 홍채 (468 = 중심)
  473~477: 왼쪽  홍채 (473 = 중심)
"""
import cv2
import numpy as np
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

# ── 모델 파일 자동 다운로드 설정 ────────────────────────────────
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print("[EyeTracker] 모델 파일 다운로드 중... (최초 1회)")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[EyeTracker] 모델 다운로드 완료")


# ── 랜드마크 인덱스 ─────────────────────────────────────────────
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER  = 473

LEFT_EYE_TOP    = 386
LEFT_EYE_BOTTOM = 374
LEFT_EYE_LEFT   = 263
LEFT_EYE_RIGHT  = 362

RIGHT_EYE_TOP    = 159
RIGHT_EYE_BOTTOM = 145
RIGHT_EYE_LEFT   = 133
RIGHT_EYE_RIGHT  = 33

LEFT_EYE_OUTLINE  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE_OUTLINE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]


@dataclass
class EyeData:
    left_iris:          Tuple[float, float]
    right_iris:         Tuple[float, float]
    left_eye_center:    Tuple[float, float]
    right_eye_center:   Tuple[float, float]
    left_eye_openness:  float
    right_eye_openness: float
    avg_iris:           Tuple[float, float]
    left_gaze_offset:   Tuple[float, float]
    right_gaze_offset:  Tuple[float, float]


class EyeTracker:
    def __init__(self, config: dict):
        _ensure_model()

        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks import python as mp_python

        cfg = config.get("eye_tracking", {})

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=cfg.get("max_num_faces", 1),
            min_face_detection_confidence=cfg.get("min_detection_confidence", 0.7),
            min_face_presence_confidence=cfg.get("min_tracking_confidence", 0.7),
            min_tracking_confidence=cfg.get("min_tracking_confidence", 0.7),
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._mp = mp
        print("[EyeTracker] MediaPipe FaceLandmarker 초기화 완료")

    def process(self, frame: np.ndarray) -> Optional[EyeData]:
        """BGR 프레임 → EyeData 반환 (얼굴 미감지 시 None)"""
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        lms = result.face_landmarks[0]   # NormalizedLandmark 리스트
        h, w = frame.shape[:2]

        def lm(idx) -> Tuple[float, float]:
            return (lms[idx].x, lms[idx].y)

        # 홍채 중심
        left_iris  = lm(LEFT_IRIS_CENTER)
        right_iris = lm(RIGHT_IRIS_CENTER)

        # 눈 윤곽 평균 중심
        lep = [lm(i) for i in LEFT_EYE_OUTLINE]
        rep = [lm(i) for i in RIGHT_EYE_OUTLINE]
        left_eye_center  = (sum(p[0] for p in lep)/len(lep), sum(p[1] for p in lep)/len(lep))
        right_eye_center = (sum(p[0] for p in rep)/len(rep), sum(p[1] for p in rep)/len(rep))

        # 눈 개폐 비율
        def openness(top, bot, left, right):
            v = abs(lm(top)[1]   - lm(bot)[1])
            h = abs(lm(left)[0]  - lm(right)[0]) + 1e-6
            return v / h

        lo = openness(LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,  LEFT_EYE_LEFT,  LEFT_EYE_RIGHT)
        ro = openness(RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)

        # 시선 오프셋
        def gaze_offset(iris, el, er, et, eb):
            cx = (lm(el)[0] + lm(er)[0]) / 2
            cy = (lm(et)[1] + lm(eb)[1]) / 2
            hw = abs(lm(er)[0] - lm(el)[0]) / 2 + 1e-6
            hh = abs(lm(eb)[1] - lm(et)[1]) / 2 + 1e-6
            return ((iris[0] - cx) / hw, (iris[1] - cy) / hh)

        lg = gaze_offset(left_iris,  LEFT_EYE_LEFT,  LEFT_EYE_RIGHT,  LEFT_EYE_TOP,  LEFT_EYE_BOTTOM)
        rg = gaze_offset(right_iris, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)

        avg_iris = ((left_iris[0]+right_iris[0])/2, (left_iris[1]+right_iris[1])/2)

        return EyeData(
            left_iris=left_iris, right_iris=right_iris,
            left_eye_center=left_eye_center, right_eye_center=right_eye_center,
            left_eye_openness=lo, right_eye_openness=ro,
            avg_iris=avg_iris,
            left_gaze_offset=lg, right_gaze_offset=rg,
        )

    def draw_debug(self, frame: np.ndarray, eye_data: Optional[EyeData]) -> np.ndarray:
        if eye_data is None:
            cv2.putText(frame, "얼굴 미감지", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame

        h, w = frame.shape[:2]
        def to_px(n): return (int(n[0]*w), int(n[1]*h))

        cv2.circle(frame, to_px(eye_data.left_iris),  5, (0, 255, 255), -1)
        cv2.circle(frame, to_px(eye_data.right_iris), 5, (0, 255, 255), -1)
        cv2.circle(frame, to_px(eye_data.left_eye_center),  3, (255, 0, 0), -1)
        cv2.circle(frame, to_px(eye_data.right_eye_center), 3, (255, 0, 0), -1)

        cv2.putText(frame, f"L:{eye_data.left_eye_openness:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"R:{eye_data.right_eye_openness:.2f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        lc = to_px(eye_data.left_eye_center)
        lo = eye_data.left_gaze_offset
        cv2.arrowedLine(frame, lc,
                        (lc[0]+int(lo[0]*20), lc[1]+int(lo[1]*20)),
                        (0, 165, 255), 2, tipLength=0.4)

        rc = to_px(eye_data.right_eye_center)
        ro = eye_data.right_gaze_offset
        cv2.arrowedLine(frame, rc,
                        (rc[0]+int(ro[0]*20), rc[1]+int(ro[1]*20)),
                        (0, 165, 255), 2, tipLength=0.4)
        return frame

    def close(self):
        self._landmarker.close()
        print("[EyeTracker] 종료됨")
