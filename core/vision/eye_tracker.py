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
    face_landmarks:     Optional[list] = None   # raw 478 landmarks (시각화용)


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
            face_landmarks=lms,
        )

    def draw_debug(self, frame: np.ndarray, eye_data: Optional[EyeData],
                   gaze_x: int = -1, gaze_y: int = -1,
                   screen_w: int = 0, screen_h: int = 0) -> np.ndarray:
        """
        시선 추적 시각화

        - 눈 윤곽선 (eye socket outline)
        - 홍채 원 + 중심점
        - 레티클 (눈 중심 기준 홍채 편차)
        - 시선 방향 화살표
        - 눈 개폐 게이지 (좌/우)
        - gaze offset 수치
        - 미니맵 (현재 시선이 화면 어디를 향하는지)
        """
        h, w = frame.shape[:2]

        if eye_data is None:
            cv2.putText(frame, "No face detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2)
            return frame

        def px(n):
            if hasattr(n, 'x'):
                return (int(n.x * w), int(n.y * h))
            return (int(n[0] * w), int(n[1] * h))

        lms = eye_data.face_landmarks

        # ── 1. 눈 윤곽선 ───────────────────────────────────────
        if lms:
            for outline_idxs, color in [
                (LEFT_EYE_OUTLINE,  (100, 220, 100)),   # 왼눈 초록
                (RIGHT_EYE_OUTLINE, (100, 160, 255)),   # 오른눈 파랑
            ]:
                pts = [px(lms[i]) for i in outline_idxs]
                for i in range(len(pts)):
                    cv2.line(frame, pts[i], pts[(i+1) % len(pts)], color, 1, cv2.LINE_AA)

        # ── 2. 홍채 원 ─────────────────────────────────────────
        for iris, eye_center, color in [
            (eye_data.left_iris,  eye_data.left_eye_center,  (0, 255, 180)),
            (eye_data.right_iris, eye_data.right_eye_center, (0, 200, 255)),
        ]:
            iris_px   = px(iris)
            center_px = px(eye_center)

            # 눈 폭으로 홍채 반지름 추정
            eye_w_norm = abs(iris[0] - eye_center[0]) * 4 + 0.02
            iris_r = max(8, int(eye_w_norm * w * 1.5))
            iris_r = min(iris_r, 22)   # 최대 22px

            # 홍채 원 (반투명)
            overlay = frame.copy()
            cv2.circle(overlay, iris_px, iris_r, color, 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # 홍채 중심점
            cv2.circle(frame, iris_px, 3, color, -1, cv2.LINE_AA)

            # 눈 중심 십자선
            cv2.line(frame, (center_px[0]-8, center_px[1]),
                     (center_px[0]+8, center_px[1]), (180, 180, 180), 1)
            cv2.line(frame, (center_px[0], center_px[1]-8),
                     (center_px[0], center_px[1]+8), (180, 180, 180), 1)

        # ── 3. 레티클 — 눈 중심 기준 홍채 편차 ─────────────────
        for iris, offset, gaze_color, label in [
            (eye_data.left_iris,  eye_data.left_gaze_offset,  (0, 255, 180), "L"),
            (eye_data.right_iris, eye_data.right_gaze_offset, (0, 200, 255), "R"),
        ]:
            iris_px = px(iris)
            ox, oy  = offset

            # 시선 방향 화살표 (오프셋 40배 증폭)
            arrow_end = (iris_px[0] + int(ox * 40), iris_px[1] + int(oy * 40))
            cv2.arrowedLine(frame, iris_px, arrow_end, gaze_color, 2,
                            cv2.LINE_AA, tipLength=0.35)

            # 오프셋 수치 텍스트
            cv2.putText(frame, f"{label}({ox:+.2f},{oy:+.2f})",
                        (iris_px[0] + 16, iris_px[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, gaze_color, 1)

        # ── 4. 눈 개폐 게이지 (우측 세로 바) ───────────────────
        bar_x = w - 28
        for i, (openness, label, color) in enumerate([
            (eye_data.left_eye_openness,  "L", (0, 255, 180)),
            (eye_data.right_eye_openness, "R", (0, 200, 255)),
        ]):
            bx = bar_x + i * 18
            by_top, by_bot = 50, 130
            bar_h_filled = int((by_bot - by_top) * min(openness / 0.35, 1.0))

            cv2.rectangle(frame, (bx, by_top), (bx+10, by_bot), (40, 40, 40), -1)
            cv2.rectangle(frame, (bx, by_bot - bar_h_filled), (bx+10, by_bot),
                          color, -1)
            cv2.rectangle(frame, (bx, by_top), (bx+10, by_bot), (80, 80, 80), 1)
            cv2.putText(frame, label, (bx, by_top - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
            cv2.putText(frame, f"{openness:.2f}", (bx - 4, by_bot + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

        # ── 5. 미니맵 — 현재 시선 화면 위치 ────────────────────
        if gaze_x >= 0 and screen_w > 0 and screen_h > 0:
            MAP_W, MAP_H = 160, 100
            MAP_X, MAP_Y = w - MAP_W - 8, h - MAP_H - 8

            # 배경
            overlay = frame.copy()
            cv2.rectangle(overlay, (MAP_X, MAP_Y),
                          (MAP_X + MAP_W, MAP_Y + MAP_H), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (MAP_X, MAP_Y),
                          (MAP_X + MAP_W, MAP_Y + MAP_H), (80, 80, 80), 1)

            # 격자선
            for gx in [MAP_X + MAP_W//3, MAP_X + 2*MAP_W//3]:
                cv2.line(frame, (gx, MAP_Y), (gx, MAP_Y+MAP_H), (50,50,50), 1)
            for gy in [MAP_Y + MAP_H//3, MAP_Y + 2*MAP_H//3]:
                cv2.line(frame, (MAP_X, gy), (MAP_X+MAP_W, gy), (50,50,50), 1)

            # 시선 포인터
            dot_x = MAP_X + int(gaze_x / screen_w * MAP_W)
            dot_y = MAP_Y + int(gaze_y / screen_h * MAP_H)
            dot_x = max(MAP_X + 4, min(MAP_X + MAP_W - 4, dot_x))
            dot_y = max(MAP_Y + 4, min(MAP_Y + MAP_H - 4, dot_y))

            cv2.circle(frame, (dot_x, dot_y), 6, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (dot_x, dot_y), 6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, "GAZE MAP", (MAP_X + 4, MAP_Y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

        return frame

    def close(self):
        self._landmarker.close()
        print("[EyeTracker] 종료됨")
