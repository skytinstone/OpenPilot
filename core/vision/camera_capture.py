"""
카메라 캡처 모듈
OpenCV를 통해 카메라 프레임을 스레드 안전하게 캡처
"""
import cv2
import threading
import queue
import time


class CameraCapture:
    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps

        self._cap = None
        self._frame_queue = queue.Queue(maxsize=2)  # 최신 2프레임만 유지
        self._thread = None
        self._stop_event = threading.Event()
        self._is_running = False

    def start(self) -> bool:
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            print("[ERROR] 카메라를 열 수 없습니다.")
            print("       시스템 설정 → 개인 정보 보호 → 카메라 권한을 확인하세요.")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._is_running = True
        print(f"[Camera] 시작됨 — {self.width}x{self.height} @ {self.fps}fps")
        return True

    def _capture_loop(self):
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # 좌우 반전 (거울 모드 — 직관적인 조작감)
            frame = cv2.flip(frame, 1)

            # 큐가 꽉 차면 오래된 프레임 제거 후 새 프레임 추가
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)

    def read(self):
        """최신 프레임 반환. 없으면 None."""
        try:
            return self._frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        self._is_running = False
        print("[Camera] 종료됨")

    @property
    def is_running(self) -> bool:
        return self._is_running


if __name__ == "__main__":
    # 단독 실행 테스트
    cam = CameraCapture()
    if not cam.start():
        exit(1)

    print("카메라 테스트 중... 'q' 키로 종료")
    fps_counter = 0
    start_time = time.time()

    while True:
        frame = cam.read()
        if frame is None:
            continue

        fps_counter += 1
        elapsed = time.time() - start_time
        fps = fps_counter / elapsed if elapsed > 0 else 0

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()
