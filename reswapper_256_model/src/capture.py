import cv2
import threading
import time


class CameraCapture:
    """Background-thread camera capture that always exposes the latest frame.

    Using CAP_DSHOW on Windows avoids the MSMF buffering overhead that can add
    100+ ms of latency. BUFFERSIZE=1 ensures we never read a stale frame.
    """

    def __init__(self, camera_index=0, width=1280, height=720, fps=30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None
        self._latest_frame = None
        self._latest_time = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        # CAP_DSHOW bypasses Windows MSMF, which buffers several frames internally
        # and adds 100–200 ms of latency before the first frame is readable.
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Keep only the newest frame in the driver buffer; prevents reading stale frames
        # when the processing loop is slower than the camera.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")
        self._running = True
        # daemon=True so this thread is killed automatically if the main process exits
        self._thread = threading.Thread(target=self._loop, daemon=True, name="capture")
        self._thread.start()

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                ts = time.perf_counter()
                # Lock protects _latest_frame/_latest_time which are read by the main thread
                with self._lock:
                    self._latest_frame = frame
                    self._latest_time = ts

    def read(self):
        """Return (timestamp, frame) of the most recent captured frame."""
        with self._lock:
            return self._latest_time, self._latest_frame

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
