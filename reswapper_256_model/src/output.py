import cv2


class FrameOutput:
    """Handles preview window display and optional virtual camera output.

    Virtual camera requires OBS Studio with OBS Virtual Camera installed and
    active. If initialization fails the pipeline degrades to preview-only.
    """

    def __init__(self, window_title: str, use_virtual_cam: bool, width: int, height: int, fps: int):
        self._title = window_title
        self._use_vcam = use_virtual_cam
        self._width = width
        self._height = height
        self._fps = fps
        self._vcam = None
        self._last_key = -1
        self._key_ready = False

    def start(self):
        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._title, self._width, self._height)

        if self._use_vcam:
            try:
                import pyvirtualcam
                self._vcam = pyvirtualcam.Camera(
                    width=self._width,
                    height=self._height,
                    fps=self._fps,
                    print_fps=False,
                )
                print(f"Virtual camera active: {self._vcam.device}")
            except Exception as exc:
                print(f"[warn] Virtual camera unavailable — preview only. ({exc})")
                self._use_vcam = False

    def send(self, frame_bgr):
        cv2.imshow(self._title, frame_bgr)
        # Pump the GUI event loop RIGHT after imshow, before any blocking operation
        # (vcam sleep_until_next_frame or pipeline time.sleep). Calling waitKey after
        # a 33 ms sleep leaves only a 1 ms window to catch keypresses — effectively
        # making Q unresponsive. Capturing here gives the full inter-frame interval.
        self._last_key = cv2.waitKey(1)
        self._key_ready = True

        if self._vcam:
            resized = cv2.resize(frame_bgr, (self._width, self._height))
            # pyvirtualcam expects RGB; OpenCV produces BGR
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self._vcam.send(rgb)
            self._vcam.sleep_until_next_frame()

    def quit_requested(self) -> bool:
        # Use the key already captured in send() for this frame if available.
        # Fall back to a fresh waitKey only in the startup path before the first
        # send() call (result=None, frame=None branch in pipeline.py).
        if self._key_ready:
            k = self._last_key
            self._key_ready = False
            return (k & 0xFF) == ord('q')
        return (cv2.waitKey(1) & 0xFF) == ord('q')

    def stop(self):
        if self._vcam:
            self._vcam.close()
        cv2.destroyAllWindows()
