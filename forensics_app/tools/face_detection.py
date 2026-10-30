"""Face detection with YuNet (OpenCV's CNN detector) on the current image or a live webcam stream."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .corner_detection import read_settings, setting_rows
from .dialogs import center_over
from .hog import choice_row


MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = f"https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/{MODEL_NAME}"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / MODEL_NAME  # models/ is git-ignored
MODEL_MIN_BYTES = 100_000  # anything smaller is a failed or partial download
# Number setting: (label, default, minimum, maximum). Defaults are the Set 6 notebook's.
PARAMETERS: dict[str, tuple[str, float, float, float]] = {
    "score_threshold": ("Confidence", 0.6, 0.05, 1.0),  # faces scoring lower are dropped
    "nms_threshold": ("NMS overlap (IoU)", 0.3, 0.05, 1.0),  # overlapping boxes above this are merged
    "top_k": ("Candidates (max)", 5000, 1, 5000),  # boxes kept before non-maximum suppression
}
INTEGER_SETTINGS = {"top_k"}
SOURCES = ("Image", "Webcam")
LANDMARK_NAMES = ("Right eye", "Left eye", "Nose", "Right mouth corner", "Left mouth corner")
BOX_COLOUR = (0, 255, 0)
LANDMARK_COLOUR = (0, 0, 255)
PREVIEW_SIZE = (640, 480)
PREVIEW_DELAY_MS = 250  # wait for a pause in typing or sliding before detecting again
FRAME_INTERVAL_MS = 15  # between webcam frames; reading a frame already waits for the camera
CAMERA_INDEX = 0  # the system's default camera

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # hide OpenCV 5's DNN backend notices


@dataclass(frozen=True)
class Face:
    box: tuple[int, int, int, int]  # x, y, width, height in px
    landmarks: tuple[tuple[float, float], ...]  # (x, y) of each of LANDMARK_NAMES
    score: float


def ensure_model(parent: tk.Misc | None = None) -> Path:
    """The YuNet model, downloaded on first use; if that fails, the user may pick the .onnx file.

    Raises ``RuntimeError`` when there is no model to use.
    """
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size >= MODEL_MIN_BYTES:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = MODEL_PATH.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, partial)
        if partial.stat().st_size < MODEL_MIN_BYTES:
            raise OSError("the downloaded file is incomplete")
        partial.replace(MODEL_PATH)
        return MODEL_PATH
    except OSError as error:
        partial.unlink(missing_ok=True)
        if parent is None:
            raise RuntimeError(f"Could not download the YuNet model: {error}") from error
        messagebox.showinfo("Face detection", f"Could not download the YuNet model ({error}).\n\n"
                            f"Choose {MODEL_NAME} instead (it is in the Set 6 materials).", parent=parent)
    filename = filedialog.askopenfilename(title=f"Choose {MODEL_NAME}", parent=parent,
                                          filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")])
    if not filename:
        raise RuntimeError("Face detection needs the YuNet model.")
    shutil.copyfile(filename, MODEL_PATH)
    return MODEL_PATH


class FaceDetector:
    """A YuNet detector that follows the size of the frames and the settings it is given."""

    def __init__(self, model: Path) -> None:
        self._model = str(model)
        self._detector: Any = None
        self._settings: tuple[float, float, int] | None = None
        self._size: tuple[int, int] | None = None

    def detect(self, image: Image.Image | np.ndarray, score_threshold: float = 0.6, nms_threshold: float = 0.3,
               top_k: int = 5000) -> list[Face]:
        """The faces in a PIL image or a BGR array (as OpenCV reads them), best first."""
        bgr = image if isinstance(image, np.ndarray) else cv2.cvtColor(np.asarray(image.convert("RGB")),
                                                                         cv2.COLOR_RGB2BGR)
        size = (bgr.shape[1], bgr.shape[0])
        settings = (float(score_threshold), float(nms_threshold), int(top_k))
        if self._detector is None or self._settings != settings:
            self._detector = cv2.FaceDetectorYN.create(self._model, "", size, *settings)
            self._settings, self._size = settings, size
        elif self._size != size:
            self._detector.setInputSize(size)
            self._size = size
        _, rows = self._detector.detect(bgr)
        if rows is None:
            return []
        faces = [Face(tuple(int(v) for v in row[:4]), tuple((float(row[i]), float(row[i + 1]))
                                                            for i in range(4, 14, 2)), float(row[14]))
                 for row in rows]
        return sorted(faces, key=lambda face: face.score, reverse=True)


def draw_faces(image: Image.Image, faces: list[Face], landmarks: bool = True) -> Image.Image:
    """A colour copy of ``image`` with a green box and score per face, and blue landmarks (as on the slide)."""
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    line = max(2, round(min(image.size) / 200))
    font = ImageFont.load_default(max(12, round(min(image.size) / 30)))
    for face in faces:
        x, y, width, height = face.box
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline=BOX_COLOUR, width=line)
        draw.text((x, max(0, y - line)), f"Face: {face.score:.2f}", fill=BOX_COLOUR, font=font, anchor="ld")
        if landmarks:
            radius = line + 1
            for lx, ly in face.landmarks:
                draw.ellipse((lx - radius, ly - radius, lx + radius, ly + radius), fill=LANDMARK_COLOUR)
    return result


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale for display only, up or down, to fit ``size``."""
    scale = min(size[0] / image.width, size[1] / image.height)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                        Image.Resampling.LANCZOS)


def ask_faces(
    parent: tk.Misc, image: Image.Image | None, detector: FaceDetector
) -> tuple[str, dict[str, float], bool, Image.Image, list[Face], Image.Image] | None:
    """Let the user tune YuNet on ``image`` or on the webcam stream; ``None`` if cancelled.

    Returns the source, the settings, whether landmarks are drawn, the analysed picture (the image or the
    captured frame), its faces and the annotated result.
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Face detection (YuNet)")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()
    controls = ttk.Frame(body)
    controls.grid(row=0, column=0, sticky="n")
    preview_side = ttk.Frame(body, padding=(20, 0, 0, 0))
    preview_side.grid(row=0, column=1, sticky="n")

    sources = SOURCES if image is not None else SOURCES[1:]
    source = choice_row(controls, "Source", sources, lambda: source_changed())
    ttk.Label(controls, text="YuNet", padding=(0, 16, 0, 0)).pack(anchor="w")
    values = setting_rows(controls, PARAMETERS, lambda: schedule_preview(), INTEGER_SETTINGS)
    ttk.Label(controls, text="Show", padding=(0, 16, 0, 0)).pack(anchor="w")
    landmarks = tk.BooleanVar(dialog, value=True)
    ttk.Checkbutton(controls, text="Landmarks (eyes, nose, mouth corners)", variable=landmarks,
                    command=lambda: schedule_preview()).pack(anchor="w", pady=(2, 0))
    mirror = tk.BooleanVar(dialog, value=True)
    ttk.Checkbutton(controls, text="Mirror the webcam", variable=mirror).pack(anchor="w", pady=(2, 0))

    ttk.Label(preview_side, text="Preview").pack(anchor="w")
    canvas = tk.Canvas(preview_side, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1],
                       background="#20242b", highlightthickness=0)
    canvas.pack(pady=(4, 2))
    status = ttk.Label(preview_side, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    preview_job: str | None = None
    frame_job: str | None = None
    camera: cv2.VideoCapture | None = None
    latest: tuple[Image.Image, list[Face], Image.Image] | None = None  # picture, faces, annotated
    photo: ImageTk.PhotoImage | None = None

    def settings() -> dict[str, float]:
        return read_settings(values, PARAMETERS, INTEGER_SETTINGS)

    def show(picture: Image.Image, faces: list[Face]) -> None:
        nonlocal latest, photo
        annotated = draw_faces(picture, faces, landmarks.get())
        latest = (picture, faces, annotated)
        photo = ImageTk.PhotoImage(_fit(annotated, PREVIEW_SIZE), master=dialog)  # Tk needs the reference
        canvas.delete("all")
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)

    def schedule_preview() -> None:
        nonlocal preview_job
        if source.get() == "Webcam":  # the next frame picks up the change
            return
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        status.configure(text="Updating preview…")
        preview_job = dialog.after(PREVIEW_DELAY_MS, update_preview)

    def update_preview() -> None:
        nonlocal preview_job, latest
        preview_job = None
        assert image is not None
        try:
            faces = detector.detect(image, **settings())
        except ValueError as error:
            latest = None
            status.configure(text=str(error))
            return
        show(image, faces)
        status.configure(text=f"Found {len(faces)} face{'' if len(faces) == 1 else 's'}.")

    def next_frame() -> None:
        nonlocal frame_job, latest
        frame_job = None
        assert camera is not None
        grabbed, frame = camera.read()
        if not grabbed:
            status.configure(text="The webcam stopped sending frames.")
            return
        if mirror.get():
            frame = cv2.flip(frame, 1)
        picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            faces = detector.detect(frame, **settings())
        except ValueError as error:
            status.configure(text=str(error))
        else:
            show(picture, faces)
            status.configure(text=f"Live: {len(faces)} face{'' if len(faces) == 1 else 's'}. "
                                  "Capture keeps the current frame.")
        frame_job = dialog.after(FRAME_INTERVAL_MS, next_frame)

    def stop_camera() -> None:
        nonlocal camera, frame_job
        if frame_job is not None:
            dialog.after_cancel(frame_job)
            frame_job = None
        if camera is not None:
            camera.release()
            camera = None

    def source_changed() -> None:
        nonlocal camera, latest
        stop_camera()
        latest = None
        canvas.delete("all")
        if source.get() == "Image":
            apply_button.configure(text="Apply")
            schedule_preview()
            return
        apply_button.configure(text="Capture")
        status.configure(text="Opening the webcam…")
        dialog.update_idletasks()
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            stop_camera()
            status.configure(text="Could not open the webcam. Check that no other app is using it and that "
                                  "this terminal or editor may use the camera (System Settings › Privacy).")
            return
        next_frame()

    result = []

    def close() -> None:
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        stop_camera()
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        try:
            current = settings()
        except ValueError as error:
            messagebox.showerror("Face detection", str(error), parent=dialog)
            return
        if source.get() == "Image":
            if preview_job is not None:  # settings changed after the last preview
                update_preview()
        if latest is None:
            messagebox.showerror("Face detection", "There is nothing to keep yet.", parent=dialog)
            return
        picture, faces, annotated = latest
        result.append((source.get(), current, landmarks.get(), picture, faces, annotated))
        close()

    buttons = ttk.Frame(preview_side)
    buttons.pack(fill="x", pady=(16, 0))
    ttk.Button(buttons, text="Cancel", command=close).pack(side="right")
    apply_button = ttk.Button(buttons, text="Apply", command=ok, default="active")
    apply_button.pack(side="right", padx=(0, 6))
    dialog.bind("<Return>", ok)
    dialog.bind("<Escape>", lambda _event: close())
    dialog.protocol("WM_DELETE_WINDOW", close)

    center_over(dialog, parent.winfo_toplevel())
    dialog.deiconify()
    source_changed()
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class FaceDetectionTool(ForensicsTool):
    tool_id = "face_detection"
    title = "Face detection (YuNet)"
    category = "Face recognition"
    description = "Detect faces and their landmarks with YuNet, on the image or a live webcam stream."
    requires_image = False  # the webcam works without an open image
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        detector = FaceDetector(ensure_model(parent))
        choice = ask_faces(parent, document.current, detector)
        if choice is None:
            return None
        source, current, landmarks, picture, faces, annotated = choice
        document_path = None
        if source == "Webcam":  # the captured frame becomes the document, so undo shows it without boxes
            with tempfile.NamedTemporaryFile(prefix="webcam-", suffix=".png", delete=False) as file:
                document_path = Path(file.name)
            picture.save(document_path)
        details: dict[str, Any] = {
            "Operation": "Face detection with YuNet",
            "Source": "Webcam frame" if source == "Webcam" else "Image",
            "Confidence threshold": f"{current['score_threshold']:g}",
            "NMS threshold": f"{current['nms_threshold']:g}",
            "Candidates (max)": current["top_k"],
            "Faces found": len(faces),
        }
        for number, face in enumerate(faces, start=1):
            x, y, width, height = face.box
            details[f"Face {number}"] = f"({x}, {y}) {width} × {height} px, score {face.score:.2f}"
        return ToolResult(
            image=annotated,
            message=f"Found {len(faces)} face{'' if len(faces) == 1 else 's'} with YuNet.",
            details=details,
            document_path=document_path,
        )
