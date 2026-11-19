"""Object detection with YOLO (Set 9.1): run a pretrained Ultralytics YOLO11 model on the opened image.

``yolo11n.pt`` is the nano YOLO11 trained on COCO's 80 classes; Ultralytics downloads it into ``models/`` the first
time. The boxes are drawn as in the Set 9 notebook: with matplotlib, one ``tab20`` colour per class, or with
Ultralytics' own ``plot()``. ``plot()`` returns a BGR array, as OpenCV does, so shown as RGB without converting it the
red and blue channels swap: that is why the people on the slide are blue.

Ultralytics (and PyTorch) are imported only when the tool runs, so the rest of the app starts without them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .cnn_classification import _figure_image
from .corner_detection import read_settings, setting_rows
from .dialogs import center_over
from .eigenfaces import _fit
from .hog import choice_row


MODEL_FOLDER = Path(__file__).resolve().parents[2] / "models"  # models/ is git-ignored
# One button each in the dialog. Ultralytics downloads its own pretrained weights when the file is missing.
MODELS = {"YOLO11n (COCO)": MODEL_FOLDER / "yolo11n.pt"}
# Number setting: (label, default, minimum, maximum). The defaults are Ultralytics'.
PARAMETERS: dict[str, tuple[str, float, float, float]] = {
    "confidence": ("Confidence", 0.25, 0.05, 1.0),  # boxes scoring lower are dropped
    "iou": ("NMS overlap (IoU)", 0.7, 0.05, 1.0),  # overlapping boxes of a class above this are merged
}
VIEWS = ("Detected objects", "Ultralytics plot()", "plot() without BGR → RGB")
COLOURS = matplotlib.colormaps["tab20"]  # the notebook's colour map, one colour per class
FIGURE_INCHES = 10  # the longer side of the matplotlib figure, so the 12 pt labels look the same on any image
LISTED = 20  # detections listed one by one in the results panel
PREVIEW_SIZE = (720, 560)
PREVIEW_DELAY_MS = 250  # wait for a pause in typing or sliding before detecting again
WHY_BLUE = ("plot() returns a BGR array (OpenCV's channel order); shown as RGB, red and blue swap, so the people "
            "are blue. Converting it with cv2.cvtColor(..., cv2.COLOR_BGR2RGB) fixes the colours.")
MISSING_ULTRALYTICS = "Install Ultralytics first: python -m pip install ultralytics"


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    confidence: float
    class_id: int
    label: str


def detections_from(result: Any) -> list[Detection]:
    """The boxes of an Ultralytics ``Results`` object, most confident first."""
    boxes = result.boxes
    corners = np.asarray(boxes.xyxy.cpu(), dtype=float).reshape(-1, 4)
    confidences = np.asarray(boxes.conf.cpu(), dtype=float).reshape(-1)
    classes = np.asarray(boxes.cls.cpu()).reshape(-1).astype(int)
    detections = [Detection(tuple(map(float, box)), float(confidence), int(class_id), str(result.names[class_id]))
                  for box, confidence, class_id in zip(corners, confidences, classes)]
    return sorted(detections, key=lambda detection: -detection.confidence)


@dataclass
class YoloDetector:
    """A YOLO model loaded with Ultralytics."""

    path: Path
    model: Any

    @classmethod
    def load(cls, path: Path) -> YoloDetector:
        """``ValueError`` when Ultralytics is missing or cannot read (or download) the weights."""
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ValueError(MISSING_ULTRALYTICS) from error
        try:
            return cls(path, YOLO(str(path)))
        except Exception as error:  # Ultralytics raises many kinds: missing file, failed download, bad checkpoint
            raise ValueError(f"Ultralytics could not load {path.name}: {error}") from error

    @property
    def classes(self) -> dict[int, str]:
        return dict(self.model.names)

    def detect(self, image: Image.Image, confidence: float = 0.25, iou: float = 0.7) -> tuple[Any, list[Detection]]:
        """The Ultralytics result for ``image`` and its detections."""
        result = self.model(image.convert("RGB"), conf=confidence, iou=iou, verbose=False)[0]
        return result, detections_from(result)


_loaded: dict[Path, YoloDetector] = {}  # loading imports PyTorch and reads the weights, so it is done once


def load_detector(path: Path) -> YoloDetector:
    if path not in _loaded:
        _loaded[path] = YoloDetector.load(path)
    return _loaded[path]


def show_detections(image: Image.Image, detections: list[Detection], title: str = "Detected Objects") -> Image.Image:
    """The image with a box and a "label confidence" tag per detection, drawn with matplotlib as in the notebook."""
    picture = image.convert("RGB")
    scale = FIGURE_INCHES / max(picture.size)
    figure = Figure(figsize=(picture.width * scale, picture.height * scale + 0.5), layout="constrained")
    axes = figure.add_subplot()
    axes.imshow(picture)
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        colour = COLOURS(detection.class_id % COLOURS.N)
        axes.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=colour, facecolor="none"))
        axes.text(x1, y1, f"{detection.label} {detection.confidence:.2f}", color="white", fontsize=12,
                  va="bottom", bbox={"facecolor": colour, "edgecolor": "none", "pad": 2}, clip_on=True)
    axes.set_xlim(0, picture.width)
    axes.set_ylim(picture.height, 0)
    axes.axis("off")
    axes.set_title(title)
    return _figure_image(figure)


def show_plot(result: Any, to_rgb: bool = True) -> Image.Image:
    """Ultralytics' own drawing; ``to_rgb=False`` shows its BGR array as if it were RGB."""
    pixels = np.asarray(result.plot())
    return Image.fromarray(np.ascontiguousarray(pixels[..., ::-1] if to_rgb else pixels))


def show_view(image: Image.Image, result: Any, detections: list[Detection], view: str = VIEWS[0]) -> Image.Image:
    if view == VIEWS[0]:
        return show_detections(image, detections)
    if view == VIEWS[1]:
        return show_plot(result)
    if view == VIEWS[2]:
        return show_plot(result, to_rgb=False)
    raise ValueError(f"Unknown view: {view}")


def summary(detections: list[Detection]) -> str:
    """ "4 × person, 1 × bus", most frequent class first."""
    counts = Counter(detection.label for detection in detections)
    return ", ".join(f"{count} × {label}" for label, count in counts.most_common()) or "nothing"


def detection_details(detector: YoloDetector, settings: dict[str, float], view: str, result: Any,
                      detections: list[Detection]) -> dict[str, object]:
    """The settings and every detection, for the results panel."""
    details: dict[str, object] = {
        "Operation": "Object detection with YOLO",
        "Model": str(detector.path),
        "Classes the model knows": len(detector.classes),
        "Confidence threshold": f"{settings['confidence']:g}",
        "NMS threshold (IoU)": f"{settings['iou']:g}",
        "Detections": len(detections),
        "Found": summary(detections),
        "Shown": view,
    }
    speed = getattr(result, "speed", None) or {}
    if speed:
        details["Time (ms)"] = ", ".join(f"{step} {milliseconds:.0f}" for step, milliseconds in speed.items())
    if view == VIEWS[2]:
        details["Colours"] = WHY_BLUE
    for number, detection in enumerate(detections[:LISTED], start=1):
        x1, y1, x2, y2 = (round(value) for value in detection.box)
        details[f"Detection {number}"] = (f"{detection.label} {detection.confidence:.2f}, "
                                          f"box ({x1}, {y1}) – ({x2}, {y2})")
    if len(detections) > LISTED:
        details["Not listed"] = f"{len(detections) - LISTED} more"
    return details


def ask_detection(parent: tk.Misc, image: Image.Image
                  ) -> tuple[YoloDetector, dict[str, float], str, Any, list[Detection], Image.Image] | None:
    """Let the user pick the model, the thresholds and the view while previewing on ``image``; ``None`` if cancelled.

    Returns the detector, the settings, the view, the Ultralytics result, its detections and the image to show.
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Object detection (YOLO)")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()
    controls = ttk.Frame(body)
    controls.grid(row=0, column=0, sticky="n")
    preview_side = ttk.Frame(body, padding=(20, 0, 0, 0))
    preview_side.grid(row=0, column=1, sticky="n")

    model_name = choice_row(controls, "Model", tuple(MODELS), lambda: schedule_preview())
    ttk.Label(controls, text="Thresholds", padding=(0, 16, 0, 0)).pack(anchor="w")
    values = setting_rows(controls, PARAMETERS, lambda: schedule_preview(), set())
    ttk.Label(controls, text="Boxes below the confidence are dropped; of two boxes of one class\n"
                             "overlapping by more than the IoU, the less confident is dropped.",
              foreground="#666666").pack(anchor="w", pady=(4, 0))

    def reset() -> None:
        for setting, (_, default, _, _) in PARAMETERS.items():
            values[setting].set(f"{default:g}")

    ttk.Button(controls, text="Reset to defaults", command=reset).pack(anchor="w", pady=(8, 0))
    ttk.Label(controls, text="Show", padding=(0, 16, 0, 0)).pack(anchor="w")
    view = tk.StringVar(dialog, value=VIEWS[0])
    for option in VIEWS:
        ttk.Radiobutton(controls, text=option, value=option, variable=view).pack(anchor="w", pady=(2, 0))
    view.trace_add("write", lambda *_: schedule_preview())

    ttk.Label(preview_side, text="Preview").pack(anchor="w")
    canvas = tk.Canvas(preview_side, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1],
                       background="#20242b", highlightthickness=0)
    canvas.pack(pady=(4, 2))
    status = ttk.Label(preview_side, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    preview_job: str | None = None
    # (detector, settings, view, result, detections, shown); None while the settings are invalid
    latest: tuple[YoloDetector, dict[str, float], str, Any, list[Detection], Image.Image] | None = None
    photo: ImageTk.PhotoImage | None = None

    def schedule_preview() -> None:
        nonlocal preview_job
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        status.configure(text="Updating preview…")
        preview_job = dialog.after(PREVIEW_DELAY_MS, update_preview)

    def update_preview() -> None:
        nonlocal preview_job, latest, photo
        preview_job = None
        latest = None
        canvas.delete("all")
        path = MODELS[model_name.get()]
        if path not in _loaded:
            status.configure(text=f"Loading {path.name} (downloaded on first use)…")
            dialog.update_idletasks()
        try:
            settings = read_settings(values, PARAMETERS, set())
            detector = load_detector(path)
            result, detections = detector.detect(image, **settings)
        except ValueError as error:
            status.configure(text=str(error))
            return
        shown = show_view(image, result, detections, view.get())
        latest = (detector, settings, view.get(), result, detections, shown)
        photo = ImageTk.PhotoImage(_fit(shown, PREVIEW_SIZE), master=dialog)  # Tk needs the reference
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        message = f"Found {summary(detections)}."
        status.configure(text=f"{message} {WHY_BLUE}" if view.get() == VIEWS[2] else message)

    result = []

    def close() -> None:
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        if preview_job is not None:  # settings changed after the last preview
            dialog.after_cancel(preview_job)
            update_preview()
        if latest is None:
            messagebox.showerror("Object detection", status.cget("text") or "There is nothing to keep yet.",
                                 parent=dialog)
            return
        result.append(latest)
        close()

    buttons = ttk.Frame(preview_side)
    buttons.pack(fill="x", pady=(16, 0))
    ttk.Button(buttons, text="Cancel", command=close).pack(side="right")
    ttk.Button(buttons, text="Apply", command=ok, default="active").pack(side="right", padx=(0, 6))
    dialog.bind("<Return>", ok)
    dialog.bind("<Escape>", lambda _event: close())
    dialog.protocol("WM_DELETE_WINDOW", close)

    center_over(dialog, parent.winfo_toplevel())
    dialog.deiconify()
    schedule_preview()
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class ObjectDetectionTool(ForensicsTool):
    tool_id = "object_detection"
    title = "Object detection (YOLO)"
    category = "Deep learning"
    description = "Detect objects with a pretrained Ultralytics YOLO11 model and draw their boxes."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.original is not None
        # The image as opened: after the first run the current image is the annotated figure.
        choice = ask_detection(parent, document.original)
        if choice is None:
            return None
        detector, settings, view, result, detections, shown = choice
        return ToolResult(
            image=shown,
            message=f"YOLO found {summary(detections)}.",
            details=detection_details(detector, settings, view, result, detections),
        )
