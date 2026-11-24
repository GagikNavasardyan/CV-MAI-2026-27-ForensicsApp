"""U-Net segmentation (Set 11.1): load the U-Net trained in ``notebooks/set11_1_train_unet.ipynb`` and show where the
food is on a meal tray image.

The notebook trains on the TrayDataset a binary task, food vs. not food (mask classes from 5 on are food), and exports
the network to ONNX, which the app runs with OpenCV's DNN module as the CNN classification tool does, so it needs no
PyTorch. ``tray_food_unet_passport.json`` gives the input size and preprocessing. The network gives one logit per pixel
at 128 × 208; the sigmoid turns it into a food probability, which is resized to the image and thresholded.

Opened on a dataset image (``.../XTest/name.jpg``), the tool also reads its mask (``.../yTest/name.png``) and shows the
ground truth and the Dice coefficient of the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import cv2
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .cnn_classification import _figure_image, load_passport, passport_lines, passport_path, preprocess
from .corner_detection import read_settings, setting_rows
from .dialogs import center_over
from .eigenfaces import _fit


MODEL = Path(__file__).resolve().parents[2] / "models" / "tray_food_unet.onnx"  # models/ is git-ignored
# Number setting: (label, default, minimum, maximum).
PARAMETERS: dict[str, tuple[str, float, float, float]] = {
    "threshold": ("Threshold", 0.5, 0.05, 0.95),  # pixels with a higher food probability are food
}
FIRST_FOOD = 5  # mask class of 'meatball'; used when the passport does not say
OVERLAY_ALPHA = 0.45
FIGURE_WIDTH = 15  # inches
PREVIEW_SIZE = (1000, 300)
PREVIEW_DELAY_MS = 150
MISSING_MODEL = ("Train the model with notebooks/set11_1_train_unet.ipynb; it saves tray_food_unet.onnx and its "
                 "passport in the models folder.")


@dataclass
class UnetModel:
    """The ONNX U-Net and its passport."""

    path: Path
    passport: dict[str, Any]
    net: Any

    @classmethod
    def load(cls, path: Path = MODEL) -> UnetModel:
        """``ValueError`` when the model or its passport is missing or unreadable."""
        if not path.is_file():
            raise ValueError(f"No model at {path}. {MISSING_MODEL}")
        passport = load_passport(passport_path(path))
        try:
            net = cv2.dnn.readNetFromONNX(str(path))
        except cv2.error as error:
            raise ValueError(f"OpenCV could not read {path.name}: {error}") from error
        return cls(path, passport, net)

    @property
    def first_food(self) -> int:
        return int(self.passport.get("dataset", {}).get("food_from_class", FIRST_FOOD))

    def probabilities(self, image: Image.Image) -> np.ndarray:
        """The food probability of every pixel of ``image``, as a (height, width) array in [0, 1]."""
        self.net.setInput(preprocess(image, self.passport))
        logits = np.asarray(self.net.forward(), dtype=np.float32)
        logits = logits.reshape(logits.shape[-2:])  # (1, 1, h, w) → (h, w)
        probability = 1 / (1 + np.exp(-logits))  # sigmoid
        return cv2.resize(probability, image.size, interpolation=cv2.INTER_LINEAR)


def mask_path_for(image_path: Path) -> Path | None:
    """Where TrayDataset keeps the mask of an image: ``.../XTest/name.jpg`` → ``.../yTest/name.png``."""
    folder = image_path.parent
    if not folder.name.startswith("X"):
        return None
    return folder.parent / f"y{folder.name[1:]}" / f"{image_path.stem}.png"


def read_ground_truth(path: Path | None, size: tuple[int, int], first_food: int = FIRST_FOOD) -> np.ndarray | None:
    """The food pixels of a TrayDataset mask (class index ≥ ``first_food``), at ``size``; ``None`` without a mask."""
    if path is None or not path.is_file():
        return None
    classes = np.asarray(Image.open(path))
    if classes.ndim == 3:  # some masks are saved with 3 identical channels
        classes = classes[..., 0]
    food = Image.fromarray(((classes >= first_food) * 255).astype(np.uint8))
    return np.asarray(food.resize(size, Image.Resampling.NEAREST)) > 0  # NEAREST: no new values at the borders


def dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    """2 |P ∩ G| / (|P| + |G|): 1 when the masks match, 0 when they do not overlap (1 when both are empty)."""
    total = prediction.sum() + truth.sum()
    return 1.0 if total == 0 else float(2 * np.logical_and(prediction, truth).sum() / total)


def show_segmentation(image: Image.Image, mask: np.ndarray, truth: np.ndarray | None = None) -> Image.Image:
    """Original image, segmentation mask and the mask overlaid on the image, as in the slides; with the ground truth
    (and the Dice in the mask's title) when there is one."""
    picture = image.convert("RGB")
    panels = 4 if truth is not None else 3
    height = FIGURE_WIDTH / panels * picture.height / picture.width + 0.8
    figure = Figure(figsize=(FIGURE_WIDTH, height), layout="constrained")
    axes = figure.subplots(1, panels)
    axes[0].imshow(picture)
    axes[0].set_title("Original Image")
    column = 1
    if truth is not None:
        axes[1].imshow(truth, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        axes[1].set_title("Ground Truth (food)")
        column = 2
    axes[column].imshow(mask, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    title = "Segmentation Mask"
    if truth is not None:
        title += f" (Dice = {dice(mask, truth):.3f})"
    axes[column].set_title(title)
    axes[column + 1].imshow(picture)
    axes[column + 1].imshow(mask, cmap="viridis", vmin=0, vmax=1, alpha=OVERLAY_ALPHA, interpolation="nearest")
    axes[column + 1].set_title("Overlayed Mask")
    return _figure_image(figure)


def segmentation_details(model: UnetModel, threshold: float, mask: np.ndarray,
                         truth: np.ndarray | None) -> dict[str, object]:
    """The result and the passport, for the results panel."""
    details: dict[str, object] = {
        "Operation": "U-Net segmentation (food vs. not food)",
        "Model": str(model.path),
        "Threshold": f"{threshold:g}",
        "Food pixels": f"{mask.mean():.1%}",
    }
    if truth is not None:
        details["Ground truth food pixels"] = f"{truth.mean():.1%}"
        details["Dice"] = f"{dice(mask, truth):.3f}"
    section = ""
    for key, value in passport_lines(model.passport):
        if key.startswith("#"):
            section = key[2:]
        else:
            details[f"{section}: {key}"] = value
    evaluation = model.passport.get("evaluation", {})
    if evaluation.get("test_dice") is not None:
        details["Evaluation: Test Dice"] = f"{evaluation['test_dice']:.3f} ± {evaluation.get('test_dice_std', 0):.3f}"
    return details


Choice = tuple[UnetModel, float, np.ndarray, np.ndarray | None, Image.Image]


def ask_segmentation(parent: tk.Misc, image: Image.Image, mask_path: Path | None = None) -> Choice | None:
    """Segment ``image`` and let the user tune the threshold while previewing; ``None`` if cancelled.

    Returns the model, the threshold, the food mask, the ground truth (``None`` without a mask file) and the figure.
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("U-Net segmentation")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()

    values = setting_rows(body, PARAMETERS, lambda: schedule_preview(), set())
    ttk.Label(body, text="Each pixel gets a food probability; above the threshold it is food. "
                         "Lower finds more food, higher only the surest.",
              foreground="#666666").pack(anchor="w", pady=(4, 0))
    canvas = tk.Canvas(body, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1], background="#20242b",
                       highlightthickness=0)
    canvas.pack(pady=(12, 2))
    status = ttk.Label(body, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    model: UnetModel | None = None
    probability: np.ndarray | None = None
    truth: np.ndarray | None = None
    try:
        model = UnetModel.load()
        probability = model.probabilities(image)
        truth = read_ground_truth(mask_path, image.size, model.first_food)
    except (ValueError, OSError, cv2.error, json.JSONDecodeError) as error:
        status.configure(text=str(error))

    preview_job: str | None = None
    latest: Choice | None = None
    photo: ImageTk.PhotoImage | None = None

    def schedule_preview() -> None:
        nonlocal preview_job
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        preview_job = dialog.after(PREVIEW_DELAY_MS, update_preview)

    def update_preview() -> None:
        nonlocal preview_job, latest, photo
        preview_job = None
        if model is None or probability is None:
            return
        latest = None
        canvas.delete("all")
        try:
            threshold = read_settings(values, PARAMETERS, set())["threshold"]
        except ValueError as error:
            status.configure(text=str(error))
            return
        mask = probability > threshold
        shown = show_segmentation(image, mask, truth)
        latest = (model, threshold, mask, truth, shown)
        photo = ImageTk.PhotoImage(_fit(shown, PREVIEW_SIZE), master=dialog)  # Tk needs the reference
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        message = f"Food: {mask.mean():.1%} of the pixels."
        if truth is not None:
            message += f" Ground truth: {truth.mean():.1%}; Dice = {dice(mask, truth):.3f}."
        else:
            message += " No mask next to this image (…/yTest/<name>.png), so no ground truth or Dice."
        status.configure(text=message)

    result = []

    def close() -> None:
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        if preview_job is not None:  # threshold changed after the last preview
            dialog.after_cancel(preview_job)
            update_preview()
        if latest is None:
            messagebox.showerror("U-Net segmentation", status.cget("text") or "There is nothing to keep yet.",
                                 parent=dialog)
            return
        result.append(latest)
        close()

    buttons = ttk.Frame(body)
    buttons.pack(fill="x", pady=(16, 0))
    ttk.Button(buttons, text="Cancel", command=close).pack(side="right")
    ttk.Button(buttons, text="Apply", command=ok, default="active").pack(side="right", padx=(0, 6))
    dialog.bind("<Return>", ok)
    dialog.bind("<Escape>", lambda _event: close())
    dialog.protocol("WM_DELETE_WINDOW", close)

    center_over(dialog, parent.winfo_toplevel())
    dialog.deiconify()
    update_preview()
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class UnetSegmentationTool(ForensicsTool):
    tool_id = "unet_segmentation"
    title = "U-Net segmentation"
    category = "Deep learning"
    description = "Find the food on a meal tray image with the U-Net trained on TrayDataset."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.original is not None
        # The image as opened: after the first run the current image is the figure.
        mask_path = mask_path_for(document.path) if document.path is not None else None
        choice = ask_segmentation(parent, document.original, mask_path)
        if choice is None:
            return None
        model, threshold, mask, truth, shown = choice
        message = f"U-Net: {mask.mean():.1%} of the pixels are food"
        message += f", Dice = {dice(mask, truth):.3f}." if truth is not None else "."
        return ToolResult(image=shown, message=message, details=segmentation_details(model, threshold, mask, truth))
