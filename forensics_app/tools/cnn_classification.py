"""CNN classification (Set 8.2): load a model trained in Colab (Set 8.1, ``notebooks/set8_1_train_cnn.ipynb``),
show its passport and classify the opened image with it.

The model is the ONNX file exported by the notebook, run with OpenCV's DNN module (as YuNet is), so the app needs no
TensorFlow. Next to it, ``<name>_passport.json`` gives the dataset, the class names, the preprocessing and the training
history; the app preprocesses the image exactly as the passport says.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .bow_classification import Prediction
from .dialogs import center_over
from .eigenfaces import _fit
from .hog import choice_row


MODEL_FOLDER = Path(__file__).resolve().parents[2] / "models"  # models/ is git-ignored
# The models the notebook trains, one button each in the dialog.
MODELS = {"Small CNN": MODEL_FOLDER / "cifar10_cnn.onnx", "InceptionV3": MODEL_FOLDER / "cifar10_inceptionv3.onnx"}
PASSPORT_SUFFIX = "_passport.json"
TOP = 5  # classes shown in the bar chart
PREVIEW_SIZE = (720, 560)
PASSPORT_WIDTH = 46  # characters
MISSING_MODEL = ("Train the model in Colab with notebooks/set8_1_train_cnn.ipynb, then put the .onnx file and its "
                 f"{PASSPORT_SUFFIX} file in the models folder.")

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # hide OpenCV 5's DNN backend notices


def passport_path(model: Path) -> Path:
    """``cifar10_cnn.onnx`` → ``cifar10_cnn_passport.json``."""
    return model.with_name(model.stem + PASSPORT_SUFFIX)


def load_passport(path: Path) -> dict[str, Any]:
    """The passport, with the keys the app needs checked; ``ValueError`` if it is missing or incomplete."""
    if not path.is_file():
        raise ValueError(f"No passport at {path}. {MISSING_MODEL}")
    passport = json.loads(path.read_text(encoding="utf-8"))
    preprocessing = passport.get("preprocessing", {})
    if not passport.get("classes") or len(preprocessing.get("input_size", ())) != 2:
        raise ValueError(f"{path.name} has no class names or no input size, so it is not a model passport.")
    return passport


def preprocess(image: Image.Image, passport: dict[str, Any]) -> np.ndarray:
    """The image as the model's input, following the passport: RGB, resized to its input size, then
    (pixel × scale − mean) / std per channel, as a (1, height, width, 3) batch (or (1, 3, height, width) for NCHW)."""
    steps = passport["preprocessing"]
    height, width = steps["input_size"]
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) * float(steps.get("scale", 1 / 255))
    pixels = (pixels - np.float32(steps.get("mean", 0.0))) / np.float32(steps.get("std", 1.0))
    if steps.get("layout", "NHWC") == "NCHW":
        pixels = pixels.transpose(2, 0, 1)
    return np.ascontiguousarray(pixels[None], dtype=np.float32)


@dataclass
class CnnModel:
    """An ONNX classifier and its passport."""

    path: Path
    passport: dict[str, Any]
    net: Any

    @classmethod
    def load(cls, path: Path) -> CnnModel:
        """``ValueError`` when the model or its passport is missing or does not fit together."""
        if not path.is_file():
            raise ValueError(f"No model at {path}. {MISSING_MODEL}")
        passport = load_passport(passport_path(path))
        try:
            net = cv2.dnn.readNetFromONNX(str(path))
        except cv2.error as error:
            raise ValueError(f"OpenCV could not read {path.name}: {error}") from error
        return cls(path, passport, net)

    @property
    def classes(self) -> list[str]:
        return [str(name) for name in self.passport["classes"]]

    def predict(self, image: Image.Image) -> Prediction:
        """The probability of every class for ``image``."""
        self.net.setInput(preprocess(image, self.passport))
        scores = np.asarray(self.net.forward(), dtype=np.float64).reshape(-1)
        if scores.size != len(self.classes):
            raise ValueError(f"The model gives {scores.size} scores but the passport names {len(self.classes)} "
                             "classes.")
        if scores.min() < 0 or not np.isclose(scores.sum(), 1, atol=1e-3):  # logits: apply the softmax
            scores = np.exp(scores - scores.max())
            scores /= scores.sum()
        return Prediction(self.classes, scores)


def _number(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def passport_lines(passport: dict[str, Any]) -> list[tuple[str, str]]:
    """The passport as (heading or field, value) rows; a heading has an empty value and starts with "#"."""
    dataset = passport.get("dataset", {})
    steps = passport["preprocessing"]
    training = passport.get("training", {})
    evaluation = passport.get("evaluation", {})
    architecture = passport.get("architecture", {})
    history = training.get("history", {})
    rows: list[tuple[str, str]] = [("# Model", "")]
    rows += [(key, str(value)) for key, value in (
        ("Name", passport.get("name")), ("Created", passport.get("created")),
        ("Framework", passport.get("framework")), ("Parameters", architecture.get("parameters")),
    ) if value is not None]
    rows += [("# Dataset", "")]
    rows += [(key, str(value)) for key, value in (
        ("Name", dataset.get("name")), ("Training images", dataset.get("train")),
        ("Validation images", dataset.get("validation")), ("Test images", dataset.get("test")),
    ) if value is not None]
    classes = passport["classes"]
    rows += [("# Classes", ""), (f"{len(classes)} classes", ", ".join(map(str, classes)))]
    rows += [("# Preprocessing", "")]
    if steps.get("steps"):
        rows += [(f"Step {number}", str(step)) for number, step in enumerate(steps["steps"], start=1)]
    else:
        height, width = steps["input_size"]
        rows += [("Input", f"{width} × {height} px RGB, {steps.get('layout', 'NHWC')}"),
                 ("Values", f"(pixel × {_number(steps.get('scale', 1 / 255))} − mean) / std")]
    rows += [("# Training", "")]
    rows += [(key, _number(value)) for key, value in (
        ("Optimizer", training.get("optimizer")), ("Learning rate", training.get("learning_rate")),
        ("Loss", training.get("loss")), ("Epochs", training.get("epochs")),
        ("Batch size", training.get("batch_size")), ("Augmentation", training.get("augmentation")),
    ) if value is not None]
    for key, label in (("accuracy", "Final training accuracy"), ("val_accuracy", "Final validation accuracy")):
        if history.get(key):
            rows.append((label, f"{history[key][-1]:.3f}"))
    if evaluation.get("test_accuracy") is not None:
        rows += [("# Evaluation", ""), ("Test accuracy", f"{evaluation['test_accuracy']:.3f}")]
        if evaluation.get("test_loss") is not None:
            rows.append(("Test loss", f"{evaluation['test_loss']:.3f}"))
    # Drop the headings of sections the passport leaves empty.
    return [row for index, row in enumerate(rows)
            if not row[0].startswith("#") or (index + 1 < len(rows) and not rows[index + 1][0].startswith("#"))]


def passport_text(passport: dict[str, Any]) -> str:
    """The passport as plain text, for the dialog."""
    lines = []
    for key, value in passport_lines(passport):
        if key.startswith("#"):
            lines.append(("\n" if lines else "") + key[2:].upper())
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _figure_image(figure: Figure) -> Image.Image:
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


def show_classification(image: Image.Image, prediction: Prediction, passport: dict[str, Any],
                        top: int = TOP) -> Image.Image:
    """The image under "p(cat) = 0.83" with the ``top`` most probable classes beside it and, below, the training
    curves from the passport (loss and accuracy per epoch, training and validation)."""
    history = passport.get("training", {}).get("history", {})
    curves = bool(history.get("loss"))
    figure = Figure(figsize=(9, 7.5 if curves else 4), layout="constrained")
    grid = figure.add_gridspec(2 if curves else 1, 2)
    picture = figure.add_subplot(grid[0, 0])
    picture.imshow(image.convert("RGB"))
    picture.set_title(f"p({prediction.label}) = {prediction.probability:.2f}", fontsize=15)
    picture.axis("off")
    bars = figure.add_subplot(grid[0, 1])
    ranked = prediction.ranked()[:top][::-1]  # most probable at the top
    colours = ["#d0d0d0"] * (len(ranked) - 1) + ["#1f77b4"]
    bars.barh([label for label, _ in ranked], [probability for _, probability in ranked], color=colours)
    for row, (_, probability) in enumerate(ranked):
        bars.text(probability + 0.01, row, f"{probability:.2f}", va="center")
    bars.set_xlim(0, 1.15)
    bars.set_xlabel("Probability")
    bars.set_title(f"Top {len(ranked)} of {len(prediction.classes)} classes")
    for side in ("top", "right"):
        bars.spines[side].set_visible(False)
    if curves:
        epochs = np.arange(1, len(history["loss"]) + 1)
        for column, (key, title) in enumerate((("loss", "Loss"), ("accuracy", "Accuracy"))):
            axes = figure.add_subplot(grid[1, column])
            for prefix, label, colour in (("", "training", "#1f77b4"), ("val_", "validation", "#ff7f0e")):
                if history.get(prefix + key):
                    axes.plot(epochs[:len(history[prefix + key])], history[prefix + key], color=colour, label=label)
            axes.set_title(f"Training history: {title.lower()}")
            axes.set_xlabel("Epoch")
            axes.set_ylabel(title)
            axes.legend()
            axes.grid(alpha=0.3)
    return _figure_image(figure)


def classification_details(model: CnnModel, prediction: Prediction) -> dict[str, object]:
    """The prediction and the passport, for the results panel."""
    details: dict[str, object] = {
        "Operation": "CNN classification",
        "Model": str(model.path),
        "Predicted": prediction.label,
        "Top probability": f"{prediction.probability:.2f}",
        **{f"p({label})": f"{probability:.2f}" for label, probability in prediction.ranked()[:TOP]},
    }
    section = ""
    for key, value in passport_lines(model.passport):
        if key.startswith("#"):
            section = key[2:]
        else:
            details[f"{section}: {key}"] = value
    return details


def ask_model(parent: tk.Misc, image: Image.Image) -> tuple[CnnModel, Prediction] | None:
    """Let the user pick a model, read its passport and see the image classified; ``None`` if cancelled."""
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("CNN classification")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()

    model_row = ttk.Frame(body)
    model_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
    model_name = choice_row(model_row, "Model", tuple(MODELS), lambda: load())

    passport_side = ttk.Frame(body)
    passport_side.grid(row=2, column=0, sticky="ns")
    ttk.Label(passport_side, text="Model passport").pack(anchor="w")
    passport_box = tk.Text(passport_side, width=PASSPORT_WIDTH, height=34, wrap="word", relief="flat",
                           background="#f4f4f4", foreground="#1e1e1e", font=("TkDefaultFont", 11), padx=8, pady=6)
    passport_box.pack(fill="y", expand=True, pady=(4, 0))
    preview_side = ttk.Frame(body, padding=(16, 0, 0, 0))
    preview_side.grid(row=2, column=1, sticky="n")
    ttk.Label(preview_side, text="Classification").pack(anchor="w")
    canvas = tk.Canvas(preview_side, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1], background="#20242b",
                       highlightthickness=0)
    canvas.pack(pady=(4, 2))
    status = ttk.Label(preview_side, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    latest: tuple[CnnModel, Prediction] | None = None
    photo: ImageTk.PhotoImage | None = None

    def show_passport(text: str) -> None:
        passport_box.configure(state="normal")
        passport_box.delete("1.0", "end")
        passport_box.insert("1.0", text)
        passport_box.configure(state="disabled")

    def load(*_: object) -> None:
        nonlocal latest, photo
        latest = None
        canvas.delete("all")
        try:
            model = CnnModel.load(MODELS[model_name.get()])
            show_passport(passport_text(model.passport))
            prediction = model.predict(image)
        except (ValueError, OSError, cv2.error, json.JSONDecodeError) as error:
            show_passport("")
            status.configure(text=str(error))
            return
        latest = (model, prediction)
        photo = ImageTk.PhotoImage(_fit(show_classification(image, prediction, model.passport), PREVIEW_SIZE),
                                   master=dialog)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        height, width = model.passport["preprocessing"]["input_size"]
        status.configure(text=f"p({prediction.label}) = {prediction.probability:.2f}. The image was resized to "
                              f"{width} × {height} px as the passport says; Apply keeps this result.")

    result = []

    def ok(_event: tk.Event | None = None) -> None:
        if latest is None:
            messagebox.showerror("CNN classification", status.cget("text") or "There is no model loaded.",
                                 parent=dialog)
            return
        result.append(latest)
        dialog.destroy()

    buttons = ttk.Frame(body)
    buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(16, 0))
    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Apply", command=ok, default="active").pack(side="right", padx=(0, 6))
    dialog.bind("<Return>", ok)
    dialog.bind("<Escape>", lambda _event: dialog.destroy())

    center_over(dialog, parent.winfo_toplevel())
    dialog.deiconify()
    load()
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class CnnClassificationTool(ForensicsTool):
    tool_id = "cnn_classification"
    title = "CNN classification"
    category = "Deep learning"
    description = "Load a CNN trained in Colab, show its passport and classify the opened image."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.original is not None
        # The image as opened: after the first run the current image is the result panel.
        choice = ask_model(parent, document.original)
        if choice is None:
            return None
        model, prediction = choice
        return ToolResult(
            image=show_classification(document.original, prediction, model.passport),
            message=f"p({prediction.label}) = {prediction.probability:.2f} with {model.path.name}.",
            details=classification_details(model, prediction),
        )
