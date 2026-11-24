"""CLIP zero-shot classification (Set 10.1): classify the opened image into classes the user writes.

CLIP has an image encoder and a text encoder trained so that a picture and its caption land close together in the same
embedding space. Each class is written into a sentence with a template ("a photo of a {}." → "a photo of a cat."), the
image and every sentence are encoded, and the softmax of the scaled cosine similarities gives the class probabilities,
as in the CLIP playground notebook. Nothing is trained: change the text and the classifier changes.

Hugging Face ``transformers`` downloads the weights the first time (about 600 MB for ViT-B/32, 1.7 GB for ViT-L/14)
into its own cache. It and PyTorch are imported only when the tool runs, so the rest of the app starts without them.
"""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .bow_classification import Prediction
from .cnn_classification import _figure_image
from .dialogs import center_over
from .eigenfaces import _fit
from .hog import choice_row


# One button each in the dialog; the larger model is slower but more accurate (exercise 2 of the notebook).
MODELS = {"ViT-B/32": "openai/clip-vit-base-patch32", "ViT-L/14": "openai/clip-vit-large-patch14"}
TEMPLATE = "a photo of a {}."  # "{}" is replaced by each class
CLASSES = ("cat", "dog", "person", "coffee", "car", "building", "landscape", "something else")  # edit freely
BAR_COLOUR = "#3b6ea8"  # the notebook's colour
PREVIEW_SIZE = (720, 520)
PREVIEW_DELAY_MS = 500  # wait for a pause in typing before classifying again
MISSING_TRANSFORMERS = "Install Hugging Face Transformers first: python -m pip install transformers"


def read_classes(text: str) -> list[str]:
    """One class per non-empty line; ``ValueError`` unless there are at least two different ones."""
    classes = [line.strip() for line in text.splitlines() if line.strip()]
    if len(classes) < 2:
        raise ValueError("Write at least two classes, one per line: the probabilities are shared among them.")
    repeated = sorted({name for name in classes if classes.count(name) > 1})
    if repeated:
        raise ValueError(f"Each class must be different; repeated: {', '.join(repeated)}.")
    return classes


def prompts(classes: list[str], template: str) -> list[str]:
    """Each class written into the template: "a photo of a {}." → "a photo of a cat."."""
    if "{}" not in template:
        raise ValueError('The template needs "{}" where the class goes, e.g. "a photo of a {}."')
    return [template.replace("{}", name) for name in classes]


@dataclass
class ClipModel:
    """A CLIP model and its processor (resizing, cropping and normalising images; tokenising text)."""

    name: str
    model: Any
    processor: Any
    device: str

    @classmethod
    def load(cls, name: str) -> ClipModel:
        """``ValueError`` when Transformers is missing or cannot load (or download) the model."""
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise ValueError(MISSING_TRANSFORMERS) from error
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = CLIPModel.from_pretrained(name).to(device).eval()
            processor = CLIPProcessor.from_pretrained(name)
        except Exception as error:  # no connection, unknown name, broken cache...
            raise ValueError(f"Transformers could not load {name}: {error}") from error
        return cls(name, model, processor, device)

    def similarities(self, image: Image.Image, texts: list[str]) -> tuple[np.ndarray, float]:
        """The cosine similarity between the image and each text, and CLIP's learned scale for them."""
        import torch
        import torch.nn.functional as F

        def embedding(output: Any) -> Any:  # newer Transformers return an output object instead of a tensor
            return F.normalize(output if torch.is_tensor(output) else output.pooler_output, dim=-1)

        with torch.no_grad():
            pixels = self.processor(images=[image.convert("RGB")], return_tensors="pt").to(self.device)
            words = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            image_embedding = embedding(self.model.get_image_features(**pixels))  # (1, D)
            text_embeddings = embedding(self.model.get_text_features(**words))  # (classes, D)
            cosines = (image_embedding @ text_embeddings.T)[0]
            return cosines.cpu().numpy().astype(np.float64), float(self.model.logit_scale.exp())

    def classify(self, image: Image.Image, classes: list[str], template: str) -> tuple[Prediction, np.ndarray]:
        """The probability of every class (softmax of the scaled similarities) and the raw similarities."""
        cosines, scale = self.similarities(image, prompts(classes, template))
        return Prediction(classes, softmax(scale * cosines)), cosines


def softmax(logits: np.ndarray) -> np.ndarray:
    exponentials = np.exp(logits - logits.max())
    return exponentials / exponentials.sum()


_loaded: dict[str, ClipModel] = {}  # loading imports PyTorch and reads the weights, so it is done once


def load_model(name: str) -> ClipModel:
    if name not in _loaded:
        _loaded[name] = ClipModel.load(name)
    return _loaded[name]


def show_zero_shot(image: Image.Image, prediction: Prediction, template: str) -> Image.Image:
    """The image beside a bar per class with its probability, most probable at the top, as in the notebook."""
    ranked = prediction.ranked()[::-1]  # barh draws from the bottom
    figure = Figure(figsize=(10, max(3.2, 0.45 * len(ranked) + 1)), layout="constrained")
    picture, bars = figure.subplots(1, 2, width_ratios=[1, 1.4])
    picture.imshow(image.convert("RGB"))
    picture.set_title(f"{prediction.label} ({prediction.probability:.2f})")
    picture.axis("off")
    bars.barh([label for label, _ in ranked], [probability for _, probability in ranked], color=BAR_COLOUR)
    for row, (_, probability) in enumerate(ranked):
        bars.text(probability + 0.01, row, f"{probability:.2f}", va="center", fontsize=8)
    bars.set_xlim(0, 1)
    bars.set_xlabel("probability")
    bars.set_title(f'template: "{template}"')
    return _figure_image(figure)


def zero_shot_details(model: ClipModel, template: str, prediction: Prediction,
                      cosines: np.ndarray) -> dict[str, object]:
    """The settings and every class with its probability and similarity, for the results panel."""
    similarity = dict(zip(prediction.classes, cosines))
    details: dict[str, object] = {
        "Operation": "CLIP zero-shot classification",
        "Model": model.name,
        "Device": model.device,
        "Template": template,
        "Classes": len(prediction.classes),
        "Predicted": prediction.label,
        "Top probability": f"{prediction.probability:.2f}",
    }
    for label, probability in prediction.ranked():
        details[f"p({label})"] = f"{probability:.3f} (cosine similarity {similarity[label]:.3f})"
    return details


Choice = tuple[ClipModel, str, Prediction, np.ndarray, Image.Image]


def ask_zero_shot(parent: tk.Misc, image: Image.Image) -> Choice | None:
    """Let the user pick the model and write the template and classes while previewing on ``image``; ``None`` if
    cancelled. Returns the model, the template, the prediction, the similarities and the figure."""
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("CLIP zero-shot classification")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()
    controls = ttk.Frame(body)
    controls.grid(row=0, column=0, sticky="n")
    preview_side = ttk.Frame(body, padding=(20, 0, 0, 0))
    preview_side.grid(row=0, column=1, sticky="n")

    model_name = choice_row(controls, "Model", tuple(MODELS), lambda: schedule_preview())
    ttk.Label(controls, text="Template", padding=(0, 16, 0, 0)).pack(anchor="w")
    template = tk.StringVar(dialog, value=TEMPLATE)
    ttk.Entry(controls, textvariable=template, width=36).pack(anchor="w", fill="x", pady=(4, 0))
    template.trace_add("write", lambda *_: schedule_preview())
    ttk.Label(controls, text='"{}" is replaced by each class.', foreground="#666666").pack(anchor="w", pady=(4, 0))
    ttk.Label(controls, text="Classes (one per line)", padding=(0, 16, 0, 0)).pack(anchor="w")
    classes_box = tk.Text(controls, width=36, height=14, wrap="none", undo=True)
    classes_box.insert("1.0", "\n".join(CLASSES))
    classes_box.pack(anchor="w", pady=(4, 0))
    classes_box.edit_modified(False)

    def classes_changed(_event: tk.Event) -> None:
        if classes_box.edit_modified():
            classes_box.edit_modified(False)
            schedule_preview()

    classes_box.bind("<<Modified>>", classes_changed)
    ttk.Label(controls, text="Any number of classes and any text: CLIP compares\nthe image with each sentence. "
                             "The probabilities\nalways sum to 1, so add an \"other\" class if none fits.",
              foreground="#666666").pack(anchor="w", pady=(4, 0))

    ttk.Label(preview_side, text="Preview").pack(anchor="w")
    canvas = tk.Canvas(preview_side, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1],
                       background="#20242b", highlightthickness=0)
    canvas.pack(pady=(4, 2))
    status = ttk.Label(preview_side, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    preview_job: str | None = None
    latest: Choice | None = None  # None while the settings are invalid
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
        name = MODELS[model_name.get()]
        if name not in _loaded:
            status.configure(text=f"Loading {name} (downloaded on first use, this can take a while)…")
            dialog.update_idletasks()
        try:
            classes = read_classes(classes_box.get("1.0", "end"))
            prompts(classes, template.get())  # check the template before loading the model
            model = load_model(name)
            prediction, cosines = model.classify(image, classes, template.get())
            shown = show_zero_shot(image, prediction, template.get())
        except ValueError as error:
            status.configure(text=str(error))
            return
        latest = (model, template.get(), prediction, cosines, shown)
        photo = ImageTk.PhotoImage(_fit(shown, PREVIEW_SIZE), master=dialog)  # Tk needs the reference
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        status.configure(text=f"{prediction.label}: p = {prediction.probability:.2f} among {len(classes)} classes "
                              f"({model.device}).")

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
            messagebox.showerror("CLIP zero-shot classification",
                                 status.cget("text") or "There is nothing to keep yet.", parent=dialog)
            return
        result.append(latest)
        close()

    buttons = ttk.Frame(preview_side)
    buttons.pack(fill="x", pady=(16, 0))
    ttk.Button(buttons, text="Cancel", command=close).pack(side="right")
    ttk.Button(buttons, text="Apply", command=ok, default="active").pack(side="right", padx=(0, 6))
    dialog.bind("<Escape>", lambda _event: close())  # no <Return>: it adds a line to the classes
    dialog.protocol("WM_DELETE_WINDOW", close)

    center_over(dialog, parent.winfo_toplevel())
    dialog.deiconify()
    schedule_preview()
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class ClipZeroShotTool(ForensicsTool):
    tool_id = "clip_zero_shot"
    title = "CLIP zero-shot classification"
    category = "Deep learning"
    description = "Classify the opened image into any classes you write, with CLIP and no training."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.original is not None
        # The image as opened: after the first run the current image is the figure.
        choice = ask_zero_shot(parent, document.original)
        if choice is None:
            return None
        model, template, prediction, cosines, shown = choice
        return ToolResult(
            image=shown,
            message=f"CLIP: {prediction.label} (p = {prediction.probability:.2f}).",
            details=zero_shot_details(model, template, prediction, cosines),
        )
