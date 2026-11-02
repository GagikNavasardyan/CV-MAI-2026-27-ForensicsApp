"""Face recognition: the training faces closest to the opened image in the eigenface space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import center_over
from .eigenfaces import DEFAULT_DATASET, MODEL_FOLDER, TILE_HEIGHT, EigenfaceModel, _fit, load_model
from .hog import number_row, read_number


# Number setting: (label, default, minimum, maximum).
NEIGHBOURS = ("Closest faces", 3, 1, 5)  # the slide asks for 2–3
GAP = 16
PREVIEW_SIZE = (640, 240)
PREVIEW_DELAY_MS = 200


@dataclass(frozen=True)
class Neighbour:
    """A training face and how far it is from the test face."""

    name: str
    path: str  # relative to the dataset folder
    distance: float


def saved_model(folder: Path) -> EigenfaceModel:
    """The model in ``folder``; raises ``ValueError`` when Eigenfaces has not built one there."""
    if not (folder / MODEL_FOLDER / "model.json").is_file():
        raise ValueError(f"No saved model in {folder}. Build it first with Eigenfaces (PCA).")
    return load_model(folder)


def face_pixels(image: Image.Image, model: EigenfaceModel) -> np.ndarray:
    """``image`` as the model sees it: greyscale and at the size of the training faces."""
    height, width = model.mean.shape
    grey = image.convert("L")
    if grey.size != (width, height):
        grey = grey.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(grey, dtype=np.float64)


def closest_faces(model: EigenfaceModel, pixels: np.ndarray, count: int = 3) -> list[Neighbour]:
    """The ``count`` smallest d_i = ‖w_test − w_i‖₂, nearest first; the first one is i* = argmin d_i."""
    distances = np.linalg.norm(model.coefficients - model.project(pixels), axis=1)
    return [Neighbour(model.names[index], model.paths[index], float(distances[index]))
            for index in np.argsort(distances, kind="stable")[:count]]


def true_name(path: Path | None, folder: Path) -> str | None:
    """The person in an image from ``folder/test/<person>/``, named as ``load_faces`` does; else ``None``."""
    if path is None or path.parent.parent.resolve() != (folder / "test").resolve():
        return None
    return path.parent.name.replace("_", " ")


def show_recognition(pixels: np.ndarray, neighbours: list[Neighbour], folder: Path, truth: str | None) -> Image.Image:
    """The test face and its closest training faces in a row, each titled above, as on the slide."""
    height, width = pixels.shape
    tile = (max(1, round(width * TILE_HEIGHT / height)), TILE_HEIGHT)
    faces = [Image.fromarray(pixels.astype(np.uint8))]
    titles = [(f"predicted: {neighbours[0].name}", f"true: {truth}" if truth else "test image")]
    for place, neighbour in enumerate(neighbours, 1):
        with Image.open(folder / neighbour.path) as image_file:
            faces.append(image_file.convert("L").copy())
        titles.append((f"{place}. {neighbour.name}", f"d = {neighbour.distance:.1f}"))
    font = ImageFont.load_default(14)
    line = round(font.size * 1.4)
    column = max(tile[0], *(round(font.getlength(text)) for title in titles for text in title))
    result = Image.new("RGB", (len(faces) * (column + GAP) + GAP, 2 * line + tile[1] + 2 * GAP), (255, 255, 255))
    draw = ImageDraw.Draw(result)
    for index, (face, title) in enumerate(zip(faces, titles)):
        left = GAP + index * (column + GAP)
        center = left + column / 2
        for row, text in enumerate(title):
            draw.text((center, GAP + line * (row + 0.5)), text, fill=(0, 0, 0), font=font, anchor="mm")
        corner = (round(center - tile[0] / 2), GAP + 2 * line)
        result.paste(face.resize(tile, Image.Resampling.LANCZOS), corner)
        draw.rectangle((corner[0] - 1, corner[1] - 1, corner[0] + tile[0], corner[1] + tile[1]), outline=(0, 0, 0))
    return result


def ask_recognition(parent: tk.Misc, image: Image.Image, path: Path | None) -> tuple[Path, EigenfaceModel, int] | None:
    """Let the user pick the dataset and how many faces to show while previewing them; ``None`` if cancelled."""
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Face recognition")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()

    ttk.Label(body, text="Dataset folder (with the model saved by Eigenfaces)").pack(anchor="w")
    folder_row = ttk.Frame(body)
    folder_row.pack(fill="x", pady=(4, 0))
    folder = tk.StringVar(dialog, value=str(DEFAULT_DATASET))
    ttk.Entry(folder_row, textvariable=folder).pack(side="left", fill="x", expand=True)
    ttk.Button(folder_row, text="Browse…", command=lambda: browse()).pack(side="left", padx=(6, 0))
    count = number_row(body, *NEIGHBOURS, lambda: schedule_preview())
    ttk.Label(body, text="Nearest training faces by d_i = ‖w_test − w_i‖₂, with w = U_K^T (x − μ).",
              foreground="#666666").pack(anchor="w", pady=(6, 0))

    canvas = tk.Canvas(body, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1], background="#20242b", highlightthickness=0)
    canvas.pack(pady=(12, 2))
    status = ttk.Label(body, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    preview_job: str | None = None
    model_cache: tuple[Path, EigenfaceModel] | None = None
    photo: ImageTk.PhotoImage | None = None

    def compute() -> tuple[Path, EigenfaceModel, int]:
        nonlocal model_cache
        chosen = Path(folder.get()).expanduser()
        if model_cache is None or model_cache[0] != chosen:
            model_cache = (chosen, saved_model(chosen))
        return chosen, model_cache[1], read_number(count, NEIGHBOURS[0], NEIGHBOURS[2], NEIGHBOURS[3])

    def schedule_preview() -> None:
        nonlocal preview_job
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        status.configure(text="Updating preview…")
        preview_job = dialog.after(PREVIEW_DELAY_MS, update_preview)

    def update_preview() -> None:
        nonlocal preview_job, photo
        preview_job = None
        canvas.delete("all")
        try:
            chosen, model, shown = compute()
            pixels = face_pixels(image, model)
            neighbours = closest_faces(model, pixels, shown)
            panel = show_recognition(pixels, neighbours, chosen, true_name(path, chosen))
        except (ValueError, OSError) as error:
            status.configure(text=str(error))
            return
        photo = ImageTk.PhotoImage(_fit(panel, PREVIEW_SIZE), master=dialog)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        height, width = model.mean.shape
        note = "" if image.size == (width, height) else (
            f" This image was resized from {image.width} × {image.height} px, so the result only means "
            "something for a tightly cropped face.")
        status.configure(text=f"Predicted: {neighbours[0].name}. K = {model.components}, "
                              f"{len(model.names)} training faces.{note}")

    def browse() -> None:
        chosen = filedialog.askdirectory(title="Choose the dataset folder", parent=dialog,
                                         initialdir=folder.get() if Path(folder.get()).is_dir() else None)
        if chosen:
            folder.set(chosen)

    folder.trace_add("write", lambda *_: schedule_preview())
    result = []

    def close() -> None:
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        try:
            result.append(compute())
        except (ValueError, OSError) as error:
            messagebox.showerror("Face recognition", str(error), parent=dialog)
            return
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
    schedule_preview()
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class FaceRecognitionTool(ForensicsTool):
    tool_id = "face_recognition"
    title = "Face recognition"
    category = "Face recognition"
    description = "Find the training faces closest to the opened image using the saved eigenface model."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.original is not None
        # The image as opened: after the first run the current image is the result panel, not a face.
        choice = ask_recognition(parent, document.original, document.path)
        if choice is None:
            return None
        folder, model, count = choice
        pixels = face_pixels(document.original, model)
        neighbours = closest_faces(model, pixels, count)
        truth = true_name(document.path, folder)
        height, width = model.mean.shape
        return ToolResult(
            image=show_recognition(pixels, neighbours, folder, truth),
            message=f"Predicted: {neighbours[0].name}.",
            details={
                "Operation": "Face recognition",
                "Dataset": str(folder),
                "Training faces": len(model.names),
                "Components (K)": model.components,
                "Face size": f"{width} × {height} px",
                "Predicted": neighbours[0].name,
                "True name": truth or "Unknown (not an image of test/<person>/)",
                **{f"{place}. {neighbour.name}": f"d = {neighbour.distance:.1f} ({neighbour.path})"
                   for place, neighbour in enumerate(neighbours, 1)},
            },
        )
