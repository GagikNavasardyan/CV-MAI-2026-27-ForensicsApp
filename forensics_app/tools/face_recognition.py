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
from .eigenfaces import DEFAULT_DATASET, MODEL_FOLDER, TILE_HEIGHT, EigenfaceModel, FaceSet, _fit, load_faces, load_model
from .hog import choice_row, number_row, read_number


# Number setting: (label, default, minimum, maximum).
NEIGHBOURS = ("Closest faces", 3, 1, 5)  # the slide asks for 2–3
L2, WHITENED_COSINE = METRICS = ("L2", "Whitened cosine")
CLOSEST, ACCURACY = VIEWS = ("Closest faces", "Test set accuracy")
GAP = 16
CELL = 72  # side of a confusion matrix cell
PREVIEW_SIZE = (640, 420)  # tall enough to read the confusion matrix
PREVIEW_DELAY_MS = 200


@dataclass(frozen=True)
class Neighbour:
    """A training face and how far it is from the test face."""

    name: str
    path: str  # relative to the dataset folder
    distance: float


@dataclass(frozen=True)
class Evaluation:
    """How the test faces were classified by their closest training face."""

    people: list[str]
    confusion: np.ndarray  # (p, p) row: the true person, column: the predicted one

    @property
    def total(self) -> int:
        return int(self.confusion.sum())

    @property
    def accuracy(self) -> float:
        return np.trace(self.confusion) / self.total

    @property
    def most_common(self) -> tuple[str, float]:
        """The person with the most test faces and the accuracy of always answering them."""
        counts = self.confusion.sum(axis=1)
        return self.people[int(counts.argmax())], counts.max() / self.total


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


def face_distances(model: EigenfaceModel, pixels: np.ndarray, metric: str = L2) -> np.ndarray:
    """d_i from the test face to every training face.

    L2 is d_i = ‖w_test − w_i‖₂. The first eigenfaces have the largest coefficients and so dominate it, although
    they tend to follow the lighting; whitening gives every eigenface the same variance, and the cosine distance
    then compares only the direction of w. The saved variance ratios differ from the variances by a constant,
    which the cosine ignores.
    """
    weights = model.project(pixels)
    if metric == L2:
        return np.linalg.norm(model.coefficients - weights, axis=1)
    scale = np.sqrt(model.explained_variance_ratio)
    scale[scale == 0] = 1  # identical training faces can leave a direction with no variance
    train, test = model.coefficients / scale, weights / scale
    norms = np.linalg.norm(train, axis=1) * np.linalg.norm(test)
    return 1 - (train @ test) / np.maximum(norms, np.finfo(float).tiny)


def closest_faces(model: EigenfaceModel, pixels: np.ndarray, count: int = 3, metric: str = L2) -> list[Neighbour]:
    """The ``count`` smallest d_i, nearest first; the first one is i* = argmin d_i."""
    distances = face_distances(model, pixels, metric)
    return [Neighbour(model.names[index], model.paths[index], float(distances[index]))
            for index in np.argsort(distances, kind="stable")[:count]]


def evaluate(model: EigenfaceModel, faces: FaceSet, metric: str = L2) -> Evaluation:
    """Name every test face after its closest training face and count the outcomes."""
    predicted = [closest_faces(model, pixels, 1, metric)[0].name for pixels in faces.images]
    people = sorted({*faces.names, *predicted})
    confusion = np.zeros((len(people), len(people)), dtype=int)
    for truth, guess in zip(faces.names, predicted):
        confusion[people.index(truth), people.index(guess)] += 1
    return Evaluation(people, confusion)


def true_name(path: Path | None, folder: Path) -> str | None:
    """The person in an image from ``folder/test/<person>/``, named as ``load_faces`` does; else ``None``."""
    if path is None or path.parent.parent.resolve() != (folder / "test").resolve():
        return None
    return path.parent.name.replace("_", " ")


def show_recognition(pixels: np.ndarray, neighbours: list[Neighbour], folder: Path, truth: str | None,
                     metric: str = L2) -> Image.Image:
    """The test face and its closest training faces in a row, each titled above, as on the slide."""
    height, width = pixels.shape
    tile = (max(1, round(width * TILE_HEIGHT / height)), TILE_HEIGHT)
    faces = [Image.fromarray(pixels.astype(np.uint8))]
    titles = [(f"predicted: {neighbours[0].name}", f"true: {truth}" if truth else "test image")]
    for place, neighbour in enumerate(neighbours, 1):
        with Image.open(folder / neighbour.path) as image_file:
            faces.append(image_file.convert("L").copy())
        titles.append((f"{place}. {neighbour.name}", f"{metric}: {neighbour.distance:.4g}"))
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


def show_evaluation(evaluation: Evaluation, metric: str, components: int) -> Image.Image:
    """The confusion matrix, each cell shaded by its share of the row, with the accuracy of every person."""
    font = ImageFont.load_default(14)
    people = evaluation.people
    baseline_name, baseline = evaluation.most_common
    summary = (f"{metric}, K = {components}: {evaluation.accuracy:.1%} of {evaluation.total} test faces. "
               f"Always {baseline_name}: {baseline:.1%}.")
    label = round(max(font.getlength(text) for text in (*people, "true \\ predicted"))) + GAP
    right = CELL  # the column with each person's accuracy
    width = max(label + len(people) * CELL + right, round(font.getlength(summary))) + 2 * GAP
    top = GAP + 2 * CELL // 3
    header = CELL // 2
    result = Image.new("RGB", (width, top + header + len(people) * CELL + GAP), (255, 255, 255))
    draw = ImageDraw.Draw(result)
    draw.text((GAP, GAP), summary, fill=(0, 0, 0), font=font)
    draw.text((GAP, top + header / 2), "true \\ predicted", fill=(102, 102, 102), font=font, anchor="lm")
    for column, person in enumerate(people):
        center = GAP + label + column * CELL + CELL / 2
        draw.text((center, top + header / 2), person.split()[-1], fill=(0, 0, 0), font=font, anchor="mm")
    draw.text((GAP + label + len(people) * CELL + right / 2, top + header / 2), "correct", fill=(0, 0, 0), font=font,
              anchor="mm")
    for row, person in enumerate(people):
        y = top + header + row * CELL
        total = evaluation.confusion[row].sum()
        draw.text((GAP, y + CELL / 2), person, fill=(0, 0, 0), font=font, anchor="lm")
        for column, count in enumerate(evaluation.confusion[row]):
            share = count / total if total else 0
            x = GAP + label + column * CELL
            fill = tuple(round(255 + (channel - 255) * share) for channel in (31, 90, 160))
            draw.rectangle((x, y, x + CELL, y + CELL), fill=fill, outline=(200, 200, 200))
            draw.text((x + CELL / 2, y + CELL / 2), str(count), font=font, anchor="mm",
                      fill=(255, 255, 255) if share > 0.5 else (0, 0, 0))
        correct = f"{evaluation.confusion[row, row] / total:.0%}" if total else "–"
        draw.text((GAP + label + len(people) * CELL + right / 2, y + CELL / 2), correct, fill=(0, 0, 0), font=font,
                  anchor="mm")
    return result


def recognize(image: Image.Image, path: Path | None, folder: Path, model: EigenfaceModel, count: int,
              metric: str) -> tuple[Image.Image, str, dict[str, object]]:
    """The closest faces of ``image``: the panel, a one-line summary and the details."""
    pixels = face_pixels(image, model)
    neighbours = closest_faces(model, pixels, count, metric)
    truth = true_name(path, folder)
    return show_recognition(pixels, neighbours, folder, truth, metric), f"Predicted: {neighbours[0].name}.", {
        "Predicted": neighbours[0].name,
        "True name": truth or "Unknown (not an image of test/<person>/)",
        **{f"{place}. {neighbour.name}": f"{metric} {neighbour.distance:.4g} ({neighbour.path})"
           for place, neighbour in enumerate(neighbours, 1)},
    }


def evaluate_test_set(faces: FaceSet, model: EigenfaceModel, metric: str) -> tuple[Image.Image, str, dict[str, object]]:
    """The accuracy on the test faces: the confusion matrix, a one-line summary and the details."""
    evaluation = evaluate(model, faces, metric)
    baseline_name, baseline = evaluation.most_common
    return show_evaluation(evaluation, metric, model.components), f"Accuracy: {evaluation.accuracy:.1%}.", {
        "Test faces": evaluation.total,
        "Accuracy": f"{evaluation.accuracy:.1%}",
        f"Always {baseline_name}": f"{baseline:.1%}",
        **{person: f"{evaluation.confusion[row, row]} of {evaluation.confusion[row].sum()} correct"
           for row, person in enumerate(evaluation.people)},
    }


def ask_recognition(parent: tk.Misc, image: Image.Image, path: Path | None) -> tuple[Path, EigenfaceModel, int, str, str] | None:
    """Let the user pick the dataset, distance and view while previewing the result; ``None`` if cancelled.

    Returns the dataset folder, the model, how many faces to show, the distance and the view.
    """
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
    view = choice_row(body, "Show", VIEWS, lambda: view_changed())
    metric = choice_row(body, "Distance", METRICS, lambda: schedule_preview())
    count = number_row(body, *NEIGHBOURS, lambda: schedule_preview())
    count_row = body.winfo_children()[-1]
    hint = ttk.Label(body, text="w = U_K^T (x − μ). L2 is the slide's d_i = ‖w_test − w_i‖₂; whitened cosine usually "
                                "recognizes more.\nTest set accuracy names every face in test/ after its closest "
                                "training face.", foreground="#666666", justify="left")
    hint.pack(anchor="w", pady=(6, 0))

    canvas = tk.Canvas(body, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1], background="#20242b", highlightthickness=0)
    canvas.pack(pady=(12, 2))
    status = ttk.Label(body, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    preview_job: str | None = None
    model_cache: tuple[Path, EigenfaceModel] | None = None
    test_cache: tuple[Path, FaceSet] | None = None
    photo: ImageTk.PhotoImage | None = None

    def compute() -> tuple[Path, EigenfaceModel, int, str, str]:
        nonlocal model_cache
        chosen = Path(folder.get()).expanduser()
        if model_cache is None or model_cache[0] != chosen:
            model_cache = (chosen, saved_model(chosen))
        # The accuracy only uses each face's closest neighbour, so the hidden number does not matter there.
        shown = NEIGHBOURS[1] if view.get() == ACCURACY else read_number(count, NEIGHBOURS[0], NEIGHBOURS[2], NEIGHBOURS[3])
        return chosen, model_cache[1], shown, metric.get(), view.get()

    def view_changed() -> None:
        if view.get() == ACCURACY:
            count_row.pack_forget()
        else:
            count_row.pack(fill="x", pady=(6, 0), before=hint)
        schedule_preview()

    def test_faces(chosen: Path) -> FaceSet:
        nonlocal test_cache
        if test_cache is None or test_cache[0] != chosen:
            test_cache = (chosen, load_faces(chosen, "test"))
        return test_cache[1]

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
            chosen, model, shown, distance, selected = compute()
            if selected == ACCURACY:
                panel, summary, _ = evaluate_test_set(test_faces(chosen), model, distance)
            else:
                panel, summary, _ = recognize(image, path, chosen, model, shown, distance)
        except (ValueError, OSError) as error:
            status.configure(text=str(error))
            return
        photo = ImageTk.PhotoImage(_fit(panel, PREVIEW_SIZE), master=dialog)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        height, width = model.mean.shape
        note = "" if selected == ACCURACY or image.size == (width, height) else (
            f" This image was resized from {image.width} × {image.height} px, so the result only means "
            "something for a tightly cropped face.")
        status.configure(text=f"{summary} K = {model.components}, {len(model.names)} training faces.{note}")

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
        folder, model, count, metric, view = choice
        if view == ACCURACY:
            panel, summary, details = evaluate_test_set(load_faces(folder, "test"), model, metric)
        else:
            panel, summary, details = recognize(document.original, document.path, folder, model, count, metric)
        height, width = model.mean.shape
        return ToolResult(
            image=panel,
            message=summary,
            details={
                "Operation": f"Face recognition: {view.lower()}",
                "Dataset": str(folder),
                "Training faces": len(model.names),
                "Components (K)": model.components,
                "Face size": f"{width} × {height} px",
                "Distance": metric,
                **details,
            },
        )
