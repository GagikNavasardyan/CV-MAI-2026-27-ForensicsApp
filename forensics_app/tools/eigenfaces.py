"""Eigenfaces: PCA of a folder of training faces, saved for recognition, with the first eigenfaces shown."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import center_over
from .hog import number_row, read_number
from .image_lookup import IMAGE_SUFFIXES


DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "datasets" / "lfw"  # datasets/ is git-ignored
SPLITS = ("train", "test")
LABELS_FILE = "labels.csv"  # split, path, name of every exported image
MODEL_FOLDER = "model"
# The Set 6 notebook's dataset and split.
LFW_SETTINGS = {"min_faces_per_person": 70, "resize": 0.4}
TEST_SIZE = 0.25
RANDOM_STATE = 42
# Number setting: (label, default, minimum, maximum).
COMPONENTS = ("Components (K)", 100, 1, 1000)  # the notebook keeps 100
SHOWN = ("Eigenfaces shown", 12, 1, 64)  # the slide shows 12, in rows of 4
COLUMNS = 4
TILE_HEIGHT = 150  # the faces are tiny (LFW at 0.4 is 50 × 37 px), so the tiles are enlarged
GAP = 12
PREVIEW_SIZE = (480, 420)
PREVIEW_DELAY_MS = 400


@dataclass(frozen=True)
class FaceSet:
    """Same-size greyscale faces with the person each one shows."""

    images: np.ndarray  # (n, height, width) float64, 0–255
    names: list[str]
    paths: list[str]  # relative to the dataset folder, with "/" separators

    @property
    def people(self) -> int:
        return len(set(self.names))


@dataclass(frozen=True)
class Pca:
    """The whole decomposition of the training faces, so a different K needs no new SVD."""

    mean: np.ndarray  # (d,) the mean face, flattened
    components: np.ndarray  # (r, d) orthonormal eigenfaces as rows, most variance first
    variances: np.ndarray  # (r,) the covariance's eigenvalues, one per component


@dataclass(frozen=True)
class EigenfaceModel:
    """What recognition needs: the mean, the first K eigenfaces and every training face's coefficients."""

    mean: np.ndarray  # (height, width) μ
    eigenfaces: np.ndarray  # (K, height, width); flattened and transposed they are U_K
    coefficients: np.ndarray  # (n, K) row i is w_i = U_K^T (x_i − μ)
    explained_variance_ratio: np.ndarray  # (K,) share of the training faces' variance per eigenface
    names: list[str]  # the person in training face i
    paths: list[str]  # training face i, relative to the dataset folder

    @property
    def components(self) -> int:
        return self.eigenfaces.shape[0]

    def project(self, images: np.ndarray) -> np.ndarray:
        """w = U_K^T (x − μ) for one (height, width) face or a stack of them; returns (K,) or (n, K)."""
        flat = np.asarray(images, dtype=np.float64).reshape(-1, self.mean.size) - self.mean.ravel()
        weights = flat @ self.eigenfaces.reshape(self.components, -1).T
        return weights[0] if np.ndim(images) == 2 else weights


def has_dataset(folder: Path) -> bool:
    return (folder / "train").is_dir()


def export_lfw(folder: Path) -> dict[str, int]:
    """Download LFW as the notebook does, split it 75/25 and save it to ``folder``; return the image counts.

    Each image goes to ``<split>/<Person_Name>/<Person_Name>_<index>.png``, and ``labels.csv`` lists the
    split, path and name of all of them. A previous export in ``folder`` (and its model) is replaced.
    """
    from sklearn.datasets import fetch_lfw_people  # slow to import, and only needed here
    from sklearn.model_selection import train_test_split

    people = fetch_lfw_people(**LFW_SETTINGS)
    images = people.images
    if images.max() <= 1.0:  # recent scikit-learn scales the pixels to 0–1
        images = images * 255.0
    images = np.clip(np.rint(images), 0, 255).astype(np.uint8)
    train, test = train_test_split(np.arange(len(images)), test_size=TEST_SIZE, random_state=RANDOM_STATE)
    for leftover in (*SPLITS, MODEL_FOLDER):
        shutil.rmtree(folder / leftover, ignore_errors=True)
    rows = []
    for split, indices in zip(SPLITS, (train, test)):
        for index in indices:
            name = str(people.target_names[people.target[index]])
            person = name.replace(" ", "_")
            path = Path(split) / person / f"{person}_{index:04d}.png"
            (folder / path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(images[index]).save(folder / path)
            rows.append((split, path.as_posix(), name))
    with open(folder / LABELS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("split", "path", "name"))
        writer.writerows(rows)
    return {"train": len(train), "test": len(test), "people": len(people.target_names)}


def load_faces(folder: Path, split: str = "train") -> FaceSet:
    """The images in ``folder/split/<person>/``, named after their subfolder ("_" read as a space).

    They are read in greyscale, and any whose size differs from the first are resized to it.
    Raises ``ValueError`` when there are none.
    """
    root = folder / split
    paths = sorted((path for path in root.glob("*/*") if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()),
                   key=lambda path: path.relative_to(root).as_posix().lower()) if root.is_dir() else []
    if not paths:
        raise ValueError(f"No images in {root}/<person name>/. Choose a dataset folder or download LFW.")
    images = []
    size = None
    for path in paths:
        with Image.open(path) as image_file:
            image = image_file.convert("L")
        size = size or image.size
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        images.append(np.asarray(image, dtype=np.float64))
    return FaceSet(np.stack(images), [path.parent.name.replace("_", " ") for path in paths],
                   [path.relative_to(folder).as_posix() for path in paths])


def principal_components(faces: FaceSet) -> Pca:
    """PCA by the SVD of the centred faces: the rows of V^T are the covariance's eigenvectors."""
    flat = faces.images.reshape(len(faces.images), -1)
    mean = flat.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(flat - mean, full_matrices=False)
    # An eigenvector's sign is arbitrary; like scikit-learn, make each one's largest entry positive.
    vt *= np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(axis=1)])[:, None]
    variances = singular_values ** 2 / max(1, len(flat) - 1)
    return Pca(mean, vt, variances)


def max_components(faces: FaceSet) -> int:
    """Centring leaves n faces at most n − 1 directions with any variance."""
    return max(1, min(len(faces.images) - 1, faces.images[0].size))


def build_model(faces: FaceSet, components: int, pca: Pca | None = None) -> EigenfaceModel:
    """Keep the first ``components`` eigenfaces and project every training face onto them."""
    if not 1 <= components <= max_components(faces):
        raise ValueError(f"K must be between 1 and {max_components(faces)} for {len(faces.images)} training faces.")
    pca = pca or principal_components(faces)
    shape = faces.images.shape[1:]
    u_k = pca.components[:components].T  # (d, K)
    coefficients = (faces.images.reshape(len(faces.images), -1) - pca.mean) @ u_k
    total = pca.variances.sum()
    ratio = pca.variances[:components] / total if total > 0 else np.zeros(components)
    return EigenfaceModel(pca.mean.reshape(shape), u_k.T.reshape(components, *shape), coefficients, ratio,
                          list(faces.names), list(faces.paths))


def save_model(model: EigenfaceModel, folder: Path) -> Path:
    """Write the model to ``folder/model/`` as .npy arrays plus ``model.json``; return that folder."""
    target = folder / MODEL_FOLDER
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "mean.npy", model.mean)
    np.save(target / "eigenfaces.npy", model.eigenfaces)
    np.save(target / "coefficients.npy", model.coefficients)
    np.save(target / "explained_variance_ratio.npy", model.explained_variance_ratio)
    height, width = model.mean.shape
    description = {
        "image_size": {"width": width, "height": height},
        "components": model.components,
        "train": [{"path": path, "name": name} for path, name in zip(model.paths, model.names)],  # row i of coefficients
    }
    (target / "model.json").write_text(json.dumps(description, indent=1), encoding="utf-8")
    return target


def load_model(folder: Path) -> EigenfaceModel:
    """The model ``save_model`` wrote to ``folder/model/``."""
    source = folder / MODEL_FOLDER
    description = json.loads((source / "model.json").read_text(encoding="utf-8"))
    return EigenfaceModel(np.load(source / "mean.npy"), np.load(source / "eigenfaces.npy"),
                          np.load(source / "coefficients.npy"), np.load(source / "explained_variance_ratio.npy"),
                          [row["name"] for row in description["train"]], [row["path"] for row in description["train"]])


def _grey(values: np.ndarray) -> Image.Image:
    """Stretched to 0–255 as ``plt.imshow(cmap="gray")`` does: an eigenface's sign and scale are arbitrary."""
    low, high = float(values.min()), float(values.max())
    scaled = (values - low) / (high - low) if high > low else np.zeros_like(values)
    return Image.fromarray(np.rint(scaled * 255).astype(np.uint8))


def show_eigenfaces(model: EigenfaceModel, count: int = 12, columns: int = COLUMNS) -> Image.Image:
    """The first ``count`` eigenfaces in rows of ``columns``, each titled above, as on the slide."""
    count = min(count, model.components)
    height, width = model.mean.shape
    tile = (max(1, round(width * TILE_HEIGHT / height)), TILE_HEIGHT)
    font = ImageFont.load_default(14)
    title = round(font.size * 1.6)
    columns = min(columns, count)
    rows = -(-count // columns)
    result = Image.new("RGB", (columns * tile[0] + (columns + 1) * GAP, rows * (tile[1] + title) + (rows + 1) * GAP),
                       (255, 255, 255))
    draw = ImageDraw.Draw(result)
    for index in range(count):
        row, column = divmod(index, columns)
        left = GAP + column * (tile[0] + GAP)
        top = GAP + row * (tile[1] + title + GAP)
        draw.text((left + tile[0] / 2, top + title / 2), f"eigenface {index}", fill=(0, 0, 0), font=font, anchor="mm")
        result.paste(_grey(model.eigenfaces[index]).resize(tile, Image.Resampling.LANCZOS), (left, top + title))
        draw.rectangle((left - 1, top + title - 1, left + tile[0], top + title + tile[1]), outline=(0, 0, 0))
    return result


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale for display only, up or down, to fit ``size``."""
    scale = min(size[0] / image.width, size[1] / image.height)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                        Image.Resampling.LANCZOS)


def ask_eigenfaces(parent: tk.Misc) -> tuple[Path, FaceSet, EigenfaceModel, int] | None:
    """Let the user pick (or download) a dataset and K while previewing the eigenfaces; ``None`` if cancelled.

    Returns the dataset folder, its training faces, the model and how many eigenfaces to show.
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Eigenfaces (PCA)")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()
    controls = ttk.Frame(body)
    controls.grid(row=0, column=0, sticky="n")
    preview_side = ttk.Frame(body, padding=(20, 0, 0, 0))
    preview_side.grid(row=0, column=1, sticky="n")

    ttk.Label(controls, text="Dataset folder").pack(anchor="w")
    folder = tk.StringVar(dialog, value=str(DEFAULT_DATASET))
    ttk.Entry(controls, textvariable=folder, width=42).pack(anchor="w", pady=(4, 0))
    folder_buttons = ttk.Frame(controls)
    folder_buttons.pack(anchor="w", pady=(6, 0))
    browse_button = ttk.Button(folder_buttons, text="Browse…", command=lambda: browse())
    browse_button.pack(side="left")
    download_button = ttk.Button(folder_buttons, text="Download LFW here", command=lambda: download())
    download_button.pack(side="left", padx=(6, 0))
    ttk.Label(controls, text="Faces are read from train/<person name>/; test/ is kept for recognition.\n"
                             "LFW: people with 70+ photos, scaled by 0.4, split 75/25 (seed 42).",
              foreground="#666666", justify="left").pack(anchor="w", pady=(6, 0))

    ttk.Label(controls, text="PCA", padding=(0, 16, 0, 0)).pack(anchor="w")
    components = number_row(controls, *COMPONENTS, lambda: schedule_preview())
    shown = number_row(controls, *SHOWN, lambda: schedule_preview())
    ttk.Label(controls, text="w_i = U_K^T (x_i − μ) is stored for every training face.",
              foreground="#666666").pack(anchor="w", pady=(6, 0))

    ttk.Label(preview_side, text="Preview").pack(anchor="w")
    canvas = tk.Canvas(preview_side, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1],
                       background="#20242b", highlightthickness=0)
    canvas.pack(pady=(4, 2))
    status = ttk.Label(preview_side, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    preview_job: str | None = None
    poll_job: str | None = None
    downloading = False
    dataset_cache: tuple[Path, FaceSet, Pca] | None = None  # loading and the SVD are slow; slicing K is not
    photo: ImageTk.PhotoImage | None = None

    def dataset() -> tuple[Path, FaceSet, Pca]:
        nonlocal dataset_cache
        path = Path(folder.get()).expanduser()
        if dataset_cache is None or dataset_cache[0] != path:
            faces = load_faces(path)
            dataset_cache = (path, faces, principal_components(faces))
        return dataset_cache

    def compute() -> tuple[Path, FaceSet, EigenfaceModel, int]:
        path, faces, pca = dataset()
        k = read_number(components, COMPONENTS[0], COMPONENTS[2], COMPONENTS[3])
        count = read_number(shown, SHOWN[0], SHOWN[2], SHOWN[3])
        return path, faces, build_model(faces, k, pca), count

    def schedule_preview() -> None:
        nonlocal preview_job
        if downloading:
            return
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        status.configure(text="Updating preview…")
        preview_job = dialog.after(PREVIEW_DELAY_MS, update_preview)

    def update_preview() -> None:
        nonlocal preview_job, photo
        preview_job = None
        canvas.delete("all")
        try:
            _, faces, model, count = compute()
        except (ValueError, OSError) as error:
            status.configure(text=str(error))
            return
        photo = ImageTk.PhotoImage(_fit(show_eigenfaces(model, count), PREVIEW_SIZE), master=dialog)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        height, width = model.mean.shape
        status.configure(text=f"{len(faces.images)} training faces of {faces.people} people, {width} × {height} px. "
                              f"K = {model.components} keeps {model.explained_variance_ratio.sum():.1%} of the variance.")

    def browse() -> None:
        chosen = filedialog.askdirectory(title="Choose the dataset folder (with train/ inside)", parent=dialog,
                                         initialdir=folder.get() if Path(folder.get()).is_dir() else None)
        if chosen:
            folder.set(chosen)

    def set_busy(busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for button in (browse_button, download_button, apply_button, cancel_button):
            button.configure(state=state)

    def download() -> None:
        nonlocal downloading, poll_job
        target = Path(folder.get()).expanduser()
        if has_dataset(target) and not messagebox.askyesno(
                "Eigenfaces", f"{target} already has a dataset. Replace it (and its model)?", parent=dialog):
            return
        downloading = True
        set_busy(True)
        status.configure(text="Downloading LFW (about 200 MB the first time, then cached) and saving the images…")
        outcome: dict[str, Any] = {}

        def work() -> None:  # no Tk calls here: Tk may only be used from the main thread
            try:
                target.mkdir(parents=True, exist_ok=True)
                outcome["counts"] = export_lfw(target)
            except Exception as error:  # report any failure in the dialog
                outcome["error"] = error

        thread = threading.Thread(target=work, daemon=True)
        thread.start()

        def poll() -> None:
            nonlocal downloading, poll_job, dataset_cache
            if thread.is_alive():
                poll_job = dialog.after(200, poll)
                return
            poll_job = None
            downloading = False
            set_busy(False)
            if "error" in outcome:
                status.configure(text=f"Could not download LFW: {outcome['error']}")
                return
            dataset_cache = None  # the folder's contents changed
            folder.set(str(target))
            schedule_preview()

        poll()

    folder.trace_add("write", lambda *_: schedule_preview())
    result = []

    def close() -> None:
        if downloading:  # an export cut short would leave a partial dataset
            return
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        if downloading:
            return
        try:
            result.append(compute())
        except (ValueError, OSError) as error:
            messagebox.showerror("Eigenfaces", str(error), parent=dialog)
            return
        close()

    buttons = ttk.Frame(preview_side)
    buttons.pack(fill="x", pady=(16, 0))
    cancel_button = ttk.Button(buttons, text="Cancel", command=close)
    cancel_button.pack(side="right")
    apply_button = ttk.Button(buttons, text="Build and save", command=ok, default="active")
    apply_button.pack(side="right", padx=(0, 6))
    dialog.bind("<Return>", ok)
    dialog.bind("<Escape>", lambda _event: close())
    dialog.protocol("WM_DELETE_WINDOW", close)

    center_over(dialog, parent.winfo_toplevel())
    dialog.deiconify()
    schedule_preview()
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class EigenfacesTool(ForensicsTool):
    tool_id = "eigenfaces"
    title = "Eigenfaces (PCA)"
    category = "Face recognition"
    description = "Build eigenfaces from a folder of training faces, save the model and show the first K."
    requires_image = False
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        choice = ask_eigenfaces(parent)
        if choice is None:
            return None
        folder, faces, model, count = choice
        target = save_model(model, folder)
        gallery = target / "eigenfaces.png"  # opened as the document, so it shows even with no image loaded
        show_eigenfaces(model, count).save(gallery)
        height, width = model.mean.shape
        return ToolResult(
            message=f"Built {model.components} eigenfaces and saved the model to {target}.",
            details={
                "Operation": "Eigenfaces (PCA)",
                "Dataset": str(folder),
                "Training faces": f"{len(faces.images)} of {faces.people} people",
                "Face size": f"{width} × {height} px ({width * height} values)",
                "Components (K)": model.components,
                "Variance kept": f"{model.explained_variance_ratio.sum():.1%}",
                "Eigenface 0 explains": f"{model.explained_variance_ratio[0]:.1%}",
                "Shown": f"First {min(count, model.components)} eigenfaces",
                "Model saved to": str(target),
                "Files": "mean.npy, eigenfaces.npy, coefficients.npy, explained_variance_ratio.npy, model.json",
            },
            document_path=gallery,
        )
