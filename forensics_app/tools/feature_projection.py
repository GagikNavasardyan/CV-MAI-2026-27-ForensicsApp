"""Feature projection (Set 8.3): extract the CNN feature vectors of the CIFAR-10 test set, save them, and project
them onto 2D with t-SNE and UMAP.

The features are the output of the model's last hidden layer (the ReLU before the classifier), taken from the same
ONNX models as CNN classification (Set 8.2) with OpenCV's DNN module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tarfile
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable
import urllib.request

import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .cnn_classification import MODELS, CnnModel, _figure_image, preprocess
from .dialogs import center_over
from .eigenfaces import _fit
from .hog import choice_row, number_row, read_number


DATASET = Path(__file__).resolve().parents[2] / "datasets" / "cifar10"  # datasets/ is git-ignored
# The binary version: plain bytes, so reading it needs no pickle.
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
ARCHIVE_FOLDER = "cifar-10-batches-bin"
TEST_FILE = "test.npz"
FEATURES_FOLDER = "features"
BATCH = 100  # images per forward pass
SEED = 0
# Number setting: (label, default, minimum, maximum).
SAMPLE = ("Test images", 2000, 100, 10000)
PERPLEXITY = ("t-SNE perplexity", 30, 5, 100)
NEIGHBOURS = ("UMAP neighbours", 15, 2, 200)
UMAP_MIN_DIST = 0.1  # umap-learn's default
COLOUR_BY = ("True class", "Predicted class")
PREVIEW_SIZE = (760, 420)
POLL_MS = 200


@dataclass(frozen=True)
class TestSet:
    images: np.ndarray  # (n, 32, 32, 3) uint8 RGB
    labels: np.ndarray  # (n,) class index
    classes: list[str]


@dataclass(frozen=True)
class Features:
    """What the extraction saves: one feature vector per test image, with its true and predicted class."""

    features: np.ndarray  # (n, d) float32
    labels: np.ndarray  # (n,)
    predictions: np.ndarray  # (n,) the model's most probable class
    classes: list[str]
    model: str  # the ONNX file name
    tensor: str  # the layer the features come from

    @property
    def accuracy(self) -> float:
        return float(np.mean(self.labels == self.predictions))


@dataclass(frozen=True)
class Projection:
    tsne: np.ndarray  # (m, 2)
    umap: np.ndarray  # (m, 2)
    indices: np.ndarray  # (m,) rows of the features that were projected
    perplexity: int
    neighbours: int


# ---------------------------------------------------------------------------------------------------- test set

def has_test_set(folder: Path = DATASET) -> bool:
    return (folder / TEST_FILE).is_file()


def read_cifar_binary(archive: Path) -> TestSet:
    """The test batch of ``cifar-10-binary.tar.gz``: each record is 1 label byte and 3072 pixel bytes (the red,
    green and blue 32 × 32 planes); the class names are in ``batches.meta.txt``."""
    with tarfile.open(archive, "r:gz") as tar:
        records = tar.extractfile(f"{ARCHIVE_FOLDER}/test_batch.bin").read()
        names = tar.extractfile(f"{ARCHIVE_FOLDER}/batches.meta.txt").read().decode("ascii").split()
    rows = np.frombuffer(records, dtype=np.uint8).reshape(-1, 1 + 3 * 32 * 32)
    images = rows[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return TestSet(np.ascontiguousarray(images), rows[:, 0].astype(np.int64), names)


def download_test_set(folder: Path = DATASET, progress: Callable[[int, int], None] | None = None) -> TestSet:
    """Download CIFAR-10 (about 160 MB), keep only its 10 000 test images in ``folder/test.npz`` and delete the
    archive. ``progress(bytes, total)`` is called while downloading."""
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / Path(CIFAR_URL).name
    try:
        with urllib.request.urlopen(CIFAR_URL, timeout=60) as response, open(archive, "wb") as file:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            while chunk := response.read(1 << 20):
                file.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        test = read_cifar_binary(archive)
    finally:
        archive.unlink(missing_ok=True)
    np.savez_compressed(folder / TEST_FILE, images=test.images, labels=test.labels, classes=np.array(test.classes))
    return test


def load_test_set(folder: Path = DATASET) -> TestSet:
    with np.load(folder / TEST_FILE) as data:
        return TestSet(data["images"], data["labels"], [str(name) for name in data["classes"]])


# ---------------------------------------------------------------------------------------------------- features

def feature_tensor(net: Any) -> str:
    """The output of the last ReLU layer, i.e. the hidden layer before the classifier.

    OpenCV's ONNX importer names layers after the Keras ones (``…/dense_4_1/Relu``) and their outputs with ``:0``.
    """
    relus = [name for name in net.getLayerNames() if name.endswith("/Relu")]
    if not relus:
        raise ValueError("The model has no ReLU layer named like Keras exports it, so its features cannot be found.")
    return relus[-1] + ":0"


def features_path(model: Path, folder: Path = DATASET) -> Path:
    return folder / FEATURES_FOLDER / f"{model.stem}.npz"


def extract_features(model: CnnModel, test: TestSet,
                     progress: Callable[[int, int], None] | None = None) -> Features:
    """Feed the test images through the model in batches, preprocessed as its passport says, and keep the
    last hidden layer and the predicted class of each one."""
    if model.classes != test.classes:
        raise ValueError(f"The model's classes ({', '.join(model.classes)}) are not CIFAR-10's.")
    tensor = feature_tensor(model.net)
    features, predictions = [], []
    for start in range(0, len(test.images), BATCH):
        batch = np.concatenate([preprocess(Image.fromarray(image), model.passport)
                                for image in test.images[start:start + BATCH]])
        model.net.setInput(batch)
        features.append(np.asarray(model.net.forward(tensor), dtype=np.float32).reshape(len(batch), -1))
        model.net.setInput(batch)  # OpenCV gives one output per pass
        predictions.append(np.asarray(model.net.forward()).reshape(len(batch), -1).argmax(axis=1))
        if progress:
            progress(start + len(batch), len(test.images))
    return Features(np.concatenate(features), test.labels, np.concatenate(predictions), test.classes,
                    model.path.name, tensor)


def save_features(features: Features, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features.features, labels=features.labels, predictions=features.predictions,
                        classes=np.array(features.classes), model=features.model, tensor=features.tensor)
    return path


def load_features(path: Path) -> Features:
    with np.load(path) as data:
        return Features(data["features"], data["labels"], data["predictions"], [str(name) for name in data["classes"]],
                        str(data["model"]), str(data["tensor"]))


# ---------------------------------------------------------------------------------------------------- projection

def stratified_sample(labels: np.ndarray, size: int, seed: int = SEED) -> np.ndarray:
    """``size`` row indices with the classes in the same proportions as ``labels`` (all of them if fewer)."""
    if size >= len(labels):
        return np.arange(len(labels))
    from sklearn.model_selection import train_test_split

    chosen, _ = train_test_split(np.arange(len(labels)), train_size=size, stratify=labels, random_state=seed)
    return np.sort(chosen)


def project(features: Features, size: int, perplexity: int, neighbours: int) -> Projection:
    """t-SNE and UMAP of a stratified sample of the features, both to 2D with a fixed seed."""
    from sklearn.manifold import TSNE  # slow to import, and only needed here
    import umap

    indices = stratified_sample(features.labels, size)
    vectors = features.features[indices]
    if not 1 <= perplexity < len(vectors):
        raise ValueError(f"The perplexity must be below the number of images ({len(vectors)}).")
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=SEED).fit_transform(vectors)
    flat = umap.UMAP(n_components=2, n_neighbors=min(neighbours, len(vectors) - 1), min_dist=UMAP_MIN_DIST,
                     random_state=SEED, n_jobs=1, compatibility_layout=True).fit_transform(vectors)
    return Projection(tsne, flat, indices, perplexity, neighbours)


def show_projection(features: Features, projection: Projection, colour_by: str = COLOUR_BY[0]) -> Image.Image:
    """t-SNE and UMAP side by side, one colour per class, as on the slide."""
    labels = (features.predictions if colour_by == COLOUR_BY[1] else features.labels)[projection.indices]
    colours = [f"C{index % 10}" for index in range(len(features.classes))]
    figure = Figure(figsize=(12, 6.2), layout="constrained")
    for column, (points, name, setting) in enumerate((
            (projection.tsne, "t-SNE", f"perplexity {projection.perplexity}"),
            (projection.umap, "UMAP", f"{projection.neighbours} neighbours, min dist {UMAP_MIN_DIST:g}"))):
        axes = figure.add_subplot(1, 2, column + 1)
        for index, label in enumerate(features.classes):
            chosen = labels == index
            axes.scatter(points[chosen, 0], points[chosen, 1], s=4, color=colours[index], label=label)
        axes.set_title(f"{name} ({setting})")
        axes.set_xlabel(f"{name} 1")
        axes.set_ylabel(f"{name} 2")
        axes.set_xticks([])
        axes.set_yticks([])
    figure.legend(*axes.get_legend_handles_labels(), loc="outside lower center", ncols=len(features.classes),
                  markerscale=3, frameon=False)
    figure.suptitle(f"{features.model}: {len(projection.indices)} CIFAR-10 test images, {features.tensor} "
                    f"({features.features.shape[1]} values), coloured by {colour_by.lower()}")
    return _figure_image(figure)


# ---------------------------------------------------------------------------------------------------- dialog

def ask_projection(parent: tk.Misc) -> tuple[Features, Projection, str, Image.Image] | None:
    """Let the user extract (if needed) and project a model's features; ``None`` if cancelled."""
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Feature projection (t-SNE, UMAP)")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()
    controls = ttk.Frame(body)
    controls.grid(row=0, column=0, sticky="n")
    preview_side = ttk.Frame(body, padding=(20, 0, 0, 0))
    preview_side.grid(row=0, column=1, sticky="n")

    ttk.Label(controls, text="1. Features of the CIFAR-10 test set").pack(anchor="w")
    model_name = choice_row(controls, "Model", tuple(MODELS), lambda: refresh())
    saved = ttk.Label(controls, foreground="#666666", wraplength=330, justify="left")
    saved.pack(anchor="w", pady=(6, 0))
    extract_button = ttk.Button(controls, text="Extract and save features", command=lambda: extract())
    extract_button.pack(anchor="w", pady=(6, 0))

    ttk.Label(controls, text="2. Projection to 2D", padding=(0, 18, 0, 0)).pack(anchor="w")
    sample = number_row(controls, *SAMPLE, lambda: mark_stale())
    perplexity = number_row(controls, *PERPLEXITY, lambda: mark_stale())
    neighbours = number_row(controls, *NEIGHBOURS, lambda: mark_stale())
    colour_by = choice_row(controls, "Colour by", COLOUR_BY, lambda: redraw())
    ttk.Label(controls, text="A stratified sample (seed 0) is projected; the same\nseed is used for t-SNE and UMAP.",
              foreground="#666666", justify="left").pack(anchor="w", pady=(6, 0))
    project_button = ttk.Button(controls, text="Project", command=lambda: run_projection())
    project_button.pack(anchor="w", pady=(6, 0))

    ttk.Label(preview_side, text="Preview").pack(anchor="w")
    canvas = tk.Canvas(preview_side, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1], background="#20242b",
                       highlightthickness=0)
    canvas.pack(pady=(4, 2))
    status = ttk.Label(preview_side, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    busy = False
    stale = False
    loaded: Features | None = None
    shown: tuple[Projection, str, Image.Image] | None = None
    photo: ImageTk.PhotoImage | None = None
    poll_job: str | None = None

    def model_path() -> Path:
        return MODELS[model_name.get()]

    def refresh() -> None:
        """Load the chosen model's saved features, if any, and clear the preview."""
        nonlocal loaded, shown
        loaded, shown = None, None
        canvas.delete("all")
        path = features_path(model_path())
        if not path.is_file():
            saved.configure(text=f"Not extracted yet. Extracting runs the model on the 10 000 test images "
                                 f"and saves {path.relative_to(DATASET.parents[1])}.")
            status.configure(text="Extract the features first.")
            return
        try:
            loaded = load_features(path)
        except (OSError, ValueError, KeyError) as error:
            saved.configure(text=f"Could not read {path.name}: {error}")
            return
        saved.configure(text=f"Saved: {len(loaded.features)} × {loaded.features.shape[1]} features from "
                             f"{loaded.tensor}. The model gets {loaded.accuracy:.1%} of these images right.")
        status.configure(text="Press Project to compute t-SNE and UMAP.")

    def set_busy(state: bool) -> None:
        nonlocal busy
        busy = state
        for button in (extract_button, project_button, apply_button, cancel_button):
            button.configure(state="disabled" if state else "normal")

    def in_background(work: Callable[[Callable[[str], None]], Any], done: Callable[[Any], None],
                      failed: str) -> None:
        """Run ``work(report)`` on a thread; then ``done(result)`` on Tk's thread, or show the error."""
        nonlocal poll_job
        set_busy(True)
        outcome: dict[str, Any] = {"message": status.cget("text")}

        def report(message: str) -> None:  # worker thread: no Tk calls, the poll shows it
            outcome["message"] = message

        def run() -> None:
            try:
                outcome["result"] = work(report)
            except Exception as error:  # report any failure in the dialog
                outcome["error"] = error

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        def poll() -> None:
            nonlocal poll_job
            status.configure(text=outcome["message"])
            if thread.is_alive():
                poll_job = dialog.after(POLL_MS, poll)
                return
            poll_job = None
            set_busy(False)
            if "error" in outcome:
                status.configure(text=f"{failed}: {outcome['error']}")
            else:
                done(outcome["result"])

        poll()

    def extract() -> None:
        path = model_path()
        target = features_path(path)
        if target.is_file() and not messagebox.askyesno(
                "Feature projection", f"{target.name} already exists. Extract the features again?", parent=dialog):
            return

        def work(report: Callable[[str], None]) -> Path:
            if has_test_set():
                test = load_test_set()
            else:
                test = download_test_set(progress=lambda done, total: report(
                    f"Downloading CIFAR-10: {done / 1e6:.0f} of {total / 1e6:.0f} MB…"))
            model = CnnModel.load(path)
            features = extract_features(model, test, lambda done, total: report(
                f"Extracting features with {path.name}: {done} of {total} images…"))
            return save_features(features, target)

        status.configure(text="Starting…")
        in_background(work, lambda _: refresh(), "Could not extract the features")

    def read_settings() -> tuple[int, int, int]:
        return tuple(read_number(variable, label, minimum, maximum) for variable, (label, _, minimum, maximum)
                     in ((sample, SAMPLE), (perplexity, PERPLEXITY), (neighbours, NEIGHBOURS)))

    def run_projection() -> None:
        nonlocal stale
        if loaded is None:
            status.configure(text="Extract the features first.")
            return
        try:
            size, perplexity_value, neighbours_value = read_settings()
        except ValueError as error:
            status.configure(text=str(error))
            return
        features = loaded
        stale = False
        status.configure(text=f"Projecting {min(size, len(features.features))} feature vectors with t-SNE and "
                              "UMAP (this takes a while)…")
        in_background(lambda _report: project(features, size, perplexity_value, neighbours_value),
                      lambda projection: draw(projection), "Could not project the features")

    def draw(projection: Projection) -> None:
        nonlocal shown, photo
        if loaded is None:
            return
        picture = show_projection(loaded, projection, colour_by.get())
        shown = (projection, colour_by.get(), picture)
        canvas.delete("all")
        photo = ImageTk.PhotoImage(_fit(picture, PREVIEW_SIZE), master=dialog)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        status.configure(text=f"{len(projection.indices)} test images projected. Apply keeps this result."
                         + (" The settings changed since: press Project to update." if stale else ""))

    def redraw() -> None:
        if shown is not None and not busy:
            draw(shown[0])  # recolouring needs no new projection

    def mark_stale() -> None:
        nonlocal stale
        if shown is not None:
            stale = True
            status.configure(text="The settings changed: press Project to update the preview.")

    result = []

    def close() -> None:
        if busy:  # a download or extraction cut short would leave partial files
            return
        if poll_job is not None:
            dialog.after_cancel(poll_job)
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        if busy:
            return
        if shown is None or loaded is None:
            messagebox.showerror("Feature projection", "Press Project first.", parent=dialog)
            return
        result.append((loaded, *shown))
        close()

    buttons = ttk.Frame(preview_side)
    buttons.pack(fill="x", pady=(16, 0))
    cancel_button = ttk.Button(buttons, text="Cancel", command=close)
    cancel_button.pack(side="right")
    apply_button = ttk.Button(buttons, text="Apply", command=ok, default="active")
    apply_button.pack(side="right", padx=(0, 6))
    dialog.bind("<Escape>", lambda _event: close())
    dialog.protocol("WM_DELETE_WINDOW", close)

    center_over(dialog, parent.winfo_toplevel())
    dialog.deiconify()
    refresh()
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class FeatureProjectionTool(ForensicsTool):
    tool_id = "feature_projection"
    title = "Feature projection"
    category = "Deep learning"
    description = "Extract CNN features of the CIFAR-10 test set and project them to 2D with t-SNE and UMAP."
    requires_image = False
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        choice = ask_projection(parent)
        if choice is None:
            return None
        features, projection, colour_by, picture = choice
        target = features_path(Path(features.model)).with_name(f"{Path(features.model).stem}_projection.png")
        picture.save(target)  # opened as the document, so it shows even with no image loaded
        return ToolResult(
            message=f"Projected {len(projection.indices)} {features.model} feature vectors with t-SNE and UMAP.",
            details={
                "Operation": "Feature projection (t-SNE, UMAP)",
                "Model": features.model,
                "Features": f"{features.tensor}, {features.features.shape[1]} values per image",
                "Saved features": str(features_path(Path(features.model))),
                "Test images": f"{len(projection.indices)} of {len(features.features)} (stratified, seed {SEED})",
                "Model accuracy on all": f"{features.accuracy:.1%}",
                "t-SNE": f"perplexity {projection.perplexity}, PCA initialisation, seed {SEED}",
                "UMAP": f"{projection.neighbours} neighbours, min dist {UMAP_MIN_DIST:g}, seed {SEED}",
                "Coloured by": colour_by,
                "Figure saved to": str(target),
            },
            document_path=target,
        )
