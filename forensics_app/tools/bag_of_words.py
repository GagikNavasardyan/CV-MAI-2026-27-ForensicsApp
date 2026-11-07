"""Bag of visual words (Set 7): a codebook of visual words built from a folder of classed images (7.1) and the
histogram of visual words of an image (7.2), saved so a classifier (7.3) can be trained on the histograms."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .corner_detection import read_settings, setting_rows
from .dialogs import center_over
from .eigenfaces import RANDOM_STATE, TEST_SIZE, _fit
from .hog import choice_row, number_row, read_number
from .image_lookup import IMAGE_SUFFIXES
from .keypoints import DESCRIPTOR_SIZES, INTEGER_SETTINGS, METHOD_PARAMETERS, METHODS, UPSAMPLING, Keypoints, \
    extract_keypoints


DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "img" / "Caltech_101_subset"  # img/ is git-ignored
VOCABULARY_FOLDER = "bovw"  # written inside the dataset folder; never read as a class
WORDS_PER_CLASS = 10  # the slide's K = 10 * num_classes
# Number setting: (label, default, minimum, maximum). The default follows the dataset's classes.
WORDS = ("Visual words (K)", WORDS_PER_CLASS * 3, 2, 500)
# Images are shrunk to this longest side before extracting (Caltech 101's already are), so any photo gets keypoints
# at the scale of the training images, and SIFT on a large photo stays fast.
MAX_SIDE = 300
PATCH = 48  # side of the stored grey patch that pictures each visual word
SAMPLES = 10  # training images shown with their keypoints, as on the slide
COLUMNS = 5
TILE_HEIGHT = 120
WORD_TILE = 64
GAP = 12
MARKER = (255, 0, 0)
PREVIEW_SIZE = (720, 380)
PREVIEW_DELAY_MS = 300


@dataclass(frozen=True)
class Vocabulary:
    """The visual words and the training histograms, with how the descriptors were extracted."""

    method: str
    parameters: dict[str, float]  # the extractor's keyword arguments
    centroids: np.ndarray  # (K, d) float32; cluster centre k is visual word k
    patches: np.ndarray  # (K, PATCH, PATCH) uint8; the training patch whose descriptor is closest to each centre
    classes: list[str]
    train: list[tuple[str, str]]  # (path relative to the dataset folder, class), one per row of ``histograms``
    histograms: np.ndarray  # (n_train, K) normalized histogram of visual words of every training image
    test: list[tuple[str, str]]  # the images kept aside, for 7.2's visualization and 7.3's evaluation

    @property
    def words(self) -> int:
        return len(self.centroids)


@dataclass(frozen=True)
class Example:
    """A training image as it was described, with its keypoints."""

    grey: Image.Image
    keypoints: Keypoints


def list_images(folder: Path) -> list[tuple[Path, str]]:
    """Every image in ``folder/<class>/``, with its class; raises ``ValueError`` with fewer than two classes."""
    subfolders = sorted((path for path in folder.iterdir() if path.is_dir() and path.name != VOCABULARY_FOLDER),
                        key=lambda path: path.name.lower()) if folder.is_dir() else []
    images = [(path, subfolder.name) for subfolder in subfolders for path in sorted(subfolder.iterdir())
              if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()]
    if len({label for _, label in images}) < 2:
        raise ValueError(f"{folder} needs a subfolder of images per class, and at least two classes.")
    return images


def split_dataset(folder: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """The images of ``folder`` split 75/25 keeping each class's share (seed 42, as for the eigenfaces)."""
    from sklearn.model_selection import train_test_split  # slow to import, and only needed here

    images = [(path.relative_to(folder).as_posix(), label) for path, label in list_images(folder)]
    train, test = train_test_split(images, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                   stratify=[label for _, label in images])
    return sorted(train), sorted(test)


def prepared(image: Image.Image) -> Image.Image:
    """Greyscale, shrunk so the longest side is at most ``MAX_SIDE``: what the extractor sees."""
    grey = image.convert("L")
    factor = max(grey.size) / MAX_SIDE
    if factor > 1:
        grey = grey.resize((max(1, round(grey.width / factor)), max(1, round(grey.height / factor))),
                           Image.Resampling.LANCZOS)
    return grey


def assign_words(descriptors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """The visual word of each descriptor: the index of its closest centre (Euclidean, like ``kmeans.predict``)."""
    if len(descriptors) == 0:
        return np.empty(0, dtype=int)
    values = np.asarray(descriptors, dtype=np.float32)
    squared = (values ** 2).sum(axis=1)[:, None] - 2 * values @ centroids.T + (centroids ** 2).sum(axis=1)[None, :]
    return squared.argmin(axis=1)


def word_histogram(words: np.ndarray, count: int) -> np.ndarray:
    """The K-dimensional frequency histogram, normalized to sum 1 (all zeros when there are no keypoints)."""
    histogram = np.bincount(words, minlength=count).astype(np.float64)
    return histogram / histogram.sum() if histogram.sum() else histogram


def _patch(grey: Image.Image, point: np.ndarray, size: float) -> np.ndarray:
    """The square of the image a keypoint describes, ``PATCH`` px across; edges are repeated outside the image."""
    half = max(2 * float(size), 8.0)
    row, col = point
    pixels = np.asarray(grey)
    top, left = int(round(row - half)), int(round(col - half))
    side = max(1, int(round(2 * half)))
    rows = np.clip(np.arange(top, top + side), 0, pixels.shape[0] - 1)
    cols = np.clip(np.arange(left, left + side), 0, pixels.shape[1] - 1)
    crop = Image.fromarray(pixels[np.ix_(rows, cols)])
    return np.asarray(crop.resize((PATCH, PATCH), Image.Resampling.LANCZOS))


def build_vocabulary(folder: Path, method: str, parameters: dict[str, float], words: int,
                     progress: Callable[[int, int], None] | None = None) -> tuple[Vocabulary, list[Example]]:
    """7.1: extract descriptors from the training images, cluster them with K-Means and keep the centres.

    Also builds every training image's histogram (7.2), so 7.3 can train a classifier on them, and returns
    ``SAMPLES`` training images with their keypoints to show. ``progress(done, total)`` is called per image.
    """
    from sklearn.cluster import MiniBatchKMeans  # slow to import, and only needed here

    train, test = split_dataset(folder)
    greys: list[Image.Image] = []
    found: list[Keypoints] = []
    for index, (path, _) in enumerate(train):
        with Image.open(folder / path) as image_file:
            grey = prepared(image_file)
        greys.append(grey)
        found.append(extract_keypoints(grey, method, **parameters))
        if progress:
            progress(index + 1, len(train))
    descriptors = np.vstack([keypoints.descriptors.astype(np.float32) for keypoints in found if len(keypoints)]) \
        if any(len(keypoints) for keypoints in found) else np.empty((0, 1), dtype=np.float32)
    if len(descriptors) < words:
        raise ValueError(f"The training images gave {len(descriptors)} descriptors, fewer than K = {words}.")
    # As on the slide: every training descriptor stacked, and K-Means with a fixed seed.
    kmeans = MiniBatchKMeans(n_clusters=words, random_state=0).fit(descriptors)
    centroids = kmeans.cluster_centers_.astype(np.float32)

    # A centre is an average descriptor, not an image: picture each word by the training patch closest to it.
    owners = np.concatenate([np.full(len(keypoints), index) for index, keypoints in enumerate(found)])
    rows = np.concatenate([np.arange(len(keypoints)) for keypoints in found])
    closest = assign_words(centroids, descriptors)  # for each centre, the nearest descriptor
    patches = np.stack([_patch(greys[owners[i]], found[owners[i]].points[rows[i]], found[owners[i]].sizes[rows[i]])
                        for i in closest])

    histograms = np.stack([word_histogram(assign_words(keypoints.descriptors, centroids), words) for keypoints in found])
    shown = np.random.default_rng(0).choice(len(train), size=min(SAMPLES, len(train)), replace=False)
    vocabulary = Vocabulary(method, dict(parameters), centroids, patches, sorted({label for _, label in train}),
                            train, histograms, test)
    return vocabulary, [Example(greys[i], found[i]) for i in shown]


def save_vocabulary(vocabulary: Vocabulary, folder: Path) -> Path:
    """Write ``folder/bovw/`` (.npy arrays plus ``vocabulary.json``); return that folder."""
    target = folder / VOCABULARY_FOLDER
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "centroids.npy", vocabulary.centroids)
    np.save(target / "word_patches.npy", vocabulary.patches)
    np.save(target / "train_histograms.npy", vocabulary.histograms)
    description = {
        "method": vocabulary.method,
        "parameters": vocabulary.parameters,
        "max_side": MAX_SIDE,
        "words": vocabulary.words,
        "classes": vocabulary.classes,
        "train": [{"path": path, "class": label} for path, label in vocabulary.train],  # row i of train_histograms
        "test": [{"path": path, "class": label} for path, label in vocabulary.test],
    }
    (target / "vocabulary.json").write_text(json.dumps(description, indent=1), encoding="utf-8")
    return target


def load_vocabulary(folder: Path) -> Vocabulary:
    """The vocabulary ``save_vocabulary`` wrote to ``folder/bovw/``; ``ValueError`` if there is none."""
    source = folder / VOCABULARY_FOLDER
    if not (source / "vocabulary.json").is_file():
        raise ValueError(f"No visual vocabulary in {folder}. Build it first with Visual vocabulary.")
    description = json.loads((source / "vocabulary.json").read_text(encoding="utf-8"))
    pairs = lambda rows: [(row["path"], row["class"]) for row in rows]  # noqa: E731
    return Vocabulary(description["method"], description["parameters"], np.load(source / "centroids.npy"),
                      np.load(source / "word_patches.npy"), description["classes"], pairs(description["train"]),
                      np.load(source / "train_histograms.npy"), pairs(description["test"]))


def describe(image: Image.Image, vocabulary: Vocabulary) -> tuple[Image.Image, Keypoints, np.ndarray, np.ndarray]:
    """7.2 for one image: its keypoints, the visual word of each one and its normalized histogram."""
    grey = prepared(image)
    keypoints = extract_keypoints(grey, vocabulary.method, **vocabulary.parameters)
    words = assign_words(keypoints.descriptors, vocabulary.centroids)
    return grey, keypoints, words, word_histogram(words, vocabulary.words)


def _titled(images: list[Image.Image], titles: list[str], columns: int, font: ImageFont.ImageFont,
            heading: str) -> Image.Image:
    """``images`` (already at their tile size) in rows of ``columns``, each titled above, under ``heading``."""
    line = round(font.size * 1.5)
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images) + line
    rows = -(-len(images) // columns)
    result = Image.new("RGB", (columns * (cell_width + GAP) + GAP, line + rows * (cell_height + GAP) + GAP),
                       (255, 255, 255))
    draw = ImageDraw.Draw(result)
    draw.text((GAP, line / 2), heading, fill=(0, 0, 0), font=font, anchor="lm")
    for index, (image, title) in enumerate(zip(images, titles)):
        row, column = divmod(index, columns)
        left = GAP + column * (cell_width + GAP)
        top = line + GAP + row * (cell_height + GAP)
        draw.text((left + cell_width / 2, top + line / 2), title, fill=(0, 0, 0), font=font, anchor="mm")
        corner = (round(left + (cell_width - image.width) / 2), top + line)
        result.paste(image, corner)
        draw.rectangle((corner[0] - 1, corner[1] - 1, corner[0] + image.width, corner[1] + image.height),
                       outline=(0, 0, 0))
    return result


def show_vocabulary(vocabulary: Vocabulary, examples: list[Example]) -> Image.Image:
    """Training images with their keypoints in red, and below, every visual word as the patch closest to it."""
    font = ImageFont.load_default(14)
    tiles = []
    for example in examples:
        marked = example.grey.convert("RGB")
        draw = ImageDraw.Draw(marked)
        radius = max(2, round(max(marked.size) / 120))
        for row, col in example.keypoints.points:
            draw.ellipse((col - radius, row - radius, col + radius, row + radius), fill=MARKER)
        tiles.append(marked.resize((max(1, round(marked.width * TILE_HEIGHT / marked.height)), TILE_HEIGHT),
                                   Image.Resampling.LANCZOS))
    keypoints = _titled(tiles, [f"{len(example.keypoints)} keypoints" for example in examples], COLUMNS, font,
                        f"Training images with their {vocabulary.method} keypoints")
    words = [Image.fromarray(patch).resize((WORD_TILE, WORD_TILE), Image.Resampling.NEAREST)
             for patch in vocabulary.patches]
    columns = max(COLUMNS, int(np.ceil(np.sqrt(vocabulary.words))))
    codebook = _titled(words, [f"vw {index}" for index in range(vocabulary.words)], columns, font,
                       f"Visual words (K = {vocabulary.words}): the training patch closest to each cluster centre")
    result = Image.new("RGB", (max(keypoints.width, codebook.width), keypoints.height + codebook.height),
                       (255, 255, 255))
    result.paste(keypoints, (0, 0))
    result.paste(codebook, (0, keypoints.height))
    return result


def show_histogram(grey: Image.Image, keypoints: Keypoints, words: np.ndarray, histogram: np.ndarray,
                   vocabulary: Vocabulary, title: str) -> Image.Image:
    """The image with each keypoint coloured by its visual word and a line to that word's bar, as on the slide."""
    count = vocabulary.words
    colours = colormaps["hsv"](np.linspace(0, 1, count, endpoint=False))
    figure = Figure(figsize=(12, 4.5), dpi=100)
    grid = figure.add_gridspec(2, 2, width_ratios=(1, 1.6), height_ratios=(1, 0.12), hspace=0.05, wspace=0.2,
                               left=0.02, right=0.98, top=0.9, bottom=0.13)
    picture = figure.add_subplot(grid[:, 0])
    bars = figure.add_subplot(grid[0, 1])
    strip = figure.add_subplot(grid[1, 1], sharex=bars)

    picture.imshow(grey, cmap="gray", vmin=0, vmax=255)
    if len(keypoints):
        picture.scatter(keypoints.points[:, 1], keypoints.points[:, 0], s=14, c=colours[words], edgecolors="black",
                        linewidths=0.4, zorder=3)
    picture.set_title(title)
    picture.set_axis_off()

    positions = np.arange(count)
    bars.bar(positions, histogram, width=0.8, color="#d0d0d0", edgecolor="black", linewidth=0.6)
    bars.set_title("Histogram of visual words")
    bars.set_ylabel("Normalized frequency")
    bars.set_xlim(-0.6, count - 0.4)
    bars.set_ylim(0, max(float(histogram.max()) * 1.1, 0.01))
    bars.tick_params(labelbottom=False)
    # Under each bar, the patch that pictures its word.
    strip.imshow(np.hstack(list(vocabulary.patches)), cmap="gray", vmin=0, vmax=255, aspect="auto",
                 extent=(-0.5, count - 0.5, 1, 0))
    strip.set_yticks([])
    strip.set_xlabel("Visual word index")

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    if len(keypoints):
        # Lines from each keypoint to the top of its word's bar, drawn in figure coordinates over both axes.
        to_figure = figure.transFigure.inverted()
        starts = to_figure.transform(picture.transData.transform(keypoints.points[:, ::-1]))
        ends = to_figure.transform(bars.transData.transform(np.column_stack((words, histogram[words]))))
        # Under the axes (made see-through), so the bars and labels stay on top as on the slide.
        for axes in (picture, bars):
            axes.set_facecolor("none")
        figure.add_artist(LineCollection(np.stack((starts, ends), axis=1), colors=colours[words], alpha=0.2,
                                         linewidths=0.5, transform=figure.transFigure, zorder=-1))
        canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


def histogram_result(image: Image.Image, path: Path | None, folder: Path,
                     vocabulary: Vocabulary) -> tuple[Image.Image, str, dict[str, object]]:
    """The 7.2 panel of ``image``, a one-line summary and the details; warns when it is a training image."""
    grey, keypoints, words, histogram = describe(image, vocabulary)
    relative = None
    if path is not None:
        try:
            relative = path.resolve().relative_to(folder.resolve()).as_posix()
        except ValueError:
            pass
    split = "training" if relative in {train for train, _ in vocabulary.train} else \
        "test" if relative in {test for test, _ in vocabulary.test} else None
    label = path.parent.name if split else (path.name if path else "image")
    panel = show_histogram(grey, keypoints, words, histogram, vocabulary, label)
    top = np.argsort(histogram, kind="stable")[::-1][:3]
    note = "" if split != "training" else " This is a training image: the slide shows test images."
    return panel, f"{len(keypoints)} keypoints in {np.count_nonzero(histogram)} of {vocabulary.words} words.{note}", {
        "Image split": f"{split} ({label})" if split else "Not in the dataset",
        "Size described": f"{grey.width} × {grey.height} px (greyscale, longest side ≤ {MAX_SIDE})",
        "Keypoints": len(keypoints),
        "Words used": f"{np.count_nonzero(histogram)} of {vocabulary.words}",
        "Most frequent words": ", ".join(f"vw {word} ({histogram[word]:.2f})" for word in top if histogram[word]),
    }


def ask_vocabulary(parent: tk.Misc) -> tuple[Path, Vocabulary, list[Example]] | None:
    """Let the user pick the dataset, extractor and K, then build the vocabulary; ``None`` if cancelled."""
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Visual vocabulary")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()

    ttk.Label(body, text="Dataset folder (a subfolder of images per class)").pack(anchor="w")
    folder_row = ttk.Frame(body)
    folder_row.pack(fill="x", pady=(4, 0))
    folder = tk.StringVar(dialog, value=str(DEFAULT_DATASET))
    ttk.Entry(folder_row, textvariable=folder, width=56).pack(side="left", fill="x", expand=True)
    browse_button = ttk.Button(folder_row, text="Browse…", command=lambda: browse())
    browse_button.pack(side="left", padx=(6, 0))
    dataset_status = ttk.Label(body, foreground="#666666", wraplength=520, justify="left")
    dataset_status.pack(anchor="w", pady=(6, 0))

    method = choice_row(body, "Extractor", METHODS, lambda: method_frames[method.get()].tkraise(), default="SIFT")
    ttk.Label(body, text="Extractor parameters", padding=(0, 12, 0, 0)).pack(anchor="w")
    method_area = ttk.Frame(body)
    method_area.pack(fill="x")
    method_frames: dict[str, ttk.Frame] = {}
    method_values: dict[str, dict[str, tk.StringVar]] = {}
    for name, parameters in METHOD_PARAMETERS.items():
        frame = ttk.Frame(method_area)
        frame.grid(row=0, column=0, sticky="nsew")
        method_frames[name] = frame
        method_values[name] = setting_rows(frame, parameters, lambda: None, INTEGER_SETTINGS)
    upsampling = choice_row(method_frames["SIFT"], "Upsampling", UPSAMPLING, lambda: None, default="2")
    method_frames[method.get()].tkraise()

    ttk.Label(body, text="Codebook", padding=(0, 12, 0, 0)).pack(anchor="w")
    words = number_row(body, *WORDS, lambda: None)
    ttk.Label(body, text=f"K-Means (MiniBatchKMeans, seed 0) on the training descriptors; K is set to "
                         f"{WORDS_PER_CLASS} × the number of classes.\nImages are read in greyscale with the longest "
                         f"side at most {MAX_SIDE} px, and split 75/25 per class (seed 42).",
              foreground="#666666", justify="left").pack(anchor="w", pady=(6, 0))
    status = ttk.Label(body, foreground="#666666", wraplength=520)
    status.pack(anchor="w", pady=(12, 0))

    building = False
    poll_job: str | None = None
    result = []

    def folder_changed(*_: object) -> None:
        try:
            images = list_images(Path(folder.get()).expanduser())
        except (ValueError, OSError) as error:
            dataset_status.configure(text=str(error))
            return
        counts: dict[str, int] = {}
        for _, label in images:
            counts[label] = counts.get(label, 0) + 1
        dataset_status.configure(text=f"{len(images)} images of {len(counts)} classes: "
                                      + ", ".join(f"{label} ({count})" for label, count in counts.items()) + ".")
        words.set(str(min(WORDS[3], WORDS_PER_CLASS * len(counts))))

    def browse() -> None:
        chosen = filedialog.askdirectory(title="Choose the dataset folder", parent=dialog,
                                         initialdir=folder.get() if Path(folder.get()).is_dir() else None)
        if chosen:
            folder.set(chosen)

    def set_busy(busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for button in (browse_button, build_button, cancel_button):
            button.configure(state=state)
        dialog.configure(cursor="watch" if busy else "")

    def close() -> None:
        if building:  # the build runs on another thread; let it finish
            return
        dialog.destroy()

    def build(_event: tk.Event | None = None) -> None:
        nonlocal building
        if building:
            return
        name = method.get()
        try:
            chosen = Path(folder.get()).expanduser()
            list_images(chosen)
            parameters: dict[str, Any] = read_settings(method_values[name], METHOD_PARAMETERS[name], INTEGER_SETTINGS)
            if name == "SIFT":
                parameters["upsampling"] = int(upsampling.get())
            k = read_number(words, WORDS[0], WORDS[2], WORDS[3])
        except (ValueError, OSError) as error:
            messagebox.showerror("Visual vocabulary", str(error), parent=dialog)
            return
        building = True
        set_busy(True)
        status.configure(text=f"Extracting {name} descriptors…")
        outcome: dict[str, Any] = {"done": 0, "total": 0}

        def progress(done: int, total: int) -> None:  # called on the worker thread: no Tk here
            outcome["done"], outcome["total"] = done, total

        def work() -> None:
            try:
                outcome["built"] = build_vocabulary(chosen, name, parameters, k, progress)
            except Exception as error:  # report any failure in the dialog
                outcome["error"] = error

        thread = threading.Thread(target=work, daemon=True)
        thread.start()

        def poll() -> None:
            nonlocal building, poll_job
            if thread.is_alive():
                if outcome["total"] and outcome["done"] < outcome["total"]:
                    status.configure(text=f"Extracting {name} descriptors: {outcome['done']} of "
                                          f"{outcome['total']} training images…")
                elif outcome["total"]:
                    status.configure(text=f"Clustering the descriptors into K = {k} visual words…")
                poll_job = dialog.after(200, poll)
                return
            poll_job = None
            building = False
            set_busy(False)
            if "error" in outcome:
                status.configure(text=f"Could not build the vocabulary: {outcome['error']}")
                return
            result.append((chosen, *outcome["built"]))
            close()

        poll()

    folder.trace_add("write", folder_changed)
    folder_changed()

    buttons = ttk.Frame(body)
    buttons.pack(fill="x", pady=(16, 0))
    cancel_button = ttk.Button(buttons, text="Cancel", command=close)
    cancel_button.pack(side="right")
    build_button = ttk.Button(buttons, text="Build and save", command=build, default="active")
    build_button.pack(side="right", padx=(0, 6))
    dialog.bind("<Return>", build)
    dialog.bind("<Escape>", lambda _event: close())
    dialog.protocol("WM_DELETE_WINDOW", close)

    center_over(dialog, parent.winfo_toplevel())
    dialog.deiconify()
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


def ask_histogram(parent: tk.Misc, image: Image.Image, path: Path | None) -> tuple[Path, Vocabulary] | None:
    """Let the user pick the dataset with a saved vocabulary while previewing the histogram; ``None`` if cancelled."""
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Histogram of visual words")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()

    ttk.Label(body, text="Dataset folder (with the vocabulary saved by Visual vocabulary)").pack(anchor="w")
    folder_row = ttk.Frame(body)
    folder_row.pack(fill="x", pady=(4, 0))
    default = path.parent.parent if path is not None and (path.parent.parent / VOCABULARY_FOLDER).is_dir() \
        else DEFAULT_DATASET
    folder = tk.StringVar(dialog, value=str(default))
    ttk.Entry(folder_row, textvariable=folder).pack(side="left", fill="x", expand=True)
    ttk.Button(folder_row, text="Browse…", command=lambda: browse()).pack(side="left", padx=(6, 0))
    ttk.Label(body, text="Each descriptor is assigned to its closest visual word; the histogram counts them and is "
                         "normalized to sum 1.", foreground="#666666").pack(anchor="w", pady=(6, 0))

    canvas = tk.Canvas(body, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1], background="#20242b", highlightthickness=0)
    canvas.pack(pady=(12, 2))
    status = ttk.Label(body, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    preview_job: str | None = None
    cache: tuple[Path, Vocabulary] | None = None
    photo: ImageTk.PhotoImage | None = None

    def compute() -> tuple[Path, Vocabulary]:
        nonlocal cache
        chosen = Path(folder.get()).expanduser()
        if cache is None or cache[0] != chosen:
            cache = (chosen, load_vocabulary(chosen))
        return cache

    def schedule_preview(*_: object) -> None:
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
            chosen, vocabulary = compute()
            panel, summary, _ = histogram_result(image, path, chosen, vocabulary)
        except (ValueError, OSError) as error:
            status.configure(text=str(error))
            return
        photo = ImageTk.PhotoImage(_fit(panel, PREVIEW_SIZE), master=dialog)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        status.configure(text=f"{summary} {vocabulary.method}, K = {vocabulary.words}.")

    def browse() -> None:
        chosen = filedialog.askdirectory(title="Choose the dataset folder", parent=dialog,
                                         initialdir=folder.get() if Path(folder.get()).is_dir() else None)
        if chosen:
            folder.set(chosen)

    folder.trace_add("write", schedule_preview)
    result = []

    def close() -> None:
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        try:
            result.append(compute())
        except (ValueError, OSError) as error:
            messagebox.showerror("Histogram of visual words", str(error), parent=dialog)
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


class VisualVocabularyTool(ForensicsTool):
    tool_id = "visual_vocabulary"
    title = "Visual vocabulary"
    category = "Bag of visual words"
    description = "Cluster the training descriptors with K-Means into visual words, save them and show them."
    requires_image = False
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        choice = ask_vocabulary(parent)
        if choice is None:
            return None
        folder, vocabulary, examples = choice
        target = save_vocabulary(vocabulary, folder)
        gallery = target / "vocabulary.png"  # opened as the document, so it shows even with no image loaded
        show_vocabulary(vocabulary, examples).save(gallery)
        return ToolResult(
            message=f"Built {vocabulary.words} visual words and saved them to {target}.",
            details={
                "Operation": "Visual vocabulary (K-Means)",
                "Dataset": str(folder),
                "Classes": ", ".join(vocabulary.classes),
                "Images": f"{len(vocabulary.train)} training, {len(vocabulary.test)} test",
                "Extractor": vocabulary.method,
                **{label: f"{vocabulary.parameters[name]:g}"
                   for name, (label, *_) in METHOD_PARAMETERS[vocabulary.method].items()},
                **({"Upsampling": vocabulary.parameters["upsampling"]} if vocabulary.method == "SIFT" else {}),
                "Descriptor": DESCRIPTOR_SIZES[vocabulary.method],
                "Visual words (K)": vocabulary.words,
                "Model saved to": str(target),
                "Files": "centroids.npy, word_patches.npy, train_histograms.npy, vocabulary.json",
            },
            document_path=gallery,
        )


class WordHistogramTool(ForensicsTool):
    tool_id = "word_histogram"
    title = "Histogram of visual words"
    category = "Bag of visual words"
    description = "Assign the opened image's descriptors to the saved visual words and show their histogram."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.original is not None
        # The image as opened: after the first run the current image is the result panel.
        choice = ask_histogram(parent, document.original, document.path)
        if choice is None:
            return None
        folder, vocabulary = choice
        panel, summary, details = histogram_result(document.original, document.path, folder, vocabulary)
        return ToolResult(
            image=panel,
            message=summary,
            details={
                "Operation": "Histogram of visual words",
                "Vocabulary": str(folder / VOCABULARY_FOLDER),
                "Extractor": vocabulary.method,
                "Visual words (K)": vocabulary.words,
                **details,
            },
        )
