"""Bag of visual words classification (Set 7.3): a classifier trained on the saved training histograms names the
opened image's class from its histogram of visual words, with the top probability as on the slide."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from forensics_app.core import ImageDocument
from .bag_of_words import DEFAULT_DATASET, DRAWN, GREY, MAX_SIDE, VOCABULARY_FOLDER, Described, Vocabulary, describe, \
    load_vocabulary, show_histograms
from .base import ForensicsTool, ToolResult
from .corner_detection import read_settings, setting_rows
from .dialogs import center_over
from .eigenfaces import _fit
from .hog import choice_row


# SVM is the most accurate on the Caltech subset (about 77% with SIFT); its probabilities come from Platt scaling.
SVM, LOGISTIC, FOREST = CLASSIFIERS = ("SVM", "Logistic regression", "Random forest")
# χ² and histogram intersection compare histograms bin by bin, the usual kernels for bags of words.
CHI2, INTERSECTION, RBF, LINEAR = KERNELS = ("χ²", "Intersection", "RBF", "Linear")
# Number setting: (label, default, minimum, maximum). Defaults are scikit-learn's, except the forest's trees.
CLASSIFIER_PARAMETERS: dict[str, dict[str, tuple[str, float, float, float]]] = {
    SVM: {
        "C": ("C (regularization)", 1.0, 0.01, 1000.0),  # larger fits the training images more closely
        "gamma": ("Gamma (0 = auto)", 0.0, 0.0, 100.0),  # χ² and RBF only: how fast similarity falls with distance
    },
    LOGISTIC: {"C": ("C (regularization)", 1.0, 0.01, 1000.0)},
    FOREST: {
        "n_estimators": ("Trees", 300, 10, 1000),
        "max_depth": ("Max depth (0 = none)", 0, 0, 50),
    },
}
INTEGER_PARAMETERS = {"n_estimators", "max_depth"}
CARD_HEIGHT = 300  # height of the classified image under its probability
GAP = 12
PREVIEW_SIZE = (760, 480)  # tall enough for the classified image above its histogram
PREVIEW_DELAY_MS = 300


@dataclass(frozen=True)
class Prediction:
    """The probability of every class for one image."""

    classes: list[str]
    probabilities: np.ndarray  # (classes,) summing to 1

    @property
    def label(self) -> str:
        return self.classes[int(self.probabilities.argmax())]

    @property
    def probability(self) -> float:
        return float(self.probabilities.max())

    def ranked(self) -> list[tuple[str, float]]:
        """Every class with its probability, most probable first."""
        order = np.argsort(self.probabilities, kind="stable")[::-1]
        return [(self.classes[index], float(self.probabilities[index])) for index in order]


def chi2_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """χ²(x, y) = Σ_k (x_k − y_k)² / (x_k + y_k) between each row of ``first`` and of ``second`` (0/0 is 0)."""
    difference = first[:, None, :] - second[None, :, :]
    total = first[:, None, :] + second[None, :, :]
    return np.divide(difference ** 2, total, out=np.zeros_like(total), where=total > 0).sum(axis=2)


def intersection_kernel(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Histogram intersection K(x, y) = Σ_k min(x_k, y_k): the share two normalized histograms have in common."""
    return np.minimum(first[:, None, :], second[None, :, :]).sum(axis=2)


def default_parameters(name: str) -> dict[str, Any]:
    """The classifier's settings at their defaults; the SVM's also has its kernel."""
    parameters: dict[str, Any] = {key: spec[1] for key, spec in CLASSIFIER_PARAMETERS[name].items()}
    return {"kernel": CHI2, **parameters} if name == SVM else parameters


def resolved_gamma(kernel: str, gamma: float, histograms: np.ndarray | None = None) -> float | str:
    """The gamma the SVM uses: the one given, or with 0 (auto) 1 / the mean χ² distance between the training
    histograms for χ² (Zhang et al., 2007; 1 without them) and scikit-learn's ``"scale"`` for RBF."""
    if gamma > 0:
        return gamma
    if kernel == CHI2:
        mean = float(chi2_distances(histograms, histograms).mean()) if histograms is not None else 0.0
        return 1 / mean if mean > 0 else 1.0
    return "scale"


def make_classifier(name: str = SVM, parameters: dict[str, Any] | None = None,
                    histograms: np.ndarray | None = None) -> Any:
    """An untrained scikit-learn classifier with ``predict_proba``; fixed seeds, so results repeat.

    ``parameters`` override ``default_parameters(name)``; ``histograms`` (the training ones) set χ²'s auto gamma.
    """
    # Slow to import, and only needed here.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {name}")
    settings = {**default_parameters(name), **(parameters or {})}
    if name == SVM:
        kernel = settings["kernel"]
        common = {"C": settings["C"], "probability": True, "random_state": 0}
        if kernel == CHI2:
            gamma = resolved_gamma(CHI2, settings["gamma"], histograms)
            return SVC(kernel=lambda first, second: np.exp(-gamma * chi2_distances(first, second)), **common)
        if kernel == INTERSECTION:
            return SVC(kernel=intersection_kernel, **common)
        if kernel == RBF:
            return SVC(kernel="rbf", gamma=resolved_gamma(RBF, settings["gamma"]), **common)
        if kernel == LINEAR:
            return SVC(kernel="linear", **common)
        raise ValueError(f"Unknown kernel: {kernel}")
    if name == LOGISTIC:  # the frequencies are small, so standardize them or every class gets about 1/n
        return make_pipeline(StandardScaler(), LogisticRegression(C=settings["C"], max_iter=5000))
    return RandomForestClassifier(n_estimators=int(settings["n_estimators"]),
                                  max_depth=int(settings["max_depth"]) or None, random_state=0)


def train_classifier(vocabulary: Vocabulary, name: str = SVM, parameters: dict[str, Any] | None = None) -> Any:
    """The classifier trained on every training image's histogram of visual words and its class."""
    classifier = make_classifier(name, parameters, vocabulary.histograms)
    return classifier.fit(vocabulary.histograms, [label for _, label in vocabulary.train])


def describe_parameters(name: str, parameters: dict[str, Any], vocabulary: Vocabulary) -> str:
    """The settings in words, with the gamma actually used when it was left on auto."""
    settings = {**default_parameters(name), **parameters}
    if name == SVM:
        kernel = settings["kernel"]
        text = f"{kernel} kernel, C = {settings['C']:g}"
        if kernel in (CHI2, RBF):
            gamma = resolved_gamma(kernel, settings["gamma"], vocabulary.histograms)
            auto = " (auto)" if not settings["gamma"] else ""
            text += f", gamma = {gamma:.3g}{auto}" if isinstance(gamma, float) else f", gamma = {gamma}{auto}"
        return text
    if name == LOGISTIC:
        return f"C = {settings['C']:g}, standardized histograms"
    depth = int(settings["max_depth"])
    return f"{int(settings['n_estimators'])} trees, max depth {depth if depth else 'none'}"


def classify(histogram: np.ndarray, classifier: Any) -> Prediction:
    """Run the trained classifier on one histogram; ``ValueError`` when the image has no visual words."""
    if not histogram.sum():
        raise ValueError("No keypoints were found in this image, so it has no visual words to classify.")
    probabilities = classifier.predict_proba(histogram[None, :].astype(np.float64))[0]
    return Prediction([str(label) for label in classifier.classes_], probabilities)


def true_class(path: Path | None, folder: Path, vocabulary: Vocabulary) -> tuple[str, str] | None:
    """The split (``"training"`` or ``"test"``) and class of an image of the dataset; else ``None``."""
    if path is None:
        return None
    try:
        relative = path.resolve().relative_to(folder.resolve()).as_posix()
    except ValueError:
        return None
    for split, images in (("training", vocabulary.train), ("test", vocabulary.test)):
        for image_path, label in images:
            if image_path == relative:
                return split, label
    return None


def show_classification(image: Image.Image, prediction: Prediction, truth: str | None = None) -> Image.Image:
    """The image in colour, framed, under its most probable class: "p(kangaroo) = 0.64" as on the slide."""
    font = ImageFont.load_default(22)
    small = ImageFont.load_default(14)
    tile = image.convert("RGB")
    tile = tile.resize((max(1, round(tile.width * CARD_HEIGHT / tile.height)), CARD_HEIGHT), Image.Resampling.LANCZOS)
    title = f"p({prediction.label}) = {prediction.probability:.2f}"
    subtitle = f"true: {truth}" if truth else ""
    line, small_line = round(font.size * 1.5), round(small.size * 1.5) if subtitle else 0
    width = max(tile.width, round(font.getlength(title)), round(small.getlength(subtitle))) + 2 * GAP
    result = Image.new("RGB", (width, GAP + line + small_line + tile.height + 2 * GAP), (255, 255, 255))
    draw = ImageDraw.Draw(result)
    draw.text((width / 2, GAP + line / 2), title, fill=(0, 0, 0), font=font, anchor="mm")
    if subtitle:
        correct = truth == prediction.label
        draw.text((width / 2, GAP + line + small_line / 2), subtitle, font=small, anchor="mm",
                  fill=(0, 128, 0) if correct else (192, 0, 0))
    corner = (round((width - tile.width) / 2), GAP + line + small_line)
    result.paste(tile, corner)
    draw.rectangle((corner[0] - 3, corner[1] - 3, corner[0] + tile.width + 2, corner[1] + tile.height + 2),
                   outline=(140, 110, 60), width=3)  # the slide's frame
    return result


def _stacked(top: Image.Image, bottom: Image.Image) -> Image.Image:
    """``top`` centred above ``bottom`` on white."""
    result = Image.new("RGB", (max(top.width, bottom.width), top.height + bottom.height), (255, 255, 255))
    result.paste(top, ((result.width - top.width) // 2, 0))
    result.paste(bottom, ((result.width - bottom.width) // 2, top.height))
    return result


def classification_result(image: Image.Image, row: Described, vocabulary: Vocabulary, classifier: Any | None,
                          truth: tuple[str, str] | None = None) -> tuple[Image.Image, str, dict[str, object]]:
    """The 7.2 histogram of the image and, once a classifier is given, the image under its top probability above
    it: the panel, a one-line summary and the details."""
    histogram = show_histograms([row], vocabulary, DRAWN[1], GREY)
    split = f"{truth[0]} image of {truth[1]}" if truth else "Not in the dataset"
    details: dict[str, object] = {
        "Image split": split,
        "Keypoints": f"{len(row.keypoints)} in {np.count_nonzero(row.histogram)} of {vocabulary.words} words",
    }
    if classifier is None:
        return histogram, f"{len(row.keypoints)} keypoints. Click Classify to run the classifier.", details
    prediction = classify(row.histogram, classifier)
    known = truth[1] if truth else None
    summary = f"p({prediction.label}) = {prediction.probability:.2f}."
    if known:
        summary += " Correct." if known == prediction.label else f" Wrong: it is {known}."
    if truth and truth[0] == "training":
        summary += " This is a training image, so the classifier has already seen it."
    return _stacked(show_classification(image, prediction, known), histogram), summary, {
        "Predicted": prediction.label,
        "Top probability": f"{prediction.probability:.2f}",
        **details,
        **{f"p({label})": f"{probability:.2f}" for label, probability in prediction.ranked()},
    }


Choice = tuple[Path, Vocabulary, str, dict[str, Any], Any, Described, tuple[str, str] | None]


def ask_classification(parent: tk.Misc, image: Image.Image, path: Path | None) -> Choice | None:
    """Show the opened image's histogram and classify it on demand; ``None`` if cancelled.

    Returns the dataset folder, the vocabulary, the classifier's name and settings, the trained classifier (``None``
    if not classified), the described image and its split and class in the dataset (if it is one of its images).
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Classify image")
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
    method = choice_row(body, "Classifier", CLASSIFIERS, lambda: method_changed())
    ttk.Label(body, text="Classifier parameters", padding=(0, 12, 0, 0)).pack(anchor="w")
    method_area = ttk.Frame(body)
    method_area.pack(fill="x")
    method_frames: dict[str, ttk.Frame] = {}
    method_values: dict[str, dict[str, tk.StringVar]] = {}
    for name, parameters in CLASSIFIER_PARAMETERS.items():
        frame = ttk.Frame(method_area)
        frame.grid(row=0, column=0, sticky="nsew")
        method_frames[name] = frame
        if name == SVM:
            kernel = choice_row(frame, "Kernel", KERNELS, lambda: parameters_changed())
        method_values[name] = setting_rows(frame, parameters, lambda: parameters_changed(), INTEGER_PARAMETERS)
    method_frames[method.get()].tkraise()
    ttk.Label(body, text="Trained on the saved histograms of the training images, then run on this image's "
                         "histogram. Gamma 0 is auto: 1 / the mean χ² distance\nbetween the training histograms for "
                         "χ², scikit-learn's \"scale\" for RBF. Classify shows the most probable class; Apply keeps "
                         "the preview.", foreground="#666666", justify="left").pack(anchor="w", pady=(6, 0))

    canvas = tk.Canvas(body, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1], background="#20242b", highlightthickness=0)
    canvas.pack(pady=(12, 2))
    status = ttk.Label(body, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    preview_job: str | None = None
    vocabulary_cache: tuple[Path, Vocabulary, Described] | None = None
    classifier_cache: dict[tuple[Any, ...], Any] = {}  # by classifier name and settings, for the cached vocabulary
    classified = False  # the slide's flow: the histogram first, the probability after clicking Classify
    photo: ImageTk.PhotoImage | None = None

    def compute(trained: bool) -> Choice:
        nonlocal vocabulary_cache
        chosen = Path(folder.get()).expanduser()
        if vocabulary_cache is None or vocabulary_cache[0] != chosen:
            vocabulary = load_vocabulary(chosen)
            vocabulary_cache = (chosen, vocabulary, describe(image, vocabulary, path.name if path else "image"))
            classifier_cache.clear()
        _, vocabulary, row = vocabulary_cache
        name = method.get()
        settings: dict[str, Any] = read_settings(method_values[name], CLASSIFIER_PARAMETERS[name], INTEGER_PARAMETERS)
        if name == SVM:
            settings["kernel"] = kernel.get()
        key = (name, *sorted(settings.items()))
        if trained and key not in classifier_cache:
            classifier_cache[key] = train_classifier(vocabulary, name, settings)
        return (chosen, vocabulary, name, settings, classifier_cache.get(key), row,
                true_class(path, chosen, vocabulary))

    def method_changed() -> None:
        method_frames[method.get()].tkraise()
        schedule_preview()

    def parameters_changed() -> None:
        if classified:  # retrain and classify again; before Classify the histogram does not change
            schedule_preview()

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
            chosen, vocabulary, name, settings, classifier, row, truth = compute(classified)
            panel, summary, _ = classification_result(image, row, vocabulary, classifier, truth)
        except (ValueError, OSError) as error:
            status.configure(text=str(error))
            return
        photo = ImageTk.PhotoImage(_fit(panel, PREVIEW_SIZE), master=dialog)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        trained = f" {name} ({describe_parameters(name, settings, vocabulary)}) trained on " \
            f"{len(vocabulary.train)} images." if classified else ""
        status.configure(text=f"{summary}{trained} {vocabulary.method}, K = {vocabulary.words}.")

    def browse() -> None:
        chosen = filedialog.askdirectory(title="Choose the dataset folder", parent=dialog,
                                         initialdir=folder.get() if Path(folder.get()).is_dir() else None)
        if chosen:
            folder.set(chosen)

    def folder_changed(*_: object) -> None:
        nonlocal classified
        classified = False  # a new vocabulary: show its histogram before classifying again
        schedule_preview()

    folder.trace_add("write", folder_changed)
    result = []

    def close() -> None:
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        dialog.destroy()

    def classify_clicked(_event: tk.Event | None = None) -> None:
        nonlocal classified
        classified = True
        schedule_preview()

    def ok(_event: tk.Event | None = None) -> None:
        try:
            choice = compute(classified)
            if classified:
                classify(choice[5].histogram, choice[4])  # refuse an image with no visual words here, not later
        except (ValueError, OSError) as error:
            messagebox.showerror("Classify image", str(error), parent=dialog)
            return
        result.append(choice)
        close()

    buttons = ttk.Frame(body)
    buttons.pack(fill="x", pady=(16, 0))
    ttk.Button(buttons, text="Cancel", command=close).pack(side="right")
    ttk.Button(buttons, text="Apply", command=ok).pack(side="right", padx=(0, 6))
    ttk.Button(buttons, text="Classify", command=classify_clicked, default="active").pack(side="right", padx=(0, 6))
    dialog.bind("<Return>", classify_clicked)
    dialog.bind("<Escape>", lambda _event: close())
    dialog.protocol("WM_DELETE_WINDOW", close)

    center_over(dialog, parent.winfo_toplevel())
    dialog.deiconify()
    schedule_preview()
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class ClassificationTool(ForensicsTool):
    tool_id = "bow_classification"
    title = "Classify image"
    category = "Bag of visual words"
    description = "Classify the opened image from its histogram of visual words and show the top probability."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.original is not None
        # The image as opened: after the first run the current image is the result panel.
        choice = ask_classification(parent, document.original, document.path)
        if choice is None:
            return None
        folder, vocabulary, name, settings, classifier, row, truth = choice
        panel, summary, details = classification_result(document.original, row, vocabulary, classifier, truth)
        trained = {
            "Classifier": f"{name}, trained on {len(vocabulary.train)} training histograms",
            "Classifier parameters": describe_parameters(name, settings, vocabulary),
        } if classifier is not None else {}
        return ToolResult(
            image=panel,
            message=summary,
            details={
                "Operation": "Bag of visual words classification" if classifier is not None
                else "Histogram of visual words (not classified)",
                "Vocabulary": str(folder / VOCABULARY_FOLDER),
                "Extractor": vocabulary.method,
                "Visual words (K)": vocabulary.words,
                **trained,
                "Classes": ", ".join(vocabulary.classes),
                "Size described": f"{row.grey.width} × {row.grey.height} px (greyscale, longest side ≤ {MAX_SIDE})",
                **details,
            },
        )
