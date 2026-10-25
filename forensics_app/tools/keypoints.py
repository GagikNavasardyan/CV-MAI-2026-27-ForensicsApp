"""Keypoints and descriptors (ORB or SIFT) with configurable extractors, drawn over the image."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .corner_detection import BASES, default_radius, read_settings, setting_rows
from .dialogs import center_over, preview_copy
from .hog import choice_row


METHODS = ("ORB", "SIFT")
# Number setting: (label, default, minimum, maximum). Defaults are skimage's, except ORB's
# n_keypoints, which follows the Set 5 example (skimage's is 500).
METHOD_PARAMETERS: dict[str, dict[str, tuple[str, float, float, float]]] = {
    "ORB": {
        "n_keypoints": ("Keypoints (max)", 200, 1, 5000),
        "n_scales": ("Pyramid levels", 8, 1, 16),
        "downscale": ("Downscale factor", 1.2, 1.05, 3.0),  # between pyramid levels
        "fast_n": ("FAST arc length", 9, 9, 16),
        "fast_threshold": ("FAST threshold", 0.08, 0.0, 1.0),
        "harris_k": ("Harris k", 0.04, 0.0, 0.25),  # ranks the FAST corners; the best n_keypoints are kept
    },
    "SIFT": {
        "n_octaves": ("Octaves", 8, 1, 10),  # skimage uses fewer when the image is too small for them
        "n_scales": ("Scales per octave", 3, 1, 10),
        "sigma_min": ("Sigma (min blur)", 1.6, 0.5, 5.0),
        "c_dog": ("Contrast threshold", 0.04 / 3, 0.0, 0.1),  # drops weak DoG extrema
        "c_edge": ("Edge threshold", 10.0, 1.0, 50.0),  # drops extrema lying on edges
    },
}
INTEGER_SETTINGS = {"n_keypoints", "n_scales", "fast_n", "n_octaves", "radius"}
UPSAMPLING = ("1", "2", "4")  # SIFT enlarges the image this much before the first octave
SIGMA_IN = 0.5  # the blur skimage's SIFT assumes the image already has
DESCRIPTOR_SIZES = {"ORB": "256 bits (binary)", "SIFT": "128 values"}
VIEWS = ("Keypoints", "Size and orientation")
MARKER_COLOUR = (255, 0, 0)
PREVIEW_SIZE = (480, 360)
PREVIEW_DELAY_MS = 400  # SIFT takes a while, so wait a little longer for a pause in typing or sliding


@dataclass(frozen=True)
class Keypoints:
    """What an extractor found; the arrays have one entry per keypoint."""

    method: str
    points: np.ndarray  # (N, 2) row, column in the image
    sizes: np.ndarray  # (N,) radius in px of the region the keypoint describes
    orientations: np.ndarray  # (N,) radians from the +x (column) axis towards +y (down the rows)
    descriptors: np.ndarray  # (N, 256) booleans for ORB, (N, 128) uint8 for SIFT

    def __len__(self) -> int:
        return len(self.points)

    def scaled(self, factor: float) -> Keypoints:
        """The keypoints of an image ``factor`` times larger; the descriptors are unchanged."""
        return Keypoints(self.method, self.points * factor, self.sizes * factor, self.orientations, self.descriptors)


def extract_keypoints(image: Image.Image, method: str = METHODS[0], **parameters: float) -> Keypoints:
    """Run ``skimage.feature.ORB`` or ``SIFT`` ``detect_and_extract`` on the grayscale image (scaled to 0–1).

    ``parameters`` are the extractor's own keyword arguments. Finding no keypoints is not an error:
    the result is then empty. Raises ``ValueError`` for a SIFT sigma too small for its upsampling.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    if method == "ORB":
        integers = {name: int(parameters[name]) for name in ("n_keypoints", "n_scales", "fast_n") if name in parameters}
        extractor = feature.ORB(**{**parameters, **integers})
        descriptor_length = 256
    elif method == "SIFT":
        # skimage divides sigma by the upsampling, then blurs by sqrt(sigma² − SIGMA_IN²), which fails below SIGMA_IN.
        upsampling, sigma = int(parameters.get("upsampling", 2)), parameters.get("sigma_min", 1.6)
        if sigma / upsampling < SIGMA_IN:
            raise ValueError(f"With ×{upsampling} upsampling, sigma must be at least {SIGMA_IN * upsampling:g} "
                             f"(skimage needs sigma ÷ upsampling ≥ {SIGMA_IN:g}).")
        integers = {name: int(parameters[name]) for name in ("upsampling", "n_octaves", "n_scales") if name in parameters}
        extractor = feature.SIFT(**{**parameters, **integers})
        descriptor_length = 128
    else:
        raise ValueError(f"Unknown method: {method}")
    try:
        extractor.detect_and_extract(gray)
    except RuntimeError:  # "... found no features"
        empty = np.empty(0)
        return Keypoints(method, np.empty((0, 2)), empty, empty, np.empty((0, descriptor_length)))
    if method == "ORB":
        # The oriented-FAST patch is 31 px across at its pyramid level; the angle is measured from the column axis.
        sizes = 15.5 * extractor.scales
        orientations = extractor.orientations
    else:
        # The Gaussian weighting the orientation histogram; skimage measures the angle from the row axis.
        sizes = extractor.lambda_ori * extractor.sigmas
        orientations = np.pi / 2 - extractor.orientations
    return Keypoints(method, extractor.keypoints.astype(np.float64), np.asarray(sizes, dtype=np.float64),
                     np.asarray(orientations, dtype=np.float64), extractor.descriptors)


def show_keypoints(
    image: Image.Image, keypoints: Keypoints, view: str = VIEWS[0], radius: int = 4, base: str = BASES[0]
) -> Image.Image:
    """Return the image with a red circle of ``radius`` on each keypoint, or ("Size and orientation")
    circles as large as each keypoint's region with a line showing its orientation."""
    if view not in VIEWS:
        raise ValueError(f"Unknown view: {view}")
    marked = image.convert("L" if base == BASES[0] else "RGB").convert("RGB")
    draw = ImageDraw.Draw(marked)
    width = max(1, (radius if view == VIEWS[0] else default_radius(image)) // 4)
    for (row, col), size, angle in zip(keypoints.points, keypoints.sizes, keypoints.orientations):
        r = radius if view == VIEWS[0] else max(float(size), 2.0)
        draw.ellipse((col - r, row - r, col + r, row + r), outline=MARKER_COLOUR, width=width)
        if view == VIEWS[1]:
            draw.line((col, row, col + r * np.cos(angle), row + r * np.sin(angle)), fill=MARKER_COLOUR, width=width)
    return marked


def ask_keypoints(
    parent: tk.Misc, image: Image.Image
) -> tuple[dict[str, Any], str, dict[str, Any], Keypoints, Image.Image] | None:
    """Let the user configure ORB or SIFT and the view while previewing on ``image``; ``None`` if cancelled.

    Returns the extractor parameters, the view, its style, the keypoints and the image to show.
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Keypoints and descriptors")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()
    controls = ttk.Frame(body)
    controls.grid(row=0, column=0, sticky="n")
    preview_side = ttk.Frame(body, padding=(20, 0, 0, 0))
    preview_side.grid(row=0, column=1, sticky="n")

    ttk.Label(preview_side, text="Preview").pack(anchor="w")
    canvas = tk.Canvas(preview_side, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1],
                       background="#20242b", highlightthickness=0)
    canvas.pack(pady=(4, 2))
    status = ttk.Label(preview_side, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    preview_job: str | None = None
    small, factor = preview_copy(image)  # a large image previews on a smaller copy; Apply uses the full one
    reduced = factor > 1
    keypoints_cache: dict[bool, tuple[tuple[Any, ...], Keypoints]] = {}  # by full size; extracting is slow, restyling is not
    preview: tuple[tuple[Any, ...], Image.Image] | None = None
    photo: ImageTk.PhotoImage | None = None

    def changed() -> None:
        schedule_preview()

    method = choice_row(controls, "Extractor", METHODS, changed)

    # Each extractor keeps its own settings in its own frame; the frames share one grid cell so the
    # window keeps its size, and the selected extractor's frame is raised on top.
    ttk.Label(controls, text="Extractor parameters", padding=(0, 16, 0, 0)).pack(anchor="w")
    method_area = ttk.Frame(controls)
    method_area.pack(fill="x")
    method_frames: dict[str, ttk.Frame] = {}
    method_values: dict[str, dict[str, tk.StringVar]] = {}
    for name, parameters in METHOD_PARAMETERS.items():
        frame = ttk.Frame(method_area)
        frame.grid(row=0, column=0, sticky="nsew")
        method_frames[name] = frame
        method_values[name] = setting_rows(frame, parameters, changed, INTEGER_SETTINGS)
    upsampling = choice_row(method_frames["SIFT"], "Upsampling", UPSAMPLING, changed, default="2")
    method_frames[METHODS[0]].tkraise()
    method.trace_add("write", lambda *_: method_frames[method.get()].tkraise())

    def reset() -> None:
        for name, parameters in METHOD_PARAMETERS.items():
            for setting, (_, default, _, _) in parameters.items():
                method_values[name][setting].set(f"{default:g}")
        upsampling.set("2")

    ttk.Button(controls, text="Reset to defaults", command=reset).pack(anchor="w", pady=(8, 0))

    ttk.Label(controls, text="Show", padding=(0, 16, 0, 0)).pack(anchor="w")
    view = tk.StringVar(dialog, value=VIEWS[0])
    for option in VIEWS:
        ttk.Radiobutton(controls, text=option, value=option, variable=view).pack(anchor="w", pady=(2, 0))
    view.trace_add("write", lambda *_: schedule_preview())
    style_parameters = {"radius": ("Circle radius (px)", default_radius(image), 1, 50)}
    style_values = setting_rows(controls, style_parameters, changed, INTEGER_SETTINGS)
    ttk.Label(controls, text="Used by the Keypoints view; the other sizes each circle by its keypoint.",
              foreground="#666666").pack(anchor="w", pady=(4, 0))
    base = choice_row(controls, "Draw on", BASES, changed)

    def compute(full: bool = not reduced) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any], Keypoints,
                                                   Image.Image]:
        """Extract on the full image, or else on the preview copy with the keypoints scaled back up."""
        name = method.get()
        current: dict[str, Any] = read_settings(method_values[name], METHOD_PARAMETERS[name], INTEGER_SETTINGS)
        if name == "SIFT":
            current["upsampling"] = int(upsampling.get())
        style = {**read_settings(style_values, style_parameters, INTEGER_SETTINGS), "base": base.get()}
        method_key = (name, *current.items())
        if full not in keypoints_cache or keypoints_cache[full][0] != method_key:
            found = extract_keypoints(image, name, **current) if full else \
                extract_keypoints(small, name, **current).scaled(factor)
            keypoints_cache[full] = (method_key, found)
        keypoints = keypoints_cache[full][1]
        key = (full, method_key, view.get(), *style.values())
        if preview is not None and preview[0] == key:  # nothing changed since the preview
            return key, current, style, keypoints, preview[1]
        return key, current, style, keypoints, show_keypoints(image, keypoints, view.get(), **style)

    def schedule_preview(*_: object) -> None:
        nonlocal preview_job
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        status.configure(text="Updating preview…")
        preview_job = dialog.after(PREVIEW_DELAY_MS, update_preview)

    def update_preview() -> None:
        nonlocal preview_job, preview, photo
        preview_job = None
        canvas.delete("all")
        try:
            key, _, _, keypoints, shown = compute()
        except ValueError as error:
            preview = photo = None
            status.configure(text=str(error))
            return
        preview = (key, shown)
        thumbnail = shown.copy()
        thumbnail.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(thumbnail, master=dialog)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        found = f"{len(keypoints)} keypoints, each with a descriptor of {DESCRIPTOR_SIZES[keypoints.method]}."
        if reduced:
            status.configure(text=f"{found} Previewed on a {small.width} × {small.height} copy for speed; "
                                  f"Apply runs on the full {image.width} × {image.height} image.")
        else:
            status.configure(text=f"{found} Apply keeps exactly this result.")

    result = []

    def close() -> None:
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        if reduced:
            status.configure(text=f"Running on the full {image.width} × {image.height} image…")
            dialog.configure(cursor="watch")
            dialog.update_idletasks()
        try:
            _, current, style, keypoints, shown = compute(full=True)  # reuses the preview when nothing changed
        except ValueError as error:
            dialog.configure(cursor="")
            messagebox.showerror("Keypoints and descriptors", str(error), parent=dialog)
            return
        result.append((current, view.get(), style, keypoints, shown))
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


class KeypointsTool(ForensicsTool):
    tool_id = "keypoints"
    title = "Keypoints (ORB, SIFT)"
    category = "Feature extraction"
    description = "ORB or SIFT keypoints and descriptors with configurable extractors."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        choice = ask_keypoints(parent, document.current)
        if choice is None:
            return None
        current, view, _, keypoints, shown = choice
        labels = {name: label for name, (label, *_) in METHOD_PARAMETERS[keypoints.method].items()}
        labels["upsampling"] = "Upsampling"
        return ToolResult(
            image=shown,
            message=f"Extracted {len(keypoints)} {keypoints.method} keypoints.",
            details={
                "Operation": f"{keypoints.method} keypoints and descriptors",
                **{labels[name]: f"{value:g}" for name, value in current.items()},
                "Keypoints": len(keypoints),
                "Descriptor": DESCRIPTOR_SIZES[keypoints.method],
                "Shown": view,
            },
        )
