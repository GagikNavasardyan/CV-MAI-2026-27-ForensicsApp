"""Corner detection (FAST or Harris) with an adjustable threshold for the peaks."""

from __future__ import annotations

from collections.abc import Callable
import re
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import center_over
from .hog import choice_row, number_row, read_number


DETECTORS = ("FAST", "Harris")
# Number setting: (label, default, minimum, maximum). The detector defaults are skimage's.
DETECTOR_PARAMETERS: dict[str, dict[str, tuple[str, float, float, float]]] = {
    "FAST": {
        "n": ("Arc length (px)", 12, 9, 16),  # contiguous circle pixels that must all be brighter or darker
        "threshold": ("Intensity threshold", 0.15, 0.0, 1.0),  # how much brighter or darker, on a 0–1 scale
    },
    "Harris": {
        "k": ("Sensitivity k", 0.05, 0.0, 0.25),
        "sigma": ("Sigma (smoothing)", 1.0, 0.1, 10.0),
    },
}
PEAK_PARAMETERS = {
    # Fraction of the strongest response. skimage's default, 0, also keeps the tiny peaks of rounding noise
    # in flat areas (thousands on a photo), so start at the customary 1 %.
    "threshold_rel": ("Peak threshold", 0.01, 0.0, 1.0),
    "min_distance": ("Min distance (px)", 1, 1, 50),
}
INTEGER_SETTINGS = {"n", "min_distance", "radius"}
BASES = ("Grayscale", "Colour image")  # what the corners are drawn on; detection always uses grayscale
VIEWS = ("Corners on image", "Response map", "Side by side")
MARKER_COLOUR = (255, 0, 0)
PREVIEW_SIZE = (480, 360)
PREVIEW_DELAY_MS = 250  # wait for a pause in typing or sliding before running the detector on the full image


def corner_response(image: Image.Image, detector: str = DETECTORS[0], **parameters: float) -> np.ndarray:
    """Return the corner measure of ``skimage.feature.corner_fast`` or ``corner_harris``.

    Both run on the grayscale image scaled to 0–1. FAST takes ``n`` and ``threshold``; Harris takes
    ``k`` and ``sigma``.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    if detector == "FAST":
        return feature.corner_fast(gray, n=int(parameters.get("n", 12)), threshold=parameters.get("threshold", 0.15))
    if detector == "Harris":
        # skimage pads with zeros, so the image frame would read as an edge and its four corners as
        # corners; repeating the border pixels instead keeps the response to what is in the image.
        sigma = parameters.get("sigma", 1.0)
        pad = int(np.ceil(4 * sigma)) + 2
        response = feature.corner_harris(np.pad(gray, pad, mode="edge"), method="k", k=parameters.get("k", 0.05),
                                         sigma=sigma)
        return response[pad:-pad, pad:-pad]
    raise ValueError(f"Unknown detector: {detector}")


def find_corners(response: np.ndarray, threshold_rel: float = 0.01, min_distance: int = 1) -> np.ndarray:
    """Return the (row, column) of each peak of ``response`` from ``skimage.feature.corner_peaks``.

    A peak is kept when it is the largest value within ``min_distance`` px and above ``threshold_rel``
    times the strongest response (and above the response's minimum, so a constant map has no corners).
    """
    return feature.corner_peaks(response, min_distance=min_distance, threshold_rel=threshold_rel)


def show_corners(
    image: Image.Image,
    response: np.ndarray,
    corners: np.ndarray,
    view: str = VIEWS[0],
    radius: int = 4,
    base: str = BASES[0],
) -> Image.Image:
    """Return red circles of ``radius`` around the corners, the colour-mapped response, or both side by side."""
    if view not in VIEWS:
        raise ValueError(f"Unknown view: {view}")
    response_map = _response_image(response)
    if view == "Response map":
        return response_map
    marked = image.convert("L" if base == BASES[0] else "RGB").convert("RGB")
    draw = ImageDraw.Draw(marked)
    width = max(1, radius // 4)
    for row, col in corners:
        draw.ellipse((col - radius, row - radius, col + radius, row + radius), outline=MARKER_COLOUR, width=width)
    if view == "Corners on image":
        return marked
    side = Image.new("RGB", (2 * image.width, image.height))
    side.paste(marked, (0, 0))
    side.paste(response_map, (image.width, 0))
    return side


def default_radius(image: Image.Image) -> int:
    """A circle size that stays visible once the image is fitted to the screen."""
    return max(3, round(max(image.size) / 200))


def _response_image(response: np.ndarray) -> Image.Image:
    """The response scaled to 0–1 and coloured with viridis, as ``plt.imshow`` shows it.

    Negative Harris values mark edges, not corners, so they show as 0 like flat areas.
    """
    response = np.clip(response, 0.0, None)
    peak = float(response.max())
    scaled = response / peak if peak > 0 else np.zeros_like(response, dtype=np.float64)
    rgb = colormaps["viridis"](scaled)[..., :3]
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8))


def _is_decimal_prefix(text: str) -> bool:
    return re.fullmatch(r"\d*\.?\d*", text) is not None


def decimal_row(parent: tk.Misc, label: str, default: float, minimum: float, maximum: float,
                on_change: Callable[[], None]) -> tk.StringVar:
    """A slider with a box for typing the exact decimal; either one updates the other."""
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=(6, 0))
    ttk.Label(row, text=label, width=17).pack(side="left")
    variable = tk.StringVar(parent, value=f"{default:g}")
    slider = ttk.Scale(row, from_=minimum, to=maximum, length=160, value=default)
    slider.pack(side="left", padx=8)
    only_decimals = (parent.register(_is_decimal_prefix), "%P")
    ttk.Entry(row, textvariable=variable, width=7, justify="center",
              validate="key", validatecommand=only_decimals).pack(side="left")
    syncing = False

    def slider_moved(value: str) -> None:
        if not syncing:
            variable.set(f"{float(value):.3f}".rstrip("0").rstrip(".") or "0")

    def text_changed(*_: object) -> None:
        nonlocal syncing
        try:
            number = float(variable.get())
        except ValueError:
            number = None
        if number is not None and minimum <= number <= maximum:
            syncing = True
            slider.set(number)
            syncing = False
        on_change()

    slider.configure(command=slider_moved)
    variable.trace_add("write", text_changed)
    return variable


def read_decimal(variable: tk.StringVar, label: str, minimum: float, maximum: float) -> float:
    """The typed number; raises ``ValueError`` with a message for the status line."""
    try:
        number = float(variable.get())
    except ValueError:
        raise ValueError(f"Enter a number for {label.lower()}.") from None
    if not minimum <= number <= maximum:
        raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}.")
    return number


def setting_rows(parent: tk.Misc, parameters: dict[str, tuple[str, float, float, float]],
                 on_change: Callable[[], None]) -> dict[str, tk.StringVar]:
    """One slider row per setting: whole numbers for ``INTEGER_SETTINGS``, decimals otherwise."""
    return {name: (number_row if name in INTEGER_SETTINGS else decimal_row)(parent, *spec, on_change)
            for name, spec in parameters.items()}


def read_settings(values: dict[str, tk.StringVar],
                  parameters: dict[str, tuple[str, float, float, float]]) -> dict[str, float]:
    settings = {}
    for name, variable in values.items():
        label, _, minimum, maximum = parameters[name]
        read = read_number if name in INTEGER_SETTINGS else read_decimal
        settings[name] = read(variable, label, minimum, maximum)
    return settings


def ask_corners(
    parent: tk.Misc, image: Image.Image
) -> tuple[str, dict[str, float], dict[str, float], str, dict[str, Any], np.ndarray, Image.Image] | None:
    """Let the user tune the detector, its peaks and the view while previewing on ``image``.

    Returns the detector, its parameters, the peak settings, the view, its style, the corners and the
    image to show; ``None`` if cancelled.
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Corner detection")
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
    response_cache: tuple[tuple[Any, ...], np.ndarray] | None = None  # the detector is slow; peaks are not
    preview: tuple[tuple[Any, ...], np.ndarray, Image.Image] | None = None
    photo: ImageTk.PhotoImage | None = None

    def changed() -> None:
        schedule_preview()

    detector = choice_row(controls, "Detector", DETECTORS, changed)

    # Each detector keeps its own settings in its own frame; the frames share one grid cell so the
    # window keeps its size, and the selected detector's frame is raised on top.
    ttk.Label(controls, text="Detector parameters", padding=(0, 16, 0, 0)).pack(anchor="w")
    detector_area = ttk.Frame(controls)
    detector_area.pack(fill="x")
    detector_frames: dict[str, ttk.Frame] = {}
    detector_values: dict[str, dict[str, tk.StringVar]] = {}
    for name, parameters in DETECTOR_PARAMETERS.items():
        frame = ttk.Frame(detector_area)
        frame.grid(row=0, column=0, sticky="nsew")
        detector_frames[name] = frame
        detector_values[name] = setting_rows(frame, parameters, changed)
    detector_frames[DETECTORS[0]].tkraise()
    detector.trace_add("write", lambda *_: detector_frames[detector.get()].tkraise())

    ttk.Label(controls, text="Peaks (corner_peaks)", padding=(0, 16, 0, 0)).pack(anchor="w")
    peak_values = setting_rows(controls, PEAK_PARAMETERS, changed)
    ttk.Label(controls, text="A peak must exceed this fraction of the strongest response.",
              foreground="#666666").pack(anchor="w", pady=(4, 0))

    def reset() -> None:
        for name, parameters in DETECTOR_PARAMETERS.items():
            for setting, (_, default, _, _) in parameters.items():
                detector_values[name][setting].set(f"{default:g}")
        for setting, (_, default, _, _) in PEAK_PARAMETERS.items():
            peak_values[setting].set(f"{default:g}")

    ttk.Button(controls, text="Reset to defaults", command=reset).pack(anchor="w", pady=(8, 0))

    ttk.Label(controls, text="Show", padding=(0, 16, 0, 0)).pack(anchor="w")
    view = tk.StringVar(dialog, value=VIEWS[0])
    for option in VIEWS:
        ttk.Radiobutton(controls, text=option, value=option, variable=view).pack(anchor="w", pady=(2, 0))
    view.trace_add("write", lambda *_: schedule_preview())
    style_parameters = {"radius": ("Circle radius (px)", default_radius(image), 1, 50)}
    style_values = setting_rows(controls, style_parameters, changed)
    base = choice_row(controls, "Draw on", BASES, changed)

    def compute() -> tuple[tuple[Any, ...], dict[str, float], dict[str, float], dict[str, Any],
                           np.ndarray, Image.Image]:
        nonlocal response_cache
        name = detector.get()
        current = read_settings(detector_values[name], DETECTOR_PARAMETERS[name])
        peaks = read_settings(peak_values, PEAK_PARAMETERS)
        style = {**read_settings(style_values, style_parameters), "base": base.get()}
        response_key = (name, *current.values())
        if response_cache is None or response_cache[0] != response_key:
            response_cache = (response_key, corner_response(image, name, **current))
        response = response_cache[1]
        key = (response_key, *peaks.values(), view.get(), *style.values())
        if preview is not None and preview[0] == key:  # nothing changed since the preview
            return key, current, peaks, style, preview[1], preview[2]
        corners = find_corners(response, **peaks)
        return key, current, peaks, style, corners, show_corners(image, response, corners, view.get(), **style)

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
            key, _, _, _, corners, shown = compute()
        except ValueError as error:
            preview = photo = None
            status.configure(text=str(error))
            return
        preview = (key, corners, shown)
        thumbnail = shown.copy()
        thumbnail.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(thumbnail, master=dialog)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        status.configure(text=f"{len(corners)} corners found. Apply keeps exactly this result.")

    result = []

    def close() -> None:
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        try:
            _, current, peaks, style, corners, shown = compute()  # reuses the preview when nothing changed
        except ValueError as error:
            messagebox.showerror("Corner detection", str(error), parent=dialog)
            return
        result.append((detector.get(), current, peaks, view.get(), style, corners, shown))
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


class CornerDetectionTool(ForensicsTool):
    tool_id = "corner_detection"
    title = "Corner detection"
    category = "Feature extraction"
    description = "FAST or Harris corners with an adjustable threshold for the peaks."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        choice = ask_corners(parent, document.current)
        if choice is None:
            return None
        detector, current, peaks, view, style, corners, shown = choice
        labels = {**DETECTOR_PARAMETERS[detector], **PEAK_PARAMETERS}
        return ToolResult(
            image=shown,
            message=f"Detected {len(corners)} corners with {detector}.",
            details={
                "Operation": f"{detector} corner detection",
                **{labels[name][0]: f"{value:g}" for name, value in {**current, **peaks}.items()},
                "Corners": len(corners),
                "Shown": view,
            },
        )
