"""Histogram of Oriented Gradients (HOG): extract the descriptor and visualise it."""

from __future__ import annotations

from collections.abc import Callable
import re
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageTk
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import center_over


# Number setting: (label, default, minimum, maximum). Defaults follow the Set 4 example (skimage's are 9, 8, 3).
# Cells and blocks are (rows, columns), so either can be rectangular.
PARAMETERS = {
    "orientations": ("Orientations (bins)", 9, 1, 18),
    "cell_rows": ("Cell height (px)", 6, 2, 64),
    "cell_cols": ("Cell width (px)", 6, 2, 64),
    "block_rows": ("Block height (cells)", 3, 1, 8),
    "block_cols": ("Block width (cells)", 3, 1, 8),
}
BLOCK_NORMS = ("L2-Hys", "L2", "L1", "L1-sqrt")
GRADIENTS = ("Colour channels", "Grayscale")  # skimage takes, per pixel, the channel with the strongest gradient
VIEWS = ("HOG image", "Over the image", "Side by side")
PREVIEW_SIZE = (480, 360)
PREVIEW_DELAY_MS = 250  # wait for a pause in typing or sliding before running HOG on the full image


def extract_hog(
    image: Image.Image,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (3, 3),
    block_norm: str = BLOCK_NORMS[0],
    gradients: str = GRADIENTS[0],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the HOG feature vector and visualisation from ``skimage.feature.hog``.

    ``pixels_per_cell`` and ``cells_per_block`` are (rows, columns), as in skimage.
    ``gradients`` picks colour (``channel_axis=-1`` on RGB) or grayscale input; a grayscale image is
    always treated as grayscale. Raises ``ValueError`` when one block does not fit in the image.
    """
    block_height = pixels_per_cell[0] * cells_per_block[0]
    block_width = pixels_per_cell[1] * cells_per_block[1]
    if block_height > image.height or block_width > image.width:
        raise ValueError(f"A block is {block_width} × {block_height} px (width × height), "
                         f"larger than the {image.width} × {image.height} image.")
    pixels, channel_axis = hog_pixels(image, gradients)
    return feature.hog(
        pixels,
        orientations=orientations,
        pixels_per_cell=tuple(pixels_per_cell),
        cells_per_block=tuple(cells_per_block),
        block_norm=block_norm,
        visualize=True,
        channel_axis=channel_axis,
    )


def hog_pixels(image: Image.Image, gradients: str = GRADIENTS[0]) -> tuple[np.ndarray, int | None]:
    """The image scaled to 0–1 for ``feature.hog``, with its ``channel_axis`` (``None`` for grayscale)."""
    if gradients == GRADIENTS[0] and image.mode not in ("L", "1", "I", "F"):
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0, -1
    return np.asarray(image.convert("L"), dtype=np.float64) / 255.0, None


def show_hog(
    image: Image.Image, hog_image: np.ndarray, view: str = VIEWS[0], brightness: float = 0.4
) -> Image.Image:
    """Return the HOG visualisation alone, drawn in yellow over the dimmed image, or next to the image.

    The visualisation is scaled so its strongest gradient is white; ``brightness`` is how much of
    the image remains under the overlay.
    """
    peak = hog_image.max()
    strength = hog_image / peak if peak > 0 else np.zeros_like(hog_image)
    if view == "HOG image":
        return Image.fromarray(np.rint(strength * 255).astype(np.uint8))
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    if view == "Over the image":
        glyphs = strength[..., np.newaxis] * np.array([1.0, 0.85, 0.0])
        rgb = np.clip(rgb * brightness * (1 - strength[..., np.newaxis]) + glyphs, 0.0, 1.0)
    elif view == "Side by side":
        rgb = np.concatenate([rgb, np.repeat(strength[..., np.newaxis], 3, axis=2)], axis=1)
    else:
        raise ValueError(f"Unknown view: {view}")
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8))


def _is_digits(text: str) -> bool:
    return re.fullmatch(r"\d*", text) is not None


def read_number(variable: tk.StringVar, label: str, minimum: int, maximum: int) -> int:
    """The typed whole number; raises ``ValueError`` with a message for the status line."""
    try:
        number = int(variable.get())
    except ValueError:
        raise ValueError(f"Enter a whole number for {label.lower()}.") from None
    if not minimum <= number <= maximum:
        raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}.")
    return number


def number_row(parent: tk.Misc, label: str, default: int, minimum: int, maximum: int,
               on_change: Callable[[], None]) -> tk.StringVar:
    """A slider with a box for typing the exact whole number; either one updates the other."""
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=(6, 0))
    ttk.Label(row, text=label, width=17).pack(side="left")
    variable = tk.StringVar(parent, value=str(default))
    slider = ttk.Scale(row, from_=minimum, to=maximum, length=160, value=default)
    slider.pack(side="left", padx=8)
    only_digits = (parent.register(_is_digits), "%P")
    ttk.Entry(row, textvariable=variable, width=7, justify="center",
              validate="key", validatecommand=only_digits).pack(side="left")
    syncing = False

    def slider_moved(value: str) -> None:
        if not syncing:
            variable.set(str(round(float(value))))

    def text_changed(*_: object) -> None:
        nonlocal syncing
        try:
            number = int(variable.get())
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


def choice_row(parent: tk.Misc, label: str, options: tuple[str, ...], on_change: Callable[[], None]) -> tk.StringVar:
    """A row of toggle buttons; the first option starts selected."""
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=(8, 0))
    ttk.Label(row, text=label, width=17).pack(side="left")
    variable = tk.StringVar(parent, value=options[0])
    for option in options:
        ttk.Radiobutton(row, text=option, value=option, variable=variable,
                        style="Toolbutton").pack(side="left", padx=(0, 4))
    variable.trace_add("write", lambda *_: on_change())
    return variable


def hog_controls(
    parent: tk.Misc, parameters: dict[str, tuple[str, int, int, int]], on_change: Callable[[], None]
) -> Callable[[], dict[str, Any]]:
    """Add the HOG settings and a reset button to ``parent``; return a function reading them.

    The reader returns ``extract_hog``'s keyword arguments, raising ``ValueError`` for bad input.
    """
    values = {name: number_row(parent, *spec, on_change) for name, spec in parameters.items()}
    ttk.Label(parent, text="Equal height and width give square cells or blocks.",
              foreground="#666666").pack(anchor="w", pady=(4, 0))
    block_norm = choice_row(parent, "Block norm", BLOCK_NORMS, on_change)
    gradients = choice_row(parent, "Gradients from", GRADIENTS, on_change)

    def reset() -> None:
        for name, (_, default, _, _) in parameters.items():
            values[name].set(str(default))
        block_norm.set(BLOCK_NORMS[0])
        gradients.set(GRADIENTS[0])

    ttk.Button(parent, text="Reset to defaults", command=reset).pack(anchor="w", pady=(8, 0))

    def read() -> dict[str, Any]:
        numbers = {}
        for name, variable in values.items():
            label, _, minimum, maximum = parameters[name]
            numbers[name] = read_number(variable, label, minimum, maximum)
        return {
            "orientations": numbers["orientations"],
            "pixels_per_cell": (numbers["cell_rows"], numbers["cell_cols"]),
            "cells_per_block": (numbers["block_rows"], numbers["block_cols"]),
            "block_norm": block_norm.get(),
            "gradients": gradients.get(),
        }

    return read


def ask_hog(
    parent: tk.Misc, image: Image.Image
) -> tuple[dict[str, Any], str, np.ndarray, Image.Image] | None:
    """Let the user tune HOG and the view while previewing on ``image``; ``None`` if cancelled.

    Returns the HOG parameters, the view, the feature vector and the image to show.
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("HOG features")
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
    hog_cache: tuple[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] | None = None  # HOG is slow; restyling is not
    preview: tuple[tuple[Any, ...], Image.Image] | None = None
    photo: ImageTk.PhotoImage | None = None

    ttk.Label(controls, text="HOG parameters").pack(anchor="w")
    hog_parameters = hog_controls(controls, PARAMETERS, lambda: schedule_preview())

    ttk.Label(controls, text="Show", padding=(0, 16, 0, 0)).pack(anchor="w")
    view = tk.StringVar(dialog, value=VIEWS[0])
    for option in VIEWS:
        ttk.Radiobutton(controls, text=option, value=option, variable=view).pack(anchor="w", pady=(2, 0))
    view.trace_add("write", lambda *_: schedule_preview())

    def compute() -> tuple[tuple[Any, ...], dict[str, Any], np.ndarray, Image.Image]:
        nonlocal hog_cache
        current = hog_parameters()
        hog_key = tuple(current.values())
        if hog_cache is None or hog_cache[0] != hog_key:
            hog_cache = (hog_key, extract_hog(image, **current))
        descriptor, hog_image = hog_cache[1]
        key = (hog_key, view.get())
        if preview is not None and preview[0] == key:  # nothing changed since the preview
            return key, current, descriptor, preview[1]
        return key, current, descriptor, show_hog(image, hog_image, view.get())

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
            key, _, _, shown = compute()
        except ValueError as error:
            preview = photo = None
            status.configure(text=str(error))
            return
        preview = (key, shown)
        thumbnail = shown.copy()
        thumbnail.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(thumbnail, master=dialog)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        status.configure(text="")

    result = []

    def close() -> None:
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        try:
            _, current, descriptor, shown = compute()  # reuses the preview when nothing changed
        except ValueError as error:
            messagebox.showerror("HOG features", str(error), parent=dialog)
            return
        result.append((current, view.get(), descriptor, shown))
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


class HogTool(ForensicsTool):
    tool_id = "hog"
    title = "HOG features"
    category = "Feature extraction"
    description = "Histogram of Oriented Gradients with adjustable orientations, cell and block sizes."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        choice = ask_hog(parent, document.current)
        if choice is None:
            return None
        current, view, descriptor, shown = choice
        (cell_rows, cell_cols), (block_rows, block_cols) = current["pixels_per_cell"], current["cells_per_block"]
        return ToolResult(
            image=shown,
            message="Extracted HOG features.",
            details={
                "Operation": "Histogram of Oriented Gradients",
                "Orientations": current["orientations"],
                "Pixels per cell (rows, cols)": f"({cell_rows}, {cell_cols})",
                "Cells per block (rows, cols)": f"({block_rows}, {block_cols})",
                "Block norm": current["block_norm"],
                "Gradients from": current["gradients"],
                "Feature vector length": f"{descriptor.size:,}",
                "Shown": view,
            },
        )
