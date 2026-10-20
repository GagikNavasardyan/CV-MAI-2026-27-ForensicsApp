"""Person detection: slide a template-sized window over the image and compare HOG features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from skimage import feature

from forensics_app.core import ImageDocument
from . import hog
from .base import ForensicsTool, ToolResult
from .dialogs import center_over


# The HOG settings of the Set 4 example; the limits and labels are the HOG tool's.
DEFAULTS = {"orientations": 16, "cell_rows": 8, "cell_cols": 8, "block_rows": 2, "block_cols": 2}
PARAMETERS = {name: (label, DEFAULTS[name], minimum, maximum)
              for name, (label, _, minimum, maximum) in hog.PARAMETERS.items()}
STEP = ("Step (px)", 2, 1, 32)  # how far the window moves between comparisons
VIEWS = ("Side by side", "Euclidean distance", "Dot product")
BOX_COLOUR = (255, 0, 0)
GAP = 8  # white space between the side-by-side panels
PANEL_TITLES = ("Original image", "Euclidean distance", "Dot product")
PREVIEW_SIZE = (480, 360)
TEMPLATE_SIZE = (90, 160)  # the template and its HOG image, each fitted into this box
PREVIEW_DELAY_MS = 250  # wait for a pause in typing or sliding before scanning the image


@dataclass(frozen=True)
class Detection:
    """Scores of every window position; ``rows`` and ``cols`` are the windows' top-left corners."""

    window: tuple[int, int]  # (height, width), the template's size
    rows: np.ndarray
    cols: np.ndarray
    euclidean: np.ndarray  # (len(rows), len(cols)) distance to the template: lower is more alike
    dot: np.ndarray  # (len(rows), len(cols)) dot product with the template: higher is more alike

    @property
    def best_euclidean(self) -> tuple[int, int]:
        """Top-left (row, column) of the window closest to the template."""
        return self._corner(np.argmin(self.euclidean))

    @property
    def best_dot(self) -> tuple[int, int]:
        """Top-left (row, column) of the window with the largest dot product."""
        return self._corner(np.argmax(self.dot))

    @property
    def windows(self) -> int:
        return self.euclidean.size

    def _corner(self, flat_index: np.intp) -> tuple[int, int]:
        i, j = np.unravel_index(flat_index, self.euclidean.shape)
        return int(self.rows[i]), int(self.cols[j])


def detect_person(
    image: Image.Image,
    template: Image.Image,
    orientations: int = 16,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
    block_norm: str = hog.BLOCK_NORMS[0],
    gradients: str = hog.GRADIENTS[0],
    step: int = 2,
) -> Detection:
    """Compare the template's HOG with the HOG of every template-sized window, ``step`` px apart.

    Every window position is tried, including the last row and column. Raises ``ValueError`` when
    the template is larger than the image or a HOG block does not fit in the template.
    """
    if step < 1:
        raise ValueError("The step must be at least 1 px.")
    if template.width > image.width or template.height > image.height:
        raise ValueError(f"The {template.width} × {template.height} template is larger than "
                         f"the {image.width} × {image.height} image.")
    block_height, block_width = pixels_per_cell[0] * cells_per_block[0], pixels_per_cell[1] * cells_per_block[1]
    if block_height > template.height or block_width > template.width:
        raise ValueError(f"A block is {block_width} × {block_height} px (width × height), "
                         f"larger than the {template.width} × {template.height} template.")
    settings = {"orientations": orientations, "pixels_per_cell": tuple(pixels_per_cell),
                "cells_per_block": tuple(cells_per_block), "block_norm": block_norm}
    template_fd, _ = hog.extract_hog(template, gradients=gradients, **settings)
    pixels, channel_axis = hog.hog_pixels(image, gradients)
    height, width = template.height, template.width
    rows = np.arange(0, image.height - height + 1, step)
    cols = np.arange(0, image.width - width + 1, step)
    euclidean = np.empty((rows.size, cols.size))
    dot = np.empty((rows.size, cols.size))
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            window_fd = feature.hog(pixels[row:row + height, col:col + width], channel_axis=channel_axis, **settings)
            euclidean[i, j] = np.linalg.norm(template_fd - window_fd)
            dot[i, j] = np.dot(window_fd, template_fd)
    return Detection((height, width), rows, cols, euclidean, dot)


def draw_box(image: Image.Image, corner: tuple[int, int], window: tuple[int, int]) -> Image.Image:
    """A colour copy of ``image`` with a red rectangle around the window at ``corner`` (row, column)."""
    result = image.convert("RGB")
    (row, col), (height, width) = corner, window
    ImageDraw.Draw(result).rectangle((col, row, col + width - 1, row + height - 1), outline=BOX_COLOUR,
                                     width=max(2, round(min(image.size) / 100)))
    return result


def show_detection(image: Image.Image, detection: Detection, view: str = VIEWS[0]) -> Image.Image:
    """The best window by Euclidean distance or by dot product, or both next to the original (as on the slide).

    Side by side, each panel has its title below it.
    """
    if view == "Euclidean distance":
        return draw_box(image, detection.best_euclidean, detection.window)
    if view == "Dot product":
        return draw_box(image, detection.best_dot, detection.window)
    if view != "Side by side":
        raise ValueError(f"Unknown view: {view}")
    panels = [image.convert("RGB"), draw_box(image, detection.best_euclidean, detection.window),
              draw_box(image, detection.best_dot, detection.window)]
    font = _title_font(image.width)
    title_height = round(font.size * 1.8)
    result = Image.new("RGB", (3 * image.width + 2 * GAP, title_height + image.height), (255, 255, 255))
    draw = ImageDraw.Draw(result)
    for index, (panel, title) in enumerate(zip(panels, PANEL_TITLES)):
        left = index * (image.width + GAP)
        result.paste(panel, (left, 0))
        draw.text((left + image.width / 2, image.height + title_height / 2), title, fill=(0, 0, 0), font=font,
                  anchor="mm")
    return result


def _title_font(panel_width: int) -> ImageFont.FreeTypeFont:
    """Pillow's built-in font, about a tenth of the panel width, shrunk until the longest title fits."""
    size = max(8, panel_width // 10)
    while size > 8 and ImageFont.load_default(size).getlength(max(PANEL_TITLES, key=len)) > panel_width - 4:
        size -= 1
    return ImageFont.load_default(size)


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale for display only, up or down, to fit ``size``."""
    scale = min(size[0] / image.width, size[1] / image.height)
    return image.convert("RGB").resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                                       Image.Resampling.LANCZOS)


def ask_detection(
    parent: tk.Misc, image: Image.Image, template: Image.Image, template_name: str
) -> tuple[dict[str, Any], str, Detection, Image.Image] | None:
    """Let the user tune HOG, the step and the view while previewing on ``image``; ``None`` if cancelled.

    Returns the settings (``detect_person``'s keyword arguments), the view, the detection and the image to show.
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Person detection (HOG)")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()
    controls = ttk.Frame(body)
    controls.grid(row=0, column=0, sticky="n")
    preview_side = ttk.Frame(body, padding=(20, 0, 0, 0))
    preview_side.grid(row=0, column=1, sticky="n")

    ttk.Label(controls, text=f"Template: {template_name} ({template.width} × {template.height} px)").pack(anchor="w")
    template_row = ttk.Frame(controls)
    template_row.pack(anchor="w", pady=(4, 0))
    template_canvases = []
    for _ in range(2):  # the template and its HOG image
        template_canvas = tk.Canvas(template_row, width=TEMPLATE_SIZE[0], height=TEMPLATE_SIZE[1],
                                    background="#20242b", highlightthickness=0)
        template_canvas.pack(side="left", padx=(0, 8))
        template_canvases.append(template_canvas)

    ttk.Label(controls, text="HOG parameters", padding=(0, 16, 0, 0)).pack(anchor="w")
    hog_parameters = hog.hog_controls(controls, PARAMETERS, lambda: schedule_preview())
    ttk.Label(controls, text="Sliding window", padding=(0, 16, 0, 0)).pack(anchor="w")
    step = hog.number_row(controls, *STEP, lambda: schedule_preview())

    ttk.Label(controls, text="Show", padding=(0, 16, 0, 0)).pack(anchor="w")
    view = tk.StringVar(dialog, value=VIEWS[0])
    for option in VIEWS:
        ttk.Radiobutton(controls, text=option, value=option, variable=view).pack(anchor="w", pady=(2, 0))
    view.trace_add("write", lambda *_: schedule_preview())

    ttk.Label(preview_side, text="Preview").pack(anchor="w")
    canvas = tk.Canvas(preview_side, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1],
                       background="#20242b", highlightthickness=0)
    canvas.pack(pady=(4, 2))
    status = ttk.Label(preview_side, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    preview_job: str | None = None
    detection_cache: tuple[tuple[Any, ...], Detection] | None = None  # scanning is slow; redrawing is not
    preview: tuple[tuple[Any, ...], Image.Image] | None = None
    photos: list[ImageTk.PhotoImage] = []

    def settings() -> dict[str, Any]:
        label, _, minimum, maximum = STEP
        return {**hog_parameters(), "step": hog.read_number(step, label, minimum, maximum)}

    def compute() -> tuple[tuple[Any, ...], dict[str, Any], Detection, Image.Image]:
        nonlocal detection_cache
        current = settings()
        scan_key = tuple(current.values())
        if detection_cache is None or detection_cache[0] != scan_key:
            detection_cache = (scan_key, detect_person(image, template, **current))
        detection = detection_cache[1]
        key = (scan_key, view.get())
        if preview is not None and preview[0] == key:  # nothing changed since the preview
            return key, current, detection, preview[1]
        return key, current, detection, show_detection(image, detection, view.get())

    def schedule_preview() -> None:
        nonlocal preview_job
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        status.configure(text="Updating preview…")
        preview_job = dialog.after(PREVIEW_DELAY_MS, update_preview)

    def draw(target: tk.Canvas, shown: Image.Image, size: tuple[int, int]) -> None:
        photo = ImageTk.PhotoImage(_fit(shown, size), master=dialog)
        photos.append(photo)  # Tk shows a photo only while Python keeps a reference
        target.create_image(size[0] // 2, size[1] // 2, image=photo)

    def update_preview() -> None:
        nonlocal preview_job, preview
        preview_job = None
        photos.clear()
        for target in (canvas, *template_canvases):
            target.delete("all")
        try:
            current = settings()
            _, template_hog = hog.extract_hog(template, **{k: v for k, v in current.items() if k != "step"})
            key, _, detection, shown = compute()
        except ValueError as error:
            preview = None
            status.configure(text=str(error))
            return
        preview = (key, shown)
        draw(template_canvases[0], template, TEMPLATE_SIZE)
        draw(template_canvases[1], hog.show_hog(template, template_hog), TEMPLATE_SIZE)
        draw(canvas, shown, PREVIEW_SIZE)
        status.configure(text=f"Searched {detection.windows:,} windows.")

    result = []

    def close() -> None:
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        try:
            _, current, detection, shown = compute()  # reuses the preview when nothing changed
        except ValueError as error:
            messagebox.showerror("Person detection", str(error), parent=dialog)
            return
        result.append((current, view.get(), detection, shown))
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


class PersonDetectionTool(ForensicsTool):
    tool_id = "person_detection"
    title = "Person detection (HOG)"
    category = "Feature extraction"
    description = "Find the window whose HOG features best match a person template."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        filename = filedialog.askopenfilename(
            title="Choose the person template",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")], parent=parent,
        )
        if not filename:
            return None
        with Image.open(filename) as template_file:
            template = template_file.copy()
        choice = ask_detection(parent, document.current, template, Path(filename).name)
        if choice is None:
            return None
        current, view, detection, shown = choice
        (cell_rows, cell_cols), (block_rows, block_cols) = current["pixels_per_cell"], current["cells_per_block"]
        euclidean_row, euclidean_col = detection.best_euclidean
        dot_row, dot_col = detection.best_dot
        return ToolResult(
            image=shown,
            message="Searched for the person with HOG.",
            details={
                "Operation": "Person detection with HOG",
                "Template": f"{Path(filename).name} ({template.width} × {template.height} px)",
                "Orientations": current["orientations"],
                "Pixels per cell (rows, cols)": f"({cell_rows}, {cell_cols})",
                "Cells per block (rows, cols)": f"({block_rows}, {block_cols})",
                "Block norm": current["block_norm"],
                "Gradients from": current["gradients"],
                "Step": f"{current['step']} px",
                "Windows searched": f"{detection.windows:,}",
                "Euclidean best (x, y)": f"({euclidean_col}, {euclidean_row}), "
                                         f"distance {detection.euclidean.min():.3f}",
                "Dot product best (x, y)": f"({dot_col}, {dot_row}), value {detection.dot.max():.3f}",
                "Methods agree": "Yes" if detection.best_euclidean == detection.best_dot else "No",
                "Shown": view,
            },
        )
