"""Normalized cross-correlation image with skimage.feature.match_template."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageTk
from skimage import feature, transform

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .edge_detection import BASES, COLOURS, VIEWS, show_edges


def _grayscale_pixels(image: Image.Image) -> np.ndarray:
    if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
        raise ValueError("Use an 8-bit grayscale or RGB image for template matching.")
    return np.asarray(image.convert("L"), dtype=float) / 255


# How transform.rotate fills the corners a rotation leaves uncovered. Black corners (the
# slide's default) count against every position, so extending the border matches better.
CORNER_FILLS = {"Extend edges": "edge", "Black (as on the slide)": "constant"}
DEFAULT_FILL = "Extend edges"


def rotate_template(
    template: Image.Image, angle: float = 0.0, fill: str = DEFAULT_FILL,
) -> np.ndarray:
    """Grayscale template rotated counter-clockwise about its centre, keeping its size."""
    return transform.rotate(_grayscale_pixels(template), angle, mode=CORNER_FILLS[fill])


def correlation_map(
    image: Image.Image, template: Image.Image, angle: float = 0.0, fill: str = DEFAULT_FILL,
) -> np.ndarray:
    """Normalized cross-correlation of the template at every pixel, between -1 and 1.

    pad_input=True gives a map with the image's dimensions, centred on each pixel.
    """
    pixels, template_pixels = _grayscale_pixels(image), rotate_template(template, angle, fill)
    if template_pixels.shape[0] > pixels.shape[0] or template_pixels.shape[1] > pixels.shape[1]:
        raise ValueError("The template must not be larger than the image.")
    if template_pixels.size == 0:
        raise ValueError("The template must contain pixels.")
    return feature.match_template(pixels, template_pixels, pad_input=True)


def correlation_image(correlation: np.ndarray) -> Image.Image:
    """Stretch the map's minimum to black and its maximum to white, like imshow."""
    low, high = correlation.min(), correlation.max()
    scaled = (correlation - low) / (high - low) if high > low else np.zeros_like(correlation)
    return Image.fromarray(np.rint(scaled * 255).astype(np.uint8))


# Default display of the slide: the dimmed grayscale image with the matches in red.
DEFAULT_VIEW, DEFAULT_COLOUR = "Bright edges, dim image", "Red"


def thresholded_image(
    image: Image.Image, correlation: np.ndarray, threshold: float = 0.7,
    view: str = DEFAULT_VIEW, colour: str = DEFAULT_COLOUR, thickness: int = 1,
) -> Image.Image:
    """Superimpose the positions scoring at least the threshold, with the views from Set 3."""
    return show_edges(image, correlation >= threshold, view, colour, thickness, base=BASES[1])


PANELS = ("Original image", "Template", "Normalized cross-correlation", "Thresholded image")
APPLY_OPTIONS = PANELS[2:]
# Number setting: (label, default, minimum, maximum, step).
SETTINGS = {
    "angle": ("Template angle (degrees, counter-clockwise)", 0.0, -180.0, 180.0, 5.0),
    "threshold": ("Threshold (score from -1 to 1)", 0.7, -1.0, 1.0, 0.01),
    "thickness": ("Mark thickness (px)", 1, 1, 10, 1),
}


class _TemplateMatchingDialog(simpledialog.Dialog):
    """The four images of the slide, updated as the settings change; OK applies one result."""

    def __init__(
        self, parent: tk.Misc, image: Image.Image, template: Image.Image, template_name: str,
    ) -> None:
        self.source, self.template, self.template_name = image.copy(), template, template_name
        self._preview_after: str | None = None
        self._preview_result: ToolResult | None = None
        super().__init__(parent, title="Template matching")

    def body(self, master: tk.Misc) -> ttk.Spinbox:
        controls = ttk.Frame(master)
        controls.grid(row=0, column=0, columnspan=2, sticky="w", padx=5)
        self.values: dict[str, tk.StringVar] = {}
        spinboxes = []
        for row, (name, (label, default, low, high, step)) in enumerate(SETTINGS.items()):
            self.values[name] = tk.StringVar(master, value=str(default))
            ttk.Label(controls, text=label + ":").grid(row=row, column=0, sticky="w", pady=2)
            spinbox = ttk.Spinbox(controls, textvariable=self.values[name], from_=low, to=high,
                                  increment=step, width=8)
            spinbox.grid(row=row, column=1, sticky="w", padx=6, pady=2)
            spinboxes.append(spinbox)
        self.view = tk.StringVar(master, value=DEFAULT_VIEW)
        self.colour = tk.StringVar(master, value=DEFAULT_COLOUR)
        self.fill = tk.StringVar(master, value=DEFAULT_FILL)
        for row, (label, variable, values, width) in enumerate((
            ("Superimposition", self.view, VIEWS, 22), ("Colour", self.colour, tuple(COLOURS), 8),
            ("Corner fill", self.fill, tuple(CORNER_FILLS), 22),
        )):
            ttk.Label(controls, text=label + ":").grid(row=row, column=2, sticky="w", padx=(20, 0))
            ttk.Combobox(controls, textvariable=variable, values=values, state="readonly",
                         width=width).grid(row=row, column=3, sticky="w", padx=6, pady=2)

        self.canvases = []
        self._photos: list[ImageTk.PhotoImage | None] = [None] * 4
        for index, title in enumerate(PANELS):
            row, column = divmod(index, 2)
            ttk.Label(master, text=title).grid(row=row * 2 + 1, column=column, pady=(6, 2))
            canvas = tk.Canvas(master, width=330, height=260, background="white",
                               highlightthickness=0)
            canvas.grid(row=row * 2 + 2, column=column, padx=4)
            self.canvases.append(canvas)
        self._show_image(0, self.source)

        self.status = tk.StringVar(master)
        ttk.Label(master, textvariable=self.status, wraplength=660).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=5, pady=8,
        )
        options = ttk.Frame(master)
        options.grid(row=6, column=0, columnspan=2, sticky="w", padx=5)
        ttk.Label(options, text="OK applies:").grid(row=0, column=0, padx=(0, 6))
        self.choice = tk.StringVar(master, value=APPLY_OPTIONS[1])
        ttk.Combobox(options, textvariable=self.choice, values=APPLY_OPTIONS,
                     state="readonly", width=28).grid(row=0, column=1)

        for variable in (*self.values.values(), self.view, self.colour, self.fill,
                         self.choice):
            variable.trace_add("write", self._schedule_preview)
        self._schedule_preview()
        return spinboxes[0]

    def _settings(self) -> tuple[float, float, int]:
        numbers = []
        for name, (label, _default, low, high, _step) in SETTINGS.items():
            try:
                number = int(self.values[name].get()) if name == "thickness" \
                    else float(self.values[name].get())
            except ValueError:
                raise ValueError(f"Enter a number for {label.lower()}.") from None
            if not low <= number <= high:
                raise ValueError(f"{label} must be between {low:g} and {high:g}.")
            numbers.append(number)
        angle, threshold, thickness = numbers
        return angle, threshold, thickness

    def _schedule_preview(self, *_args: str) -> None:
        self._preview_result = None
        self.status.set("Updating preview…")
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
        # Wait for a pause while typing; the correlation always uses the full image.
        self._preview_after = self.after(250, self._update_preview)

    def _show_image(self, index: int, image: Image.Image | None) -> None:
        canvas = self.canvases[index]
        canvas.delete("all")
        self._photos[index] = None
        if image is not None:
            self._photos[index] = ImageTk.PhotoImage(_fit(image, 320, 250), master=self)
            canvas.create_image(165, 130, image=self._photos[index])

    def _update_preview(self) -> bool:
        self._preview_after = None
        try:
            angle, threshold, thickness = self._settings()
            template = Image.fromarray(np.rint(rotate_template(self.template, angle, self.fill.get()) * 255)
                                       .astype(np.uint8))
            correlation = correlation_map(self.source, self.template, angle, self.fill.get())
        except ValueError as error:
            self._preview_result = None
            for index in (1, 2, 3):
                self._show_image(index, None)
            self.status.set(str(error))
            return False
        results = {
            APPLY_OPTIONS[0]: correlation_image(correlation),
            APPLY_OPTIONS[1]: thresholded_image(self.source, correlation, threshold,
                                                self.view.get(), self.colour.get(), thickness),
        }
        for index, image in enumerate((template, *results.values()), start=1):
            self._show_image(index, image)

        row, column = np.unravel_index(np.argmax(correlation), correlation.shape)
        best, best_score = (int(column), int(row)), float(correlation[row, column])
        matches = int((correlation >= threshold).sum())
        self.status.set(
            f"Best match at (x, y) = {best} with score {best_score:.3f}. {matches} position(s) "
            f"score at least {threshold:g}. Scores range from -1 to 1; the correlation image "
            "shows its lowest score as black and its highest as white."
            + (" The rotated template's black corners also take part in the matching."
               if angle % 90 and self.fill.get() != DEFAULT_FILL else "")
        )
        self._preview_result = ToolResult(
            image=results[self.choice.get()],
            message=f"Applied the {self.choice.get().lower()} for {self.template_name}.",
            details={
                "Template": self.template_name,
                "Template size": f"{self.template.width} × {self.template.height}",
                "Template angle (degrees)": angle,
                "Corner fill": self.fill.get(),
                "Best match (x, y)": f"{best}",
                "Best score": round(best_score, 4),
                "Score range": f"{correlation.min():.4f} to {correlation.max():.4f}",
                "Threshold": threshold,
                "Positions at or above threshold": matches,
                "Superimposition": self.view.get(),
            },
        )
        return True

    def validate(self) -> bool:
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
            self._preview_after = None
        if self._preview_result is None and not self._update_preview():
            messagebox.showerror(self.title(), self.status.get(), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._preview_result

    def cancel(self, event: tk.Event | None = None) -> None:
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
            self._preview_after = None
        super().cancel(event)


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    """Scale for display only; small images such as templates are enlarged pixel by pixel."""
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resample = Image.Resampling.NEAREST if scale >= 1 else Image.Resampling.LANCZOS
    return image.convert("RGB").resize(size, resample)


class TemplateMatchingTool(ForensicsTool):
    tool_id = "template_matching"
    title = "Template matching"
    category = "Template matching"
    description = "Find a template with normalized cross-correlation, a rotation angle and a threshold."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        filename = filedialog.askopenfilename(
            title="Choose the template to look for",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")], parent=parent,
        )
        if not filename:
            return None
        with Image.open(filename) as template_file:
            template = template_file.copy()
        _grayscale_pixels(template)  # reject unsupported modes before opening the dialog
        return _TemplateMatchingDialog(
            parent, document.current, template, Path(filename).name,
        ).result
