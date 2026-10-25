"""Match ORB or SIFT descriptors between the current image and a second one, drawn side by side."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .corner_detection import BASES, default_radius, read_settings, setting_rows
from .dialogs import center_over, preview_copy
from .hog import choice_row
from .keypoints import INTEGER_SETTINGS, METHOD_PARAMETERS, METHODS, UPSAMPLING, Keypoints, extract_keypoints


# ORB descriptors are bit strings, compared by the number of differing bits; SIFT's are vectors of numbers.
METRICS = {"ORB": "hamming", "SIFT": "euclidean"}
CROSS_CHECK = ("On", "Off")
# max_ratio keeps a match only if its distance is below this fraction of the second-best one; 1 keeps them all.
# 0.8 is Lowe's value: without it, an image with more keypoints simply gets more matches.
MATCH_PARAMETERS = {"max_ratio": ("Max distance ratio", 0.8, 0.1, 1.0)}
KEYPOINT_COLOUR = (255, 0, 0)
LINE_COLOURS = ((31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                (140, 86, 75), (227, 119, 194), (188, 189, 34), (23, 190, 207))  # matplotlib's tab10, minus grey
IMAGE_TYPES = [("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")]
PREVIEW_SIZE = (640, 360)
PREVIEW_DELAY_MS = 400  # SIFT takes a while, so wait a little longer for a pause in typing or sliding


def match_keypoints(first: Keypoints, second: Keypoints, cross_check: bool = True,
                    max_ratio: float = 1.0) -> np.ndarray:
    """``skimage.feature.match_descriptors`` with the metric that suits the extractor.

    Returns an (M, 2) array: index into ``first``, index into ``second``.
    """
    if first.method != second.method:
        raise ValueError("Both images must use the same extractor.")
    if not len(first) or not len(second):
        return np.empty((0, 2), dtype=np.intp)
    return feature.match_descriptors(first.descriptors, second.descriptors, metric=METRICS[first.method],
                                     cross_check=cross_check, max_ratio=max_ratio)


def show_matches(first_image: Image.Image, second_image: Image.Image, first: Keypoints, second: Keypoints,
                 matches: np.ndarray, base: str = BASES[0]) -> Image.Image:
    """Both images side by side, the second scaled to the first one's height, with every keypoint as a red
    circle and a line joining each match."""
    factor = first_image.height / second_image.height
    size = (max(1, round(second_image.width * factor)), first_image.height)
    mode = "L" if base == BASES[0] else "RGB"
    left = first_image.convert(mode).convert("RGB")
    right = second_image.convert(mode).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    shown = Image.new("RGB", (left.width + right.width, left.height))
    shown.paste(left)
    shown.paste(right, (left.width, 0))
    second = second.scaled(factor)
    offset = np.array([0, left.width])

    draw = ImageDraw.Draw(shown)
    radius = default_radius(first_image)
    width = max(1, radius // 4)
    for row, col in (*first.points, *(second.points + offset)):
        draw.ellipse((col - radius, row - radius, col + radius, row + radius), outline=KEYPOINT_COLOUR, width=width)
    for number, (i, j) in enumerate(matches):
        (row1, col1), (row2, col2) = first.points[i], second.points[j] + offset
        draw.line((col1, row1, col2, row2), fill=LINE_COLOURS[number % len(LINE_COLOURS)], width=width)
    return shown


def matching_controls(parent: tk.Misc, on_change: Callable[[], None]) -> Callable[[], dict[str, Any]]:
    """Add the extractor and matching settings to ``parent``; return a function reading them.

    The reader returns the method, its ``extract_keypoints`` parameters, ``cross_check`` and ``max_ratio``,
    raising ``ValueError`` for bad input.
    """
    method = choice_row(parent, "Extractor", METHODS, on_change)

    # Each extractor keeps its own settings in its own frame; the frames share one grid cell so the
    # window keeps its size, and the selected extractor's frame is raised on top.
    ttk.Label(parent, text="Extractor parameters", padding=(0, 16, 0, 0)).pack(anchor="w")
    method_area = ttk.Frame(parent)
    method_area.pack(fill="x")
    method_frames: dict[str, ttk.Frame] = {}
    method_values: dict[str, dict[str, tk.StringVar]] = {}
    for name, parameters in METHOD_PARAMETERS.items():
        frame = ttk.Frame(method_area)
        frame.grid(row=0, column=0, sticky="nsew")
        method_frames[name] = frame
        method_values[name] = setting_rows(frame, parameters, on_change, INTEGER_SETTINGS)
    upsampling = choice_row(method_frames["SIFT"], "Upsampling", UPSAMPLING, on_change, default="2")
    method_frames[METHODS[0]].tkraise()
    method.trace_add("write", lambda *_: method_frames[method.get()].tkraise())

    ttk.Label(parent, text="Matching", padding=(0, 16, 0, 0)).pack(anchor="w")
    cross_check = choice_row(parent, "Cross check", CROSS_CHECK, on_change)
    match_values = setting_rows(parent, MATCH_PARAMETERS, on_change, INTEGER_SETTINGS)
    ttk.Label(parent, text="Cross check keeps a match only if each keypoint is the other's best;\n"
                           "a ratio below 1 also drops matches almost as close as the runner-up.",
              foreground="#666666").pack(anchor="w", pady=(4, 0))

    def reset() -> None:
        for name, parameters in METHOD_PARAMETERS.items():
            for setting, (_, default, _, _) in parameters.items():
                method_values[name][setting].set(f"{default:g}")
        upsampling.set("2")
        cross_check.set(CROSS_CHECK[0])
        match_values["max_ratio"].set(f"{MATCH_PARAMETERS['max_ratio'][1]:g}")

    ttk.Button(parent, text="Reset to defaults", command=reset).pack(anchor="w", pady=(8, 0))

    def read() -> dict[str, Any]:
        name = method.get()
        parameters: dict[str, Any] = read_settings(method_values[name], METHOD_PARAMETERS[name], INTEGER_SETTINGS)
        if name == "SIFT":
            parameters["upsampling"] = int(upsampling.get())
        return {
            "method": name,
            "parameters": parameters,
            "cross_check": cross_check.get() == CROSS_CHECK[0],
            **read_settings(match_values, MATCH_PARAMETERS, INTEGER_SETTINGS),
        }

    return read


def matching_details(settings: dict[str, Any]) -> dict[str, Any]:
    """The settings as rows for Results."""
    labels = {name: label for name, (label, *_) in METHOD_PARAMETERS[settings["method"]].items()}
    labels["upsampling"] = "Upsampling"
    return {
        "Extractor": settings["method"],
        **{labels[name]: f"{value:g}" for name, value in settings["parameters"].items()},
        "Metric": METRICS[settings["method"]],
        "Cross check": "On" if settings["cross_check"] else "Off",
        "Max distance ratio": f"{settings['max_ratio']:g}",
    }


def ask_matches(
    parent: tk.Misc, image: Image.Image, second_image: Image.Image, second_name: str
) -> tuple[dict[str, Any], Keypoints, Keypoints, np.ndarray, Image.Image] | None:
    """Let the user configure the extractor and the matching while previewing; ``None`` if cancelled.

    Returns the settings, the keypoints of each image, the matches and the image to show.
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Descriptor matching")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()
    controls = ttk.Frame(body)
    controls.grid(row=0, column=0, sticky="n")
    preview_side = ttk.Frame(body, padding=(20, 0, 0, 0))
    preview_side.grid(row=0, column=1, sticky="n")

    ttk.Label(preview_side, text=f"Preview: current image ↔ {second_name}").pack(anchor="w")
    canvas = tk.Canvas(preview_side, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1],
                       background="#20242b", highlightthickness=0)
    canvas.pack(pady=(4, 2))
    status = ttk.Label(preview_side, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    preview_job: str | None = None
    # Large images preview on smaller copies; Apply uses the full ones.
    copies = [preview_copy(image), preview_copy(second_image)]
    reduced = any(factor > 1 for _, factor in copies)
    keypoints_cache: dict[tuple[int, bool], tuple[tuple[Any, ...], Keypoints]] = {}  # extracting is slow
    preview: tuple[tuple[Any, ...], Image.Image] | None = None
    photo: ImageTk.PhotoImage | None = None

    read_matching = matching_controls(controls, lambda: schedule_preview())
    base = choice_row(controls, "Draw on", BASES, lambda: schedule_preview())

    def keypoints_of(index: int, full: bool, method_key: tuple[Any, ...]) -> Keypoints:
        """Extract on the full image, or else on its preview copy with the keypoints scaled back up."""
        if (index, full) not in keypoints_cache or keypoints_cache[index, full][0] != method_key:
            name, parameters = method_key[0], dict(method_key[1:])
            original, (small, factor) = (image, second_image)[index], copies[index]
            found = extract_keypoints(original, name, **parameters) if full else \
                extract_keypoints(small, name, **parameters).scaled(factor)
            keypoints_cache[index, full] = (method_key, found)
        return keypoints_cache[index, full][1]

    def compute(full: bool = not reduced) -> tuple[tuple[Any, ...], dict[str, Any], Keypoints, Keypoints,
                                                   np.ndarray, Image.Image]:
        settings = read_matching()
        method_key = (settings["method"], *settings["parameters"].items())
        first, second = keypoints_of(0, full, method_key), keypoints_of(1, full, method_key)
        matches = match_keypoints(first, second, settings["cross_check"], settings["max_ratio"])
        key = (full, method_key, settings["cross_check"], settings["max_ratio"], base.get())
        if preview is not None and preview[0] == key:  # nothing changed since the preview
            return key, settings, first, second, matches, preview[1]
        return key, settings, first, second, matches, show_matches(image, second_image, first, second, matches,
                                                                   base.get())

    def schedule_preview() -> None:
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
            key, _, first, second, matches, shown = compute()
        except ValueError as error:
            preview = photo = None
            status.configure(text=str(error))
            return
        preview = (key, shown)
        thumbnail = shown.copy()
        thumbnail.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(thumbnail, master=dialog)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        found = f"{len(first)} and {len(second)} keypoints, {len(matches)} matches."
        if reduced:
            status.configure(text=f"{found} Previewed on smaller copies for speed; Apply runs on the full images.")
        else:
            status.configure(text=f"{found} Apply keeps exactly this result.")

    result = []

    def close() -> None:
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        if reduced:
            status.configure(text="Running on the full images…")
            dialog.configure(cursor="watch")
            dialog.update_idletasks()
        try:
            result.append(compute(full=True)[1:])  # reuses the preview when nothing changed
        except ValueError as error:
            dialog.configure(cursor="")
            messagebox.showerror("Descriptor matching", str(error), parent=dialog)
            return
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


class DescriptorMatchingTool(ForensicsTool):
    tool_id = "descriptor_matching"
    title = "Descriptor matching"
    category = "Feature extraction"
    description = "Match ORB or SIFT descriptors with a second image."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        filename = filedialog.askopenfilename(title="Choose the second image", filetypes=IMAGE_TYPES, parent=parent)
        if not filename:
            return None
        with Image.open(filename) as second_file:
            second_image = second_file.copy()
        choice = ask_matches(parent, document.current, second_image, Path(filename).name)
        if choice is None:
            return None
        settings, first, second, matches, shown = choice
        return ToolResult(
            image=shown,
            message=f"Found {len(matches)} {settings['method']} matches.",
            details={
                "Operation": "Descriptor matching",
                "Second image": f"{Path(filename).name} ({second_image.width} × {second_image.height} px)",
                **matching_details(settings),
                "Keypoints (current, second)": f"{len(first)}, {len(second)}",
                "Matches": len(matches),
            },
        )
