"""Find the image of a folder that best matches the current one: the one with the most descriptor matches."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .corner_detection import BASES
from .descriptor_matching import RANSAC, find_matches, matching_controls, matching_details, show_matches
from .dialogs import center_over, preview_copy
from .hog import choice_row
from .keypoints import Keypoints, extract_keypoints


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def folder_images(folder: Path, exclude: Path | None = None) -> list[Path]:
    """The image files in ``folder`` and its subfolders, by path, leaving out ``exclude`` (the current image)."""
    skipped = exclude.resolve() if exclude is not None else None
    return sorted((path for path in folder.rglob("*")
                   if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file() and path.resolve() != skipped),
                  key=lambda path: path.relative_to(folder).as_posix().lower())


def compared_keypoints(image: Image.Image, settings: dict[str, Any]) -> tuple[Keypoints, float]:
    """The keypoints of ``image`` at no more than ``PREVIEW_MAX_SIDE`` px, with the factor back to full size.

    Every image, the reference included, is compared at this size, so a folder of large photos stays quick
    and the RANSAC tolerance means the same for all of them.
    """
    small, factor = preview_copy(image)
    return extract_keypoints(small, settings["method"], **settings["parameters"]), factor


def rank_images(
    reference: Image.Image, paths: list[Path], settings: dict[str, Any],
    progress: Callable[[int, Path], bool] = lambda index, path: True,
) -> list[tuple[Path, int]] | None:
    """Match ``reference`` against every image in ``paths``; return the images and their number of matches
    (only the consistent ones with RANSAC), most first (ties keep the folder order).

    ``progress`` is called before each image and stops the search, returning ``None``, when it returns ``False``.
    """
    reference_keypoints, _ = compared_keypoints(reference, settings)
    counts = []
    for index, path in enumerate(paths):
        if not progress(index, path):
            return None
        with Image.open(path) as image_file:
            keypoints, _ = compared_keypoints(image_file.copy(), settings)
        counts.append((path, len(find_matches(reference_keypoints, keypoints, settings))))
    return sorted(counts, key=lambda item: -item[1])


def best_match_image(reference: Image.Image, path: Path, settings: dict[str, Any],
                     base: str = BASES[0]) -> tuple[Image.Image, Keypoints, Keypoints, np.ndarray]:
    """The reference and the image at ``path`` side by side with their matches, as ``rank_images`` found them,
    and the keypoints of each at full scale."""
    with Image.open(path) as image_file:
        image = image_file.copy()
    (reference_keypoints, reference_factor), (keypoints, factor) = (compared_keypoints(reference, settings),
                                                                    compared_keypoints(image, settings))
    matches = find_matches(reference_keypoints, keypoints, settings)
    reference_keypoints, keypoints = reference_keypoints.scaled(reference_factor), keypoints.scaled(factor)
    shown = show_matches(reference, image, reference_keypoints, keypoints, matches, base)
    return shown, reference_keypoints, keypoints, matches


def ask_lookup(
    parent: tk.Misc, image: Image.Image, folder: Path, paths: list[Path]
) -> tuple[dict[str, Any], str, list[tuple[Path, int]]] | None:
    """Let the user configure the extractor and the matching, then search the folder; ``None`` if cancelled.

    Returns the settings, the base to draw on and the ranking.
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Closest image in a folder")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()

    ttk.Label(body, text=f"Compare the current image with the {len(paths)} images in “{folder.name}” "
                         f"and its subfolders.").pack(anchor="w")
    read_matching = matching_controls(body, lambda: None, ransac_default=RANSAC[1])
    base = choice_row(body, "Draw on", BASES, lambda: None)
    status = ttk.Label(body, foreground="#666666", wraplength=420,
                       text="The image with the most matches wins; with RANSAC on, only the consistent "
                            "matches count. Large images are compared on smaller copies for speed.")
    status.pack(anchor="w", pady=(12, 0))

    result = []
    searching = False
    stopped = False

    def close() -> None:
        nonlocal stopped
        stopped = True  # also stops a search in progress
        dialog.destroy()

    def progress(index: int, path: Path) -> bool:
        if stopped:
            return False
        status.configure(text=f"Comparing {index + 1} of {len(paths)}: {path.relative_to(folder).as_posix()}…")
        dialog.update()  # keeps the window responsive, so Cancel works during the search
        return not stopped

    def search(_event: tk.Event | None = None) -> None:
        nonlocal searching
        if searching:
            return
        try:
            settings = read_matching()
        except ValueError as error:
            messagebox.showerror("Closest image in a folder", str(error), parent=dialog)
            return
        searching = True
        search_button.state(["disabled"])
        dialog.configure(cursor="watch")
        try:
            found = rank_images(image, paths, settings, progress)
        except (OSError, ValueError) as error:
            if not stopped:
                dialog.configure(cursor="")
                search_button.state(["!disabled"])
                searching = False
                messagebox.showerror("Closest image in a folder", str(error), parent=dialog)
            return
        if found is not None and not stopped:
            result.append((settings, base.get(), found))
            close()

    buttons = ttk.Frame(body)
    buttons.pack(fill="x", pady=(16, 0))
    ttk.Button(buttons, text="Cancel", command=close).pack(side="right")
    search_button = ttk.Button(buttons, text="Search", command=search, default="active")
    search_button.pack(side="right", padx=(0, 6))
    dialog.bind("<Return>", search)
    dialog.bind("<Escape>", lambda _event: close())
    dialog.protocol("WM_DELETE_WINDOW", close)

    center_over(dialog, parent.winfo_toplevel())
    dialog.deiconify()
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class ImageLookupTool(ForensicsTool):
    tool_id = "image_lookup"
    title = "Closest image in a folder"
    category = "Feature extraction"
    description = "Find the image of a folder with the most ORB or SIFT matches, optionally checked with RANSAC."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        chosen = filedialog.askdirectory(title="Choose the folder to search", mustexist=True, parent=parent)
        if not chosen:
            return None
        folder = Path(chosen)
        paths = folder_images(folder, exclude=document.path)
        if not paths:
            raise ValueError(f"There are no other images in {folder.name}.")
        choice = ask_lookup(parent, document.current, folder, paths)
        if choice is None:
            return None
        settings, base, ranking = choice
        best, best_count = ranking[0]
        shown, reference_keypoints, keypoints, matches = best_match_image(document.current, best, settings, base)
        names = {path: path.relative_to(folder).as_posix() for path, _ in ranking}
        counted = "consistent matches" if settings["ransac"] else "matches"
        return ToolResult(
            image=shown,
            message=f"Closest image: {names[best]} ({best_count} {counted}).",
            details={
                "Operation": "Closest image in a folder",
                "Folder": str(folder),
                "Images compared": len(ranking),
                **matching_details(settings),
                "Closest image": names[best],
                "Keypoints (current, closest)": f"{len(reference_keypoints)}, {len(keypoints)}",
                counted.capitalize(): len(matches),
                **{f"{place}. {names[path]}": f"{count} {counted}" for place, (path, count) in enumerate(ranking, 1)},
            },
        )
