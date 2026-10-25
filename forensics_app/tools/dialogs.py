"""Small helpers shared by the tools' dialog windows."""

from __future__ import annotations

import tkinter as tk

from PIL import Image


# Above this many pixels on the longest side, slow tools preview on a smaller copy and only Apply
# runs on the full image, so the dialog stays responsive while the user changes settings.
PREVIEW_MAX_SIDE = 1200


def preview_copy(image: Image.Image, max_side: int = PREVIEW_MAX_SIDE) -> tuple[Image.Image, float]:
    """``image`` shrunk so its longest side is at most ``max_side``, with the factor back to full size."""
    factor = max(image.size) / max_side
    if factor <= 1:
        return image, 1.0
    size = (max(1, round(image.width / factor)), max(1, round(image.height / factor)))
    return image.resize(size, Image.Resampling.LANCZOS), image.width / size[0]


def center_over(dialog: tk.Toplevel, window: tk.Misc) -> None:
    """Place ``dialog`` so its centre matches the centre of ``window``."""
    dialog.update_idletasks()
    x = window.winfo_rootx() + (window.winfo_width() - dialog.winfo_reqwidth()) // 2
    title_bar = window.winfo_rooty() - window.winfo_y()  # "+x+y" places the frame, not the content
    y = window.winfo_rooty() + (window.winfo_height() - dialog.winfo_reqheight()) // 2 - title_bar
    dialog.geometry(f"+{max(x, 0)}+{max(y, 0)}")
