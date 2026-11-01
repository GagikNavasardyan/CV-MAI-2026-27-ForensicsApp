"""Register course functionality here so it appears in the sidebar."""

from .contrast_stretching import ContrastStretchingTool
from .convolution import ConvolutionTool
from .corner_detection import CornerDetectionTool
from .descriptor_matching import DescriptorMatchingTool
from .edge_detection import EdgeDetectionTool
from .eigenfaces import EigenfacesTool
from .face_detection import FaceDetectionTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .hog import HogTool
from .image_filters import ImageFiltersTool
from .image_info import ImageInfoTool
from .image_lookup import ImageLookupTool
from .keypoints import KeypointsTool
from .masking import MaskingTool
from .person_detection import PersonDetectionTool
from .registry import ToolRegistry
from .sharpening import SharpeningTool
from .split_channel import SplitChannelTool
from .swap_channels import SwapChannelsTool
from .template_matching import TemplateMatchingTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            *(SplitChannelTool(channel) for channel in "RGB"),
            SwapChannelsTool(),
            MaskingTool(),
            HistogramTool(),
            ContrastStretchingTool(),
            ConvolutionTool(),
            ImageFiltersTool(),
            EdgeDetectionTool(),
            SharpeningTool(),
            TemplateMatchingTool(),
            HogTool(),
            PersonDetectionTool(),
            CornerDetectionTool(),
            KeypointsTool(),
            DescriptorMatchingTool(),
            ImageLookupTool(),
            FaceDetectionTool(),
            EigenfacesTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
