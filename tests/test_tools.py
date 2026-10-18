import unittest

import numpy as np
from PIL import Image
from scipy import ndimage

from forensics_app.core import ImageDocument
from forensics_app.tools.convolution import (
    _is_integer_prefix,
    _is_kernel_size_prefix,
    _toggle_sign,
    _without_leading_zeros,
    convolve,
    sobel,
)
from forensics_app.tools.edge_detection import VIEWS, detect_edges, show_edges
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.histogram import histogram_image
from forensics_app.tools.hog import VIEWS as HOG_VIEWS, extract_hog, show_hog
from forensics_app.tools.person_detection import detect_person, show_detection
from forensics_app.tools.registry import ToolRegistry
from forensics_app.tools.split_channel import SplitChannelTool
from forensics_app.tools.swap_channels import swap_channels


class ToolTests(unittest.TestCase):
    def test_grayscale_returns_image_without_mutating_document(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        result = GrayscaleTool().run(None, document)  # parent is unused by this tool
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(document.current.mode, "RGB")

    def test_split_channel_returns_selected_band(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (2, 2), (10, 20, 30))
        result = SplitChannelTool("G").run(None, document)
        self.assertEqual(result.image.getpixel((0, 0)), 20)

    def test_swap_channels_swaps_two_bands(self) -> None:
        image = Image.new("RGB", (2, 2), (10, 20, 30))
        self.assertEqual(swap_channels(image, "R", "B").getpixel((0, 0)), (30, 20, 10))
        self.assertEqual(swap_channels(image, "G", "R").getpixel((0, 0)), (20, 10, 30))

    def test_histogram_image_is_a_plot(self) -> None:
        plot = histogram_image(Image.new("RGB", (4, 4), (10, 20, 30)))
        self.assertEqual(plot.mode, "RGB")

    def test_channel_button_on_histogram_redraws_that_channel(self) -> None:
        source = Image.new("RGB", (4, 4), (10, 20, 30))
        document = ImageDocument()
        document.current = source
        document.apply(histogram_image(source))
        result = SplitChannelTool("R").run(None, document)
        self.assertIs(result.image.info["histogram_source"], source)

    def test_convolution_flips_the_kernel(self) -> None:
        image = Image.frombytes("L", (3, 1), bytes([10, 20, 30]))
        left_cell = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        # convolution: out[x] = in[x + 1]; correlation (no flip) would give in[x - 1] = [10, 10, 20]
        self.assertEqual(list(convolve(image, left_cell).getdata()), [20, 30, 30])

    def test_convolution_output_modes_handle_negative_responses(self) -> None:
        step = Image.frombytes("L", (4, 1), bytes([0, 0, 10, 10]))
        horizontal = sobel("Horizontal")
        self.assertEqual(list(convolve(step, horizontal, "Absolute value").getdata()), [0, 40, 40, 0])
        self.assertEqual(list(convolve(step, horizontal, "Clip to 0–255").getdata()), [0, 0, 0, 0])
        self.assertEqual(list(convolve(step, sobel("Vertical")).getdata()), [0, 0, 0, 0])

    def test_convolution_colour_modes_return_one_channel(self) -> None:
        image = Image.new("RGB", (3, 3), (10, 20, 30))
        identity = np.zeros((3, 3))
        identity[1, 1] = 1
        self.assertEqual(convolve(image, identity, channels="Mean of channels").getpixel((1, 1)), 20)
        self.assertEqual(convolve(image, identity, channels="Grayscale").getpixel((1, 1)), 18)  # luminance

    def test_channel_mean_finds_edges_between_equally_bright_colours(self) -> None:
        image = Image.new("RGB", (4, 1), (0, 100, 0))
        image.paste((196, 0, 0), (2, 0, 4, 1))  # same luminance as the green half
        horizontal = sobel("Horizontal")
        self.assertEqual(convolve(image, horizontal, channels="Grayscale").getpixel((1, 0)), 0)
        self.assertEqual(convolve(image, horizontal, channels="Mean of channels").getpixel((1, 0)), 170)

    def test_normalized_ones_kernel_is_a_mean_blur(self) -> None:
        image = Image.frombytes("L", (5, 1), bytes([0, 0, 90, 0, 0]))
        ones = np.ones((3, 3))
        self.assertEqual(list(convolve(image, ones, normalize=True).getdata()), [0, 30, 30, 30, 0])
        self.assertEqual(list(convolve(image, ones).getdata()), [0, 255, 255, 255, 0])  # sum of 3 rows of 90
        flat = Image.new("RGB", (4, 4), (40, 80, 120))
        self.assertEqual(convolve(flat, np.ones((7, 7)), normalize=True).getpixel((2, 2)), 80)

    def test_normalize_scales_zero_sum_kernels_by_their_positive_weights(self) -> None:
        step = Image.frombytes("L", (4, 1), bytes([0, 0, 10, 10]))
        self.assertEqual(list(convolve(step, sobel("Horizontal"), normalize=True).getdata()), [0, 10, 10, 0])  # 40 / 4
        edge = np.zeros((40, 40), dtype=np.uint8)
        edge[:, 20:] = 255
        for size in (3, 7, 15):  # a full black-to-white edge peaks at exactly 255, whatever the size
            result = np.asarray(convolve(Image.fromarray(edge), sobel("Horizontal", size), normalize=True))
            self.assertEqual(int(result.max()), 255, size)
            self.assertEqual(int(result[:, 0].max()), 0, size)  # flat areas stay black
        self.assertEqual(list(convolve(step, np.zeros((3, 3)), normalize=True).getdata()), [0, 0, 0, 0])

    def test_slide_box_blurs_with_rectangular_kernels(self) -> None:
        # Set 3.1: np.ones((1, 15)) / 15, its transpose, and np.ones((15, 15)) / 225
        rng = np.random.default_rng(0)
        pixels = rng.integers(0, 256, (40, 50)).astype(np.uint8)
        image = Image.fromarray(pixels)
        for shape in ((1, 15), (15, 1), (15, 15)):
            expected = ndimage.convolve(pixels.astype(np.float64), np.ones(shape) / np.prod(shape), mode="nearest")
            result = np.asarray(convolve(image, np.ones(shape), normalize=True))
            self.assertTrue(np.array_equal(result, np.rint(expected).astype(np.uint8)), shape)

    def test_even_sized_kernels(self) -> None:
        rng = np.random.default_rng(1)
        pixels = rng.integers(0, 256, (20, 30)).astype(np.uint8)
        for shape in ((2, 2), (1, 4), (6, 3)):
            expected = ndimage.convolve(pixels.astype(np.float64), np.ones(shape) / np.prod(shape), mode="nearest")
            result = np.asarray(convolve(Image.fromarray(pixels), np.ones(shape), normalize=True))
            self.assertTrue(np.array_equal(result, np.rint(expected).astype(np.uint8)), shape)

    def test_sobel_kernels_match_opencv(self) -> None:
        self.assertEqual(sobel("Horizontal", 3).tolist(), [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.assertEqual(sobel("Vertical", 3).tolist(), [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.assertEqual(sobel("Horizontal", 5)[2].tolist(), [-6, -12, 0, 12, 6])
        self.assertEqual(sobel("Horizontal", 7)[0].tolist(), [-1, -4, -5, 0, 5, 4, 1])
        self.assertTrue(np.array_equal(sobel("Vertical", 5), sobel("Horizontal", 5).T))

    def test_convolution_accepts_larger_kernels(self) -> None:
        image = Image.frombytes("L", (5, 1), bytes([10, 20, 30, 40, 50]))
        two_left = np.zeros((5, 5))
        two_left[2, 0] = 1  # convolution: out[x] = in[x + 2]
        self.assertEqual(list(convolve(image, two_left).getdata()), [30, 40, 50, 50, 50])

    def test_kernel_cells_accept_only_integers(self) -> None:
        for text in ("", "-", "0", "7", "-12", "100"):
            self.assertTrue(_is_integer_prefix(text), text)
        for text in ("a", "1a", "0.5", "1.", "--1", "1-", "+1", " 1", "1e3"):
            self.assertFalse(_is_integer_prefix(text), text)

    def test_kernel_size_boxes_accept_only_1_to_15(self) -> None:
        for text in ("", "1", "2", "9", "10", "15"):
            self.assertTrue(_is_kernel_size_prefix(text), text)
        for text in ("0", "01", "16", "99", "100", "-1", "a", "1.5", " 3"):
            self.assertFalse(_is_kernel_size_prefix(text), text)

    def test_kernel_cells_drop_leading_zeros(self) -> None:
        cases = {"00": "0", "01": "1", "007": "7", "-05": "-5", "-0": "-", "0": "0", "100": "100", "-": "-", "": ""}
        for text, expected in cases.items():
            self.assertEqual(_without_leading_zeros(text), expected, text)

    def test_minus_key_toggles_the_sign(self) -> None:
        cases = {"3": "-3", "-3": "3", "12": "-12", "0": "-", "": "-", "-": ""}
        for text, expected in cases.items():
            self.assertEqual(_toggle_sign(text), expected, text)

    def test_canny_finds_the_border_of_a_square(self) -> None:
        image = Image.new("L", (40, 40), 0)
        image.paste(255, (10, 10, 30, 30))
        edges = detect_edges(image)
        self.assertTrue(edges[20, 9:12].any())  # left side of the square
        self.assertFalse(edges[20, 15:25].any())  # flat inside
        self.assertGreater(detect_edges(image, sigma=0.5).sum(), 0)
        with self.assertRaises(ValueError):
            detect_edges(image, low=0.3, high=0.2)

    def test_edge_views(self) -> None:
        image = Image.new("RGB", (4, 1), (100, 100, 100))
        edges = np.array([[False, True, False, False]])
        self.assertEqual(show_edges(image, edges).mode, "L")
        self.assertEqual(list(show_edges(image, edges).getdata()), [0, 255, 0, 0])
        red = show_edges(image, edges, "Coloured edges")
        self.assertEqual(red.getpixel((1, 0)), (255, 0, 0))
        self.assertEqual(red.getpixel((0, 0)), (100, 100, 100))
        overlay = show_edges(image, edges, "Transparent overlay")
        self.assertEqual(overlay.getpixel((1, 0)), (178, 50, 50))
        self.assertEqual(overlay.getpixel((0, 0)), (50, 50, 50))
        dim = show_edges(image, edges, "Bright edges, dim image")
        self.assertEqual(dim.getpixel((1, 0)), (255, 255, 255))
        self.assertEqual(dim.getpixel((0, 0)), (40, 40, 40))
        for view in VIEWS[1:]:
            self.assertEqual(show_edges(image, edges, view).size, image.size, view)
        self.assertEqual(show_edges(image, edges, "Coloured edges", colour="Blue").getpixel((1, 0)), (0, 0, 255))
        self.assertEqual(show_edges(image, edges, "Transparent overlay", colour="Green", opacity=1).getpixel((1, 0)),
                         (0, 255, 0))
        self.assertEqual(show_edges(image, edges, "Bright edges, dim image", brightness=0).getpixel((0, 0)), (0, 0, 0))
        square = np.zeros((9, 9), dtype=bool)
        square[4, 2:7] = True
        contour = show_edges(Image.new("L", (9, 9), 100), square, "Contour overlay")
        self.assertEqual(contour.getpixel((4, 3)), (255, 0, 0))  # outline just above the edge line
        self.assertEqual(contour.getpixel((0, 0)), (100, 100, 100))

    def test_edges_are_drawn_on_the_colour_image_by_default(self) -> None:
        image = Image.new("RGB", (3, 1), (200, 100, 0))
        edges = np.array([[False, True, False]])
        self.assertEqual(show_edges(image, edges, "Coloured edges", colour="Blue").getpixel((0, 0)), (200, 100, 0))
        self.assertEqual(show_edges(image, edges, "Coloured edges", colour="Blue").getpixel((1, 0)), (0, 0, 255))
        gray = show_edges(image, edges, "Coloured edges", base="Grayscale").getpixel((0, 0))
        self.assertEqual(gray, (118, 118, 118))  # luminance of the orange: 0.299 × 200 + 0.587 × 100
        self.assertEqual(show_edges(image, edges, "Bright edges, dim image").getpixel((0, 0)), (80, 40, 0))

    def test_edge_thickness_widens_edges_but_not_the_edge_map(self) -> None:
        image = Image.new("L", (9, 9), 100)
        edges = np.zeros((9, 9), dtype=bool)
        edges[4, 4] = True
        thick = np.asarray(show_edges(image, edges, "Coloured edges", colour="Green", thickness=3))
        self.assertEqual(int((thick == (0, 255, 0)).all(axis=2).sum()), 9)  # a 3 × 3 block
        self.assertEqual(int(np.asarray(show_edges(image, edges, thickness=3)).astype(bool).sum()), 1)
        wide = np.asarray(show_edges(image, edges, "Contour overlay", thickness=1)).copy()
        wider = np.asarray(show_edges(image, edges, "Contour overlay", thickness=4))
        self.assertGreater(int((wider == (255, 0, 0)).all(axis=2).sum()), int((wide == (255, 0, 0)).all(axis=2).sum()))

    def test_hog_descriptor_length_follows_cells_and_blocks(self) -> None:
        image = Image.new("RGB", (48, 64), (100, 100, 100))
        image.paste((255, 255, 255), (16, 16, 32, 48))
        descriptor, hog_image = extract_hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        self.assertEqual(descriptor.size, (8 - 1) * (6 - 1) * 2 * 2 * 9)  # blocks down × across × cells × bins
        self.assertEqual(hog_image.shape, (64, 48))
        self.assertGreater(hog_image.max(), 0)
        gray, _ = extract_hog(image, 4, (8, 8), (2, 2), gradients="Grayscale")
        self.assertEqual(gray.size, 7 * 5 * 2 * 2 * 4)
        tall, _ = extract_hog(image, 9, (16, 8), (1, 3))  # 4 × 6 cells of 16 × 8 px, blocks 1 cell high, 3 wide
        self.assertEqual(tall.size, (4 - 0) * (6 - 2) * 1 * 3 * 9)
        with self.assertRaises(ValueError):
            extract_hog(image, 9, (16, 16), (4, 4))  # a 64 px block in a 48 px wide image
        with self.assertRaises(ValueError):
            extract_hog(image, 9, (8, 16), (2, 4))  # 64 px wide again, though only 16 px high

    def test_hog_views(self) -> None:
        image = Image.new("RGB", (2, 1), (100, 100, 100))
        hog_image = np.array([[0.0, 0.5]])
        self.assertEqual(list(show_hog(image, hog_image).getdata()), [0, 255])
        self.assertEqual(list(show_hog(image, np.zeros((1, 2))).getdata()), [0, 0])
        over = show_hog(image, hog_image, "Over the image", brightness=0.5)
        self.assertEqual(over.getpixel((0, 0)), (50, 50, 50))
        self.assertEqual(over.getpixel((1, 0)), (255, 217, 0))
        self.assertEqual(show_hog(image, hog_image, "Side by side").size, (4, 1))
        for view in HOG_VIEWS:
            self.assertEqual(show_hog(image, hog_image, view).height, 1, view)

    def test_person_detection_finds_the_template_in_a_flat_scene(self) -> None:
        template = Image.new("RGB", (16, 32), (60, 60, 60))
        template.paste((230, 230, 230), (4, 4, 12, 28))
        scene = Image.new("RGB", (64, 48), (120, 120, 120))
        scene.paste(template, (37, 11))  # x, y: the last possible column, so the scan must reach it
        detection = detect_person(scene, template, orientations=9, step=1)
        self.assertEqual(detection.best_euclidean, (11, 37))  # (row, column)
        self.assertEqual(detection.best_dot, (11, 37))
        self.assertEqual(detection.windows, (48 - 32 + 1) * (64 - 16 + 1))
        self.assertAlmostEqual(detection.euclidean.min(), 0.0)
        coarse = detect_person(scene, template, orientations=9, step=4)
        self.assertEqual((coarse.rows.size, coarse.cols.size), (5, 13))
        with self.assertRaises(ValueError):
            detect_person(template, scene)  # the template is larger than the image
        with self.assertRaises(ValueError):
            detect_person(scene, template, step=0)

    def test_person_detection_views(self) -> None:
        template = Image.new("RGB", (16, 32), (60, 60, 60))
        template.paste((230, 230, 230), (4, 4, 12, 28))
        scene = Image.new("RGB", (40, 40), (120, 120, 120))
        scene.paste(template, (10, 5))
        detection = detect_person(scene, template, orientations=9, step=1)
        boxed = show_detection(scene, detection, "Euclidean distance")
        self.assertEqual(boxed.size, scene.size)
        self.assertEqual(boxed.getpixel((10, 5)), (255, 0, 0))  # top-left corner of the box
        self.assertEqual(boxed.getpixel((25, 36)), (255, 0, 0))  # bottom-right corner
        self.assertEqual(boxed.getpixel((0, 0)), (120, 120, 120))
        self.assertEqual(show_detection(scene, detection, "Dot product").getpixel((10, 5)), (255, 0, 0))
        side = show_detection(scene, detection)
        title_height = side.height - 40
        self.assertGreater(title_height, 0)  # titles below the panels
        self.assertEqual(side.width, 3 * 40 + 2 * 8)
        self.assertEqual(side.getpixel((10, 5)), (60, 60, 60))  # the first panel has no box
        self.assertEqual(side.getpixel((48 + 10, 5)), (255, 0, 0))  # the second one has
        self.assertTrue((np.asarray(side)[40:] < 128).any())  # some dark title text

    def test_registry_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            ToolRegistry([GrayscaleTool(), GrayscaleTool()])


if __name__ == "__main__":
    unittest.main()
