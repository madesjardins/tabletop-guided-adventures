# Copyright 2026 Marc-Antoine Desjardins
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to generate pages of 70 MicroQR codes to print.

Requires template images in images/MicroQR folder:
    - MicroQr_Int_Small_Base_Template_Page.png
    - MicroQr_Int_No_Base_Template_Page.png
    - Letter_300dpi.png

Example usage:
    python generate_micro_qr_code.py 0 --small_base
    python generate_micro_qr_code.py 1000
"""

import os
import argparse

import pyboof as pb
from pyboof import pbg
from PySide6 import QtGui

PROJECT_ROOT_DIR_PATH = os.path.dirname(os.path.dirname(__file__))

MICRO_QR_INT_FILE_PATH_TEMPLATE = PROJECT_ROOT_DIR_PATH + "/images/MicroQR/_tmp/MicroQr_Int_s{size:02d}_{value:04d}.png"
MICRO_QR_INT_PAGE_FILE_PATH_TEMPLATE = PROJECT_ROOT_DIR_PATH + "/images/MicroQR/MicroQr_Int_s{size:02d}_Page_{value_min:04d}_{value_max:04d}.png"
MICRO_QR_INT_SMALL_BASE_TEMPLATE_PAGE_FILE_PATH = PROJECT_ROOT_DIR_PATH + "/images/MicroQR/MicroQr_Int_Small_Base_Template_Page.png"
MICRO_QR_INT_NO_BASE_TEMPLATE_PAGE_FILE_PATH = PROJECT_ROOT_DIR_PATH + "/images/MicroQR/MicroQr_Int_No_Base_Template_Page.png"
LETTER_PAGE_300DPI_FILE_PATH = PROJECT_ROOT_DIR_PATH + "/images/MicroQR/Letter_300dpi.png"


def generate_micro_qr_int(size: int, value_min: int, value_max: int, file_path_template: str) -> list[str]:
    """Generate MicroQR code in a range of values.

    Args:
        size: The number of pixels_per_module.
        value_min: The lowest MicroQR number to generate.
        value_max: The highest MicroQR number to generate.
        file_path_template: The file path template for the generated images.
            Example: "D:/Dev_Projects/project/images/MicroQR/_tmp/MicroQr_Int_s{size:02d}_{value:04d}.png"

    Returns:
        The list of image file paths.
    """
    image_file_paths_list = []
    for value in range(value_min, value_max + 1):
        print(f"Generating: {value}")
        generator = pb.MicroQrCodeGenerator(pixels_per_module=size)
        generator.set_message(value)
        boof_gray_image = generator.generate()
        file_path = file_path_template.format(size=size, value=value)
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        elif os.path.exists(file_path):
            os.remove(file_path)
        pbg.gateway.jvm.boofcv.io.image.UtilImageIO.saveImage(boof_gray_image, file_path)
        image_file_paths_list.append(file_path)

    return image_file_paths_list


def composite_images(image_base: QtGui.QImage, image_overlay: QtGui.QImage, overlay_x: int = 0, overlay_y: int = 0) -> QtGui.QImage:
    """Composite 2 images using over mode.

    Args:
        image_base: The base image.
        image_overlay: The image to composite over the base image.
        overlay_x: Horizontal overlay offset in pixels.
        overlay_y: Vertical overlay offset in pixels.

    Returns:
        Composite image.
    """
    image_result = QtGui.QImage(image_base.size(), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    painter = QtGui.QPainter(image_result)

    painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Source)
    painter.drawImage(0, 0, image_base)

    painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.drawImage(overlay_x, overlay_y, image_overlay)

    painter.end()

    return image_result


def compose_micro_qr_int_page(
    template_page_image_file_path: str,
    qr_image_file_paths_list: list[str],
    column_count: int,
    row_count: int,
    page_image_file_path: str,
    printable_page_image_file_path: str | None = None
) -> str:
    """Merge the MicroQR code images on top of a page template ready for print.

    Args:
        template_page_image_file_path: The template page image file path.
            Example: "D:/Dev_Projects/project/images/MicroQR/MicroQr_Int_Small_Base_Template_Page.png"
        qr_image_file_paths_list: The list of MicroQR image filepaths.
        column_count: The number of columns in the template page.
        row_count: The number of rows in the template page.
        page_image_file_path: The output image file path.
            Example: "D:/Dev_Projects/project/images/MicroQR/MicroQr_Int_s16_Page_0000_0069.png"
        printable_page_image_file_path: If set, will composite the page result in the middle of this image.
            Example: "D:/Dev_Projects/project/images/MicroQR/Letter_300dpi.png"

    Returns:
        The output image file path.
    """
    template_image = QtGui.QImage(template_page_image_file_path)
    width = template_image.width()
    offset_step_f = width / column_count
    offset_base_f = offset_step_f / 2

    for row in range(row_count):
        for column in range(column_count):
            index = row * column_count + column
            qr_image = QtGui.QImage(qr_image_file_paths_list[index])
            template_image = composite_images(
                template_image,
                qr_image,
                overlay_x=int(offset_base_f + offset_step_f * column - qr_image.width() / 2 + 0.5),
                overlay_y=int(offset_base_f + offset_step_f * row - qr_image.height() / 2 + 0.5)
            )

    if printable_page_image_file_path is not None:
        printable_image = QtGui.QImage(printable_page_image_file_path)
        template_image = composite_images(
            printable_image,
            template_image,
            overlay_x=int((printable_image.width() - template_image.width()) / 2 + 0.5),
            overlay_y=int((printable_image.height() - template_image.height()) / 2 + 0.5)
        )

    print(f"Saving page result to '{page_image_file_path}'.")
    if not os.path.exists(os.path.dirname(page_image_file_path)):
        os.makedirs(os.path.dirname(page_image_file_path))
    elif os.path.exists(page_image_file_path):
        os.remove(page_image_file_path)
    template_image.save(page_image_file_path)

    return page_image_file_path


def generate_micro_qr_int_page(column_count: int, row_count: int, value_min: int, size: int, with_small_base_circle: bool = False) -> str:
    """Generate a whole page of MicroQR code starting from a certain value.

    Args:
        column_count: The number of columns in the template page.
        row_count: The number of rows in the template page.
        value_min: The lowest MicroQR number to generate.
        size: The number of pixels_per_module.
        with_small_base_circle: If set to True, will circle in light gray the MicroQR as hint for cut.

    Returns:
        The output image file path.
    """
    value_max = value_min + column_count * row_count - 1
    micro_qr_image_file_paths_list = generate_micro_qr_int(
        size=size,
        value_min=value_min,
        value_max=value_max,
        file_path_template=MICRO_QR_INT_FILE_PATH_TEMPLATE
    )

    template_page_image_file_path = MICRO_QR_INT_SMALL_BASE_TEMPLATE_PAGE_FILE_PATH if with_small_base_circle else MICRO_QR_INT_NO_BASE_TEMPLATE_PAGE_FILE_PATH

    output_file_path = compose_micro_qr_int_page(
        template_page_image_file_path=template_page_image_file_path,
        qr_image_file_paths_list=micro_qr_image_file_paths_list,
        column_count=column_count,
        row_count=row_count,
        page_image_file_path=MICRO_QR_INT_PAGE_FILE_PATH_TEMPLATE.format(size=size, value_min=value_min, value_max=value_max),
        printable_page_image_file_path=LETTER_PAGE_300DPI_FILE_PATH
    )

    print("Deleting temporary files...")
    for qr_image_file_path in micro_qr_image_file_paths_list:
        os.remove(qr_image_file_path)

    return output_file_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("min_value", type=int, help="MicroQR minimum value")
    parser.add_argument("-sb", "--small_base", action="store_true", help="set to have small base circle around MicroQR")
    args = parser.parse_args()

    image_file_path = generate_micro_qr_int_page(
        column_count=7,
        row_count=10,
        value_min=args.min_value,
        size=16,
        with_small_base_circle=args.small_base
    )
