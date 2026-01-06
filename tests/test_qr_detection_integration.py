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

"""Integration test for QR detection module.

This script tests the QRDetector class with a sample image containing MicroQR codes,
detects them, and saves an output image with detection results drawn.
"""

import os
import sys

import cv2 as cv

root_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_path = os.path.join(root_dir_path, "python")
if python_path not in sys.path:
    sys.path.append(python_path)

from ttga.qr_detection import QRDetector, draw_qr_detections  # noqa: E402


def test_qr_detection() -> None:
    """Test QR detection with a sample image."""
    # Define paths
    test_image_path = os.path.join(root_dir_path, "tests", "images", "qr_detection.jpg")
    output_image_path = os.path.join(root_dir_path, "tests", "images", "qr_detection_detected.jpg")

    # Check if test image exists
    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image not found at {test_image_path}")
        print("Please download test images from the Google Drive link.")
        print("See tests/images/_add_test_images_here.txt for instructions.")
        return

    # Load test image
    print(f"Loading test image: {test_image_path}")
    image = cv.imread(test_image_path)

    if image is None:
        print(f"ERROR: Failed to load image from {test_image_path}")
        return

    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

    # Create detector
    print("Creating QR detector...")
    detector = QRDetector()

    # Detect QR codes
    print("Detecting MicroQR codes...")
    detections = detector.detect(image)

    print(f"\nFound {len(detections)} QR code(s):")
    for i, detection in enumerate(detections, 1):
        print(f"\n  QR Code #{i}:")
        print(f"    Message: {detection.message}")

        # Try to parse as integer
        try:
            value = int(detection.message)
            print(f"    Integer value: {value}")
        except ValueError:
            print("    (Not an integer)")

        print(f"    Corners: {detection.corners}")
        print(f"    Bounds: x={detection.bounds[0]}, y={detection.bounds[1]}, "
              f"w={detection.bounds[2]}, h={detection.bounds[3]}")

    # Draw detections on image
    print("\nDrawing detections on image...")
    result_image = draw_qr_detections(image, detections)

    # Save result
    print(f"Saving result to: {output_image_path}")
    cv.imwrite(output_image_path, result_image)

    print("\n✓ QR detection test completed successfully!")
    print(f"  Input:  {test_image_path}")
    print(f"  Output: {output_image_path}")


def main() -> None:
    """Main entry point for the test."""
    print("=" * 60)
    print("QR Detection Integration Test")
    print("=" * 60)
    print()

    test_qr_detection()


if __name__ == "__main__":
    main()
