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

"""Integration tests for camera calibration module.

This script tests the camera calibration functionality by loading test images,
detecting checkerboard corners, and performing full camera calibration with
multiple views.
"""

import os
import sys

import cv2 as cv

root_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_path = os.path.join(root_dir_path, "python")
if python_path not in sys.path:
    sys.path.append(python_path)

from ttga.camera_calibration import (  # noqa: E402
    CameraCalibration,
    CalibrationView,
    DEFAULT_NUMBER_OF_SQUARES_W,
    DEFAULT_NUMBER_OF_SQUARES_H
)


def test_load_and_detect_corners():
    """Test loading calibration image and detecting checkerboard corners."""
    print("=" * 80)
    print("TEST: Load calibration image and detect corners")
    print("=" * 80)

    # Path to test image
    image_path = os.path.join(root_dir_path, "tests", "images", "calibration_top.jpg")

    print(f"\nLoading image: {image_path}")

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found at {image_path}")
        return False

    # Load image
    image = cv.imread(image_path)
    if image is None:
        print(f"ERROR: Failed to load image from {image_path}")
        return False

    print(f"Image loaded successfully: {image.shape[1]}x{image.shape[0]} pixels")

    # Create calibration object
    calibration = CameraCalibration(
        number_of_squares_w=DEFAULT_NUMBER_OF_SQUARES_W,
        number_of_squares_h=DEFAULT_NUMBER_OF_SQUARES_H
    )

    print(f"\nCheckerboard dimensions: {DEFAULT_NUMBER_OF_SQUARES_W}x{DEFAULT_NUMBER_OF_SQUARES_H} squares")
    print(f"Expected corners: {(DEFAULT_NUMBER_OF_SQUARES_W - 1) * (DEFAULT_NUMBER_OF_SQUARES_H - 1)}")

    # Attempt to create calibration frame
    print("\nAttempting to detect checkerboard corners...")
    calib_frame = calibration.make_calibration_frame(image)

    if calib_frame is None:
        print("ERROR: Failed to detect checkerboard corners")
        return False

    print(f"SUCCESS: Detected {len(calib_frame.corners)} corners")

    # Get reference points
    reference_points = calibration.get_checkerboard_3d_reference_points()

    # Flatten reference points for comparison
    reference_points_flat = [
        point
        for row in reference_points
        for point in row
    ]

    print("\n3D Reference Points (first 5):")
    for i, point in enumerate(reference_points_flat[:5]):
        print(f"  [{i}]: {point}")

    print("\n2D Detected Corners (first 5):")
    for i in range(min(5, len(calib_frame.corners))):
        corner = calib_frame.corners[i][0]  # corners is shape (N, 1, 2)
        print(f"  [{i}]: ({corner[0]:.2f}, {corner[1]:.2f})")

    # Verify counts match
    expected_count = len(reference_points_flat)
    actual_count = len(calib_frame.corners)

    print("\nCorner count verification:")
    print(f"  Expected: {expected_count}")
    print(f"  Detected: {actual_count}")

    if expected_count != actual_count:
        print("ERROR: Corner count mismatch!")
        return False

    print("SUCCESS: Corner counts match")

    # Display corner ordering information
    print("\nCorner ordering:")
    print("  Reference points are in row-major order (left-to-right, top-to-bottom)")
    print(f"  Rows: {DEFAULT_NUMBER_OF_SQUARES_H - 1}")
    print(f"  Columns: {DEFAULT_NUMBER_OF_SQUARES_W - 1}")

    # Show a few key corners to verify ordering
    corners_w = DEFAULT_NUMBER_OF_SQUARES_W - 1
    corners_h = DEFAULT_NUMBER_OF_SQUARES_H - 1

    print("\nKey corner positions:")

    # Top-left corner (0, 0)
    idx_tl = 0
    ref_tl = reference_points_flat[idx_tl]
    det_tl = calib_frame.corners[idx_tl][0]
    print(f"  Top-left [0]: ref={ref_tl}, detected=({det_tl[0]:.2f}, {det_tl[1]:.2f})")

    # Top-right corner (last column, first row)
    idx_tr = corners_w - 1
    ref_tr = reference_points_flat[idx_tr]
    det_tr = calib_frame.corners[idx_tr][0]
    print(f"  Top-right [{idx_tr}]: ref={ref_tr}, detected=({det_tr[0]:.2f}, {det_tr[1]:.2f})")

    # Bottom-left corner (first column, last row)
    idx_bl = (corners_h - 1) * corners_w
    ref_bl = reference_points_flat[idx_bl]
    det_bl = calib_frame.corners[idx_bl][0]
    print(f"  Bottom-left [{idx_bl}]: ref={ref_bl}, detected=({det_bl[0]:.2f}, {det_bl[1]:.2f})")

    # Bottom-right corner (last column, last row)
    idx_br = corners_h * corners_w - 1
    ref_br = reference_points_flat[idx_br]
    det_br = calib_frame.corners[idx_br][0]
    print(f"  Bottom-right [{idx_br}]: ref={ref_br}, detected=({det_br[0]:.2f}, {det_br[1]:.2f})")

    print("\nNote: Corner ordering depends on checkerboard orientation in the image.")
    print("OpenCV detects corners consistently regardless of rotation or flip.")

    # Create visualization with corners drawn
    print("\nCreating visualization...")
    vis_image = image.copy()
    cv.drawChessboardCorners(
        vis_image,
        (corners_w, corners_h),
        calib_frame.corners,
        True
    )

    # Save visualization
    output_path = os.path.join(root_dir_path, "tests", "images", "calibration_top_corners.jpg")
    cv.imwrite(output_path, vis_image)
    print(f"Saved visualization to: {output_path}")

    print("\n" + "=" * 80)
    print("TEST PASSED")
    print("=" * 80)

    return True


def test_full_calibration():
    """Test full camera calibration with all three views."""
    print("\n" + "=" * 80)
    print("TEST: Full camera calibration with three views")
    print("=" * 80)

    # Image paths
    image_files = [
        ("calibration_top.jpg", CalibrationView.TOP),
        ("calibration_front.jpg", CalibrationView.FRONT),
        ("calibration_side.jpg", CalibrationView.SIDE)
    ]

    # Create calibration object
    calibration = CameraCalibration(
        number_of_squares_w=DEFAULT_NUMBER_OF_SQUARES_W,
        number_of_squares_h=DEFAULT_NUMBER_OF_SQUARES_H
    )

    print(f"\nCheckerboard dimensions: {DEFAULT_NUMBER_OF_SQUARES_W}x{DEFAULT_NUMBER_OF_SQUARES_H} squares")
    print(f"Expected corners per image: {(DEFAULT_NUMBER_OF_SQUARES_W - 1) * (DEFAULT_NUMBER_OF_SQUARES_H - 1)}")

    # Load and process each image
    for filename, view in image_files:
        image_path = os.path.join(root_dir_path, "tests", "images", filename)

        print(f"\nProcessing {filename}...")

        # Check if file exists
        if not os.path.exists(image_path):
            print(f"  ERROR: Image file not found at {image_path}")
            return False

        # Load image
        image = cv.imread(image_path)
        if image is None:
            print(f"  ERROR: Failed to load image from {image_path}")
            return False

        print(f"  Loaded: {image.shape[1]}x{image.shape[0]} pixels")

        # Detect corners
        calib_frame = calibration.make_calibration_frame(image)

        if calib_frame is None:
            print("  ERROR: Failed to detect checkerboard corners")
            return False

        print(f"  SUCCESS: Detected {len(calib_frame.corners)} corners")

        # Set calibration frame
        calibration.set_calibration_frame(view, calib_frame)

        # Create and save visualization
        corners_w = DEFAULT_NUMBER_OF_SQUARES_W - 1
        corners_h = DEFAULT_NUMBER_OF_SQUARES_H - 1
        vis_image = image.copy()
        cv.drawChessboardCorners(
            vis_image,
            (corners_w, corners_h),
            calib_frame.corners,
            True
        )

        output_filename = filename.replace(".jpg", "_corners.jpg")
        output_path = os.path.join(root_dir_path, "tests", "images", output_filename)
        cv.imwrite(output_path, vis_image)
        print(f"  Saved visualization to: {output_filename}")

    # Perform calibration
    print("\nPerforming camera calibration...")
    calib_data = calibration.calibrate_camera()

    if calib_data is None:
        print("ERROR: Camera calibration failed")
        return False

    print("SUCCESS: Camera calibration completed")

    # Display calibration results
    print("\nCalibration Results:")
    print("=" * 80)

    print("\nCamera Matrix (mtx):")
    print(calib_data.mtx)

    print("\nDistortion Coefficients (dist):")
    print(calib_data.dist)

    print(f"\nMean Reprojection Error: {calib_data.mean_reprojection_error:.4f} pixels")

    print("\nRotation Vectors (rvecs):")
    for i, rvec in enumerate(calib_data.rvecs_list):
        view_name = ["TOP", "FRONT", "SIDE"][i]
        print(f"  {view_name}: {rvec.ravel()}")

    print("\nTranslation Vectors (tvecs):")
    for i, tvec in enumerate(calib_data.tvecs_list):
        view_name = ["TOP", "FRONT", "SIDE"][i]
        print(f"  {view_name}: {tvec.ravel()}")

    # Evaluate calibration quality
    print("\nCalibration Quality Assessment:")
    if calib_data.mean_reprojection_error < 0.5:
        print("  EXCELLENT: Mean error < 0.5 pixels")
    elif calib_data.mean_reprojection_error < 1.0:
        print("  GOOD: Mean error < 1.0 pixels")
    elif calib_data.mean_reprojection_error < 2.0:
        print("  ACCEPTABLE: Mean error < 2.0 pixels")
    else:
        print("  POOR: Mean error >= 2.0 pixels (consider recalibrating)")

    print("\n" + "=" * 80)
    print("TEST PASSED")
    print("=" * 80)

    return True


def main():
    """Main entry point for the test."""
    print("\nCamera Calibration Unit Tests")
    print("=" * 80)

    # Run first test
    success1 = test_load_and_detect_corners()

    if not success1:
        print("\nTests failed!")
        return 1

    # Run full calibration test
    success2 = test_full_calibration()

    if success2:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nTests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
