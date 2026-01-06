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

"""Camera calibration module for computing camera intrinsic and extrinsic parameters.

This module provides classes and utilities for calibrating cameras using checkerboard
patterns to determine camera matrix, distortion coefficients, and pose information.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import cv2 as cv
import numpy as np


DEFAULT_NUMBER_OF_SQUARES_W = 23
DEFAULT_NUMBER_OF_SQUARES_H = 18


class CalibrationView(IntEnum):
    """Enumeration for calibration frame views."""
    TOP = 0
    FRONT = 1
    SIDE = 2


@dataclass
class CalibrationFrame:
    """Data class representing a single calibration frame.

    Attributes:
        number_of_squares_w: Number of squares in checkerboard width.
        number_of_squares_h: Number of squares in checkerboard height.
        image: The captured image as numpy array.
        corners: List of detected checkerboard corner coordinates.
    """
    number_of_squares_w: int
    number_of_squares_h: int
    image: np.ndarray
    corners: np.ndarray


@dataclass
class CameraCalibrationData:
    """Data class containing camera calibration results.

    Attributes:
        mtx: Camera matrix (intrinsic parameters).
        dist: Distortion coefficients.
        rvecs_list: List of rotation vectors for each calibration frame.
        tvecs_list: List of translation vectors for each calibration frame.
        mean_reprojection_error: Mean reprojection error for this data.
    """
    mtx: np.ndarray
    dist: np.ndarray
    rvecs_list: list[np.ndarray]
    tvecs_list: list[np.ndarray]
    mean_reprojection_error: float


class CalibrationResult:
    """Data class containing calibration result information."""
    pass


class CameraCalibration:
    """Camera calibration using checkerboard pattern detection.

    This class manages the calibration process by collecting multiple views of a
    checkerboard pattern and computing camera parameters.

    Attributes:
        number_of_squares_w: Number of squares in checkerboard width.
        number_of_squares_h: Number of squares in checkerboard height.

    Example:
        >>> calibration = CameraCalibration(number_of_squares_w=9, number_of_squares_h=6)
        >>> frame = calibration.make_calibration_frame(captured_image)
        >>> if frame is not None:
        ...     calibration.set_calibration_frame(CalibrationView.TOP, frame)
        >>> # ... collect FRONT and SIDE frames ...
        >>> calib_data = calibration.calibrate_camera()
    """

    def __init__(
        self,
        number_of_squares_w: int = DEFAULT_NUMBER_OF_SQUARES_W,
        number_of_squares_h: int = DEFAULT_NUMBER_OF_SQUARES_H,
    ) -> None:
        """Initialize the camera calibration.

        Args:
            number_of_squares_w: Number of squares in checkerboard width (default: DEFAULT_NUMBER_OF_SQUARES_W).
            number_of_squares_h: Number of squares in checkerboard height (default: DEFAULT_NUMBER_OF_SQUARES_H).
        """
        self._number_of_squares_w: int = number_of_squares_w
        self._number_of_squares_h: int = number_of_squares_h
        self._calibration_frames: list[Optional[CalibrationFrame]] = [None, None, None]

    @property
    def number_of_squares_w(self) -> int:
        """Get the number of squares in checkerboard width."""
        return self._number_of_squares_w

    @number_of_squares_w.setter
    def number_of_squares_w(self, value: int) -> None:
        """Set the number of squares in checkerboard width and reset frames.

        Args:
            value: New number of squares in width.
        """
        self._number_of_squares_w = value
        self._calibration_frames = [None, None, None]

    @property
    def number_of_squares_h(self) -> int:
        """Get the number of squares in checkerboard height."""
        return self._number_of_squares_h

    @number_of_squares_h.setter
    def number_of_squares_h(self, value: int) -> None:
        """Set the number of squares in checkerboard height and reset frames.

        Args:
            value: New number of squares in height.
        """
        self._number_of_squares_h = value
        self._calibration_frames = [None, None, None]

    def get_checkerboard_3d_reference_points(self) -> list[list[tuple[float, float, float]]]:
        """Get 3D reference points for checkerboard corners.

        Returns a row-major matrix of square corner intersection coordinates where
        X and Y correspond to column and negative row indices, and Z is always 0.0.

        Returns:
            List of rows, where each row contains tuples of (x, y, z) coordinates.

        Example:
            >>> calibration = CameraCalibration(number_of_squares_w=3, number_of_squares_h=3)
            >>> points = calibration.get_checkerboard_3d_reference_points()
            >>> # Returns 2x2 grid of corner intersections:
            >>> # [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            >>> #  [(0.0, -1.0, 0.0), (1.0, -1.0, 0.0)]]
        """
        points: list[list[tuple[float, float, float]]] = []

        # Number of internal corners (intersections) is one less than number of squares
        corners_h = self._number_of_squares_h - 1
        corners_w = self._number_of_squares_w - 1

        for row in range(corners_h):
            row_points: list[tuple[float, float, float]] = []
            for col in range(corners_w):
                row_points.append((float(col), -float(row), 0.0))
            points.append(row_points)

        return points

    def make_calibration_frame(self, captured_frame: np.ndarray) -> Optional[CalibrationFrame]:
        """Create a calibration frame from a captured image.

        Converts the image to grayscale and attempts to detect checkerboard corners.

        Args:
            captured_frame: The captured image as numpy array (BGR format).

        Returns:
            CalibrationFrame if corners were detected, None otherwise.

        Example:
            >>> frame = calibration.make_calibration_frame(camera_image)
            >>> if frame is not None:
            ...     print(f"Found {len(frame.corners)} corners")
        """
        # Convert to grayscale
        gray_frame = cv.cvtColor(captured_frame, cv.COLOR_BGR2GRAY)

        # Checkerboard dimensions (internal corners)
        checkerboard_dim = (
            self._number_of_squares_w - 1,
            self._number_of_squares_h - 1
        )

        # Find checkerboard corners
        ret, corners = cv.findChessboardCorners(
            gray_frame,
            checkerboard_dim,
            flags=cv.CALIB_CB_FAST_CHECK
        )

        if not ret or corners is None:
            return None

        return CalibrationFrame(
            number_of_squares_w=self._number_of_squares_w,
            number_of_squares_h=self._number_of_squares_h,
            image=gray_frame,
            corners=corners
        )

    def set_calibration_frame(
        self,
        view: CalibrationView,
        frame: CalibrationFrame
    ) -> None:
        """Set a calibration frame for a specific view.

        Args:
            view: The calibration view (TOP, FRONT, or SIDE).
            frame: The calibration frame to store.

        Example:
            >>> calibration.set_calibration_frame(CalibrationView.TOP, top_frame)
        """
        self._calibration_frames[view] = frame

    def get_calibration_frame(self, view: CalibrationView) -> Optional[CalibrationFrame]:
        """Get a calibration frame for a specific view.

        Args:
            view: The calibration view (TOP, FRONT, or SIDE).

        Returns:
            The calibration frame if set, None otherwise.
        """
        return self._calibration_frames[view]

    def calibrate_camera(self) -> Optional[CameraCalibrationData]:
        """Calibrate the camera using collected calibration frames.

        Requires all three calibration frames (TOP, FRONT, SIDE) to be set with
        matching checkerboard dimensions. Uses corner subpixel refinement for
        improved accuracy.

        Returns:
            CameraCalibrationData if calibration succeeded, None otherwise.

        Example:
            >>> calib_data = calibration.calibrate_camera()
            >>> if calib_data is not None:
            ...     print(f"Camera matrix: {calib_data.mtx}")
        """
        # Check if all frames are set
        if any(frame is None for frame in self._calibration_frames):
            return None

        # Type narrowing - we know all frames are not None
        frames: list[CalibrationFrame] = [
            frame for frame in self._calibration_frames if frame is not None
        ]

        # Check if all frames have same dimensions
        if not all(
            frame.number_of_squares_w == self._number_of_squares_w and
            frame.number_of_squares_h == self._number_of_squares_h
            for frame in frames
        ):
            return None

        # Get 3D reference points (same for all frames)
        reference_points_2d = self.get_checkerboard_3d_reference_points()

        # Flatten to 1D list of tuples for OpenCV
        reference_points_flat = [
            point
            for row in reference_points_2d
            for point in row
        ]

        # Convert to numpy array format expected by cv.calibrateCamera
        objp = np.array(reference_points_flat, dtype=np.float32)

        # Prepare object points (same for all frames) and image points
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []

        # Criteria for corner subpixel refinement
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for frame in frames:
            # Add object points (same for each frame)
            object_points.append(objp)

            # Refine corner positions with subpixel accuracy
            # Note: frame.image is already grayscale from make_calibration_frame
            refined_corners = cv.cornerSubPix(
                frame.image,
                frame.corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            image_points.append(refined_corners)

        # Get image size from first frame
        image_size = (frames[0].image.shape[1], frames[0].image.shape[0])

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None
        )

        if not ret:
            return None

        reprojection_error = 0
        for i in range(len(object_points)):
            projected_image_points, _ = cv.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(image_points[i], projected_image_points, cv.NORM_L2) / len(projected_image_points)
            reprojection_error += error

        return CameraCalibrationData(
            mtx=mtx,
            dist=dist,
            rvecs_list=rvecs,
            tvecs_list=tvecs,
            mean_reprojection_error=reprojection_error / len(object_points)
        )
