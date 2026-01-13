#!/usr/bin/env python3
"""Unit test to verify zone perspective transform calculations.

This test creates a zone with specific camera mapping points and game dimensions
to verify that the perspective transform matrices are calculated correctly.
"""

import numpy as np
import cv2

# Test parameters
CAMERA_VERTICES = [
    (2021, 136),   # P0: top-left
    (205, 177),    # P1: top-right
    (208, 1339),   # P2: bottom-right
    (2049, 1329)   # P3: bottom-left
]

CAMERA_RESOLUTION = (2552, 1425)  # width x height

GAME_WIDTH_INCHES = 34
GAME_HEIGHT_INCHES = 22
PIXELS_PER_INCH = 32

# Calculate game dimensions in pixels
GAME_WIDTH_PX = GAME_WIDTH_INCHES * PIXELS_PER_INCH  # 1088
GAME_HEIGHT_PX = GAME_HEIGHT_INCHES * PIXELS_PER_INCH  # 704

print("=" * 80)
print("ZONE PERSPECTIVE TRANSFORM TEST")
print("=" * 80)
print()

print("Test Configuration:")
print("  Camera vertices (in full camera frame):")
for i, v in enumerate(CAMERA_VERTICES):
    print("    P" + str(i) + ": " + str(v))
print("  Camera resolution: " + str(CAMERA_RESOLUTION[0]) + "x" + str(CAMERA_RESOLUTION[1]))
print("  Game dimensions: " + str(GAME_WIDTH_INCHES) + "\" x " + str(GAME_HEIGHT_INCHES) + "\"")
print("  Pixels per inch: " + str(PIXELS_PER_INCH))
print("  Game dimensions in pixels: " + str(GAME_WIDTH_PX) + "x" + str(GAME_HEIGHT_PX))
print()

# Step 1: Calculate ROI from camera vertices
print("-" * 80)
print("STEP 1: Calculate ROI from camera vertices")
print("-" * 80)

xs = [v[0] for v in CAMERA_VERTICES]
ys = [v[1] for v in CAMERA_VERTICES]
roi = {
    'min_x': min(xs),
    'min_y': min(ys),
    'max_x': max(xs),
    'max_y': max(ys)
}

roi_width = roi['max_x'] - roi['min_x']
roi_height = roi['max_y'] - roi['min_y']

print(f"ROI min: ({roi['min_x']}, {roi['min_y']})")
print(f"ROI max: ({roi['max_x']}, {roi['max_y']})")
print(f"ROI size: {roi_width}x{roi_height}")
print()

# Step 2: Convert camera vertices to ROI coordinates
print("-" * 80)
print("STEP 2: Convert camera vertices to ROI coordinates")
print("-" * 80)

roi_vertices = np.float32([
    [v[0] - roi['min_x'], v[1] - roi['min_y']] for v in CAMERA_VERTICES
])

print("Camera vertices in ROI coordinates:")
for i, v in enumerate(roi_vertices):
    print(f"  P{i}: ({v[0]:.2f}, {v[1]:.2f})")
print()

# Step 3: Define game corner points
print("-" * 80)
print("STEP 3: Define game corner points (in pixels)")
print("-" * 80)

game_points = np.float32([
    [0, 0],                                 # P0: top-left
    [GAME_WIDTH_PX - 1, 0],                 # P1: top-right
    [GAME_WIDTH_PX - 1, GAME_HEIGHT_PX - 1],  # P2: bottom-right
    [0, GAME_HEIGHT_PX - 1]                 # P3: bottom-left
])

print("Game corner points:")
for i, p in enumerate(game_points):
    print(f"  P{i}: ({p[0]:.2f}, {p[1]:.2f})")
print()

# Step 4: Calculate perspective transform matrices
print("-" * 80)
print("STEP 4: Calculate perspective transform matrices")
print("-" * 80)

camera_to_game_matrix = cv2.getPerspectiveTransform(roi_vertices, game_points)
game_to_camera_matrix = cv2.getPerspectiveTransform(game_points, roi_vertices)

print("Camera-to-Game Matrix:")
print(camera_to_game_matrix)
print()

print("Game-to-Camera Matrix:")
print(game_to_camera_matrix)
print()

# Step 5: Test camera to game transformations
print("=" * 80)
print("TEST: Camera ROI coordinates -> Game coordinates")
print("=" * 80)
print()

# Test the camera vertices (should map to game corners)
print("Testing camera vertices (should map to game corners):")
for i, roi_vertex in enumerate(roi_vertices):
    # Convert to homogeneous coordinates
    pos_homo = np.float32([roi_vertex[0], roi_vertex[1], 1.0])
    # Apply transform
    warp_pos_homo = camera_to_game_matrix.dot(pos_homo)
    # Convert back from homogeneous
    warp_pos = (warp_pos_homo / warp_pos_homo[2])[:2]

    expected = game_points[i]
    error = np.linalg.norm(warp_pos - expected)

    print(f"  P{i}: ROI ({roi_vertex[0]:.2f}, {roi_vertex[1]:.2f}) -> Game ({warp_pos[0]:.2f}, {warp_pos[1]:.2f})")
    print(f"       Expected: ({expected[0]:.2f}, {expected[1]:.2f}), Error: {error:.4f}")
print()

# Test center of ROI
print("Testing center of camera ROI:")
roi_center = (roi_width / 2, roi_height / 2)
pos_homo = np.float32([roi_center[0], roi_center[1], 1.0])
warp_pos_homo = camera_to_game_matrix.dot(pos_homo)
warp_pos = (warp_pos_homo / warp_pos_homo[2])[:2]
print(f"  ROI center ({roi_center[0]:.2f}, {roi_center[1]:.2f}) -> Game ({warp_pos[0]:.2f}, {warp_pos[1]:.2f})")
print(f"  Expected approximately: ({GAME_WIDTH_PX/2:.2f}, {GAME_HEIGHT_PX/2:.2f})")
print()

# Test some sample points
print("Testing sample points in camera ROI:")
test_points_roi = [
    (100, 100),
    (500, 500),
    (1000, 600),
]

for test_point in test_points_roi:
    pos_homo = np.float32([test_point[0], test_point[1], 1.0])
    warp_pos_homo = camera_to_game_matrix.dot(pos_homo)
    warp_pos = (warp_pos_homo / warp_pos_homo[2])[:2]

    # Convert to game units (inches)
    game_units = (warp_pos[0] / PIXELS_PER_INCH, warp_pos[1] / PIXELS_PER_INCH)

    print(f"  ROI ({test_point[0]}, {test_point[1]}) -> Game pixels ({warp_pos[0]:.2f}, {warp_pos[1]:.2f}) -> Game units ({game_units[0]:.2f}\", {game_units[1]:.2f}\")")
print()

# Step 6: Test game to camera transformations
print("=" * 80)
print("TEST: Game coordinates -> Camera ROI coordinates")
print("=" * 80)
print()

# Test the game corners (should map back to camera vertices)
print("Testing game corners (should map back to camera vertices):")
for i, game_point in enumerate(game_points):
    # Convert to homogeneous coordinates
    pos_homo = np.float32([game_point[0], game_point[1], 1.0])
    # Apply transform
    warp_pos_homo = game_to_camera_matrix.dot(pos_homo)
    # Convert back from homogeneous
    warp_pos = (warp_pos_homo / warp_pos_homo[2])[:2]

    expected = roi_vertices[i]
    error = np.linalg.norm(warp_pos - expected)

    print(f"  P{i}: Game ({game_point[0]:.2f}, {game_point[1]:.2f}) -> ROI ({warp_pos[0]:.2f}, {warp_pos[1]:.2f})")
    print(f"       Expected: ({expected[0]:.2f}, {expected[1]:.2f}), Error: {error:.4f}")
print()

# Test center of game
print("Testing center of game area:")
game_center = (GAME_WIDTH_PX / 2, GAME_HEIGHT_PX / 2)
pos_homo = np.float32([game_center[0], game_center[1], 1.0])
warp_pos_homo = game_to_camera_matrix.dot(pos_homo)
warp_pos = (warp_pos_homo / warp_pos_homo[2])[:2]
print(f"  Game center ({game_center[0]:.2f}, {game_center[1]:.2f}) -> ROI ({warp_pos[0]:.2f}, {warp_pos[1]:.2f})")
print(f"  Expected approximately: ({roi_width/2:.2f}, {roi_height/2:.2f})")
print()

# Test overlay corners
print("Testing overlay corners (game area bounds):")
overlay_corners = [
    (0, 0),
    (GAME_WIDTH_PX, 0),
    (GAME_WIDTH_PX, GAME_HEIGHT_PX),
    (0, GAME_HEIGHT_PX)
]

for i, corner in enumerate(overlay_corners):
    pos_homo = np.float32([corner[0], corner[1], 1.0])
    warp_pos_homo = game_to_camera_matrix.dot(pos_homo)
    warp_pos = (warp_pos_homo / warp_pos_homo[2])[:2]

    # Convert to full camera frame coordinates
    full_camera_pos = (warp_pos[0] + roi['min_x'], warp_pos[1] + roi['min_y'])

    print(f"  Corner {i}: Game ({corner[0]}, {corner[1]}) -> ROI ({warp_pos[0]:.2f}, {warp_pos[1]:.2f}) -> Full camera ({full_camera_pos[0]:.2f}, {full_camera_pos[1]:.2f})")
print()

print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print()
print("Analysis:")
print("  - Camera vertices should map to game corners with near-zero error")
print("  - Game corners should map back to camera vertices with near-zero error")
print("  - Game coordinates should be in range [0, 1088] for X and [0, 704] for Y")
print("  - Game units should be in range [0, 34] inches for X and [0, 22] inches for Y")
