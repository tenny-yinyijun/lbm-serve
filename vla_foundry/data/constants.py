"""Constants for data processing and preprocessing."""

# Point map encoding constants
# Point maps are stored as uint16 with an offset to handle negative values
# Original depth in mm = (uint16_value - POINT_MAP_UINT16_OFFSET)
# Depth in meters = (uint16_value - POINT_MAP_UINT16_OFFSET) / POINT_MAP_MM_TO_M_SCALE
POINT_MAP_UINT16_OFFSET = 32768.0  # 2^15, allows encoding of values from -32768 to +32767 mm
POINT_MAP_MM_TO_M_SCALE = 1000.0  # Convert millimeters to meters

# Point map valid range after offset (for clipping during encoding)
POINT_MAP_MIN_MM = -32768  # Minimum valid coordinate in mm
POINT_MAP_MAX_MM = 32767  # Maximum valid coordinate in mm

# Camera fisheye distortion parameters and intrinsics
# Dictionary mapping camera semantic names to their fisheye distortion coefficients and intrinsics
# Source: ../anzu/intuitive/visuomotor/config/cabot_*_multitask_scenarios.yaml
#
# IMPORTANT: Due to a simulator bug, cabot depth images are stored as PINHOLE (undistorted)
# while RGB images are FISHEYE (distorted). To create properly aligned point clouds, we must:
# 1. Detect if data is from cabot by matching K_fisheye intrinsics (up to 3 decimal places)
# 2. Apply forward fisheye distortion to depth images to match RGB
#
# K_fisheye: Intrinsics for the fisheye (distorted) RGB images
# K_pinhole: Intrinsics for the pinhole (undistorted) depth images
CAMERA_FISHEYE_DISTORTION = {
    "wrist_left_minus": {
        "d": [0.01143222, 0.03684697, -0.05852974, 0.03031175],
        "K_fisheye": [[416.31268, 0.0, 484.25815], [0.0, 415.3666, 304.04462], [0.0, 0.0, 1.0]],
        "K_pinhole": [[231.778, 0.0, 486.614], [0.0, 231.252, 303.9256], [0.0, 0.0, 1.0]],
    },  # BFS_23595718
    "wrist_left_plus": {
        "d": [0.01607469, 0.00335369, 0.00144981, -0.00187433],
        "K_fisheye": [[413.27625, 0.0, 483.4717], [0.0, 412.59454, 299.43713], [0.0, 0.0, 1.0]],
        "K_pinhole": [[222.245, 0.0, 486.036], [0.0, 221.878, 299.461], [0.0, 0.0, 1.0]],
    },  # BFS_23595721
    "wrist_right_minus": {
        "d": [0.16117042, -0.32232478, 0.3279451, -0.11335791],
        "K_fisheye": [[385.8977, 0.0, 459.91806], [0.0, 385.20963, 301.92267], [0.0, 0.0, 1.0]],
        "K_pinhole": [[178.822, 0.0, 436.410], [0.0, 178.504, 301.674], [0.0, 0.0, 1.0]],
    },  # BFS_23595722
    "wrist_right_plus": {
        "d": [0.06496022, -0.01126586, -0.04739537, 0.03582876],
        "K_fisheye": [[391.72803, 0.0, 473.89243], [0.0, 392.77228, 278.94452], [0.0, 0.0, 1.0]],
        "K_pinhole": [[212.426, 0.0, 470.830], [0.0, 212.992, 279.669], [0.0, 0.0, 1.0]],
    },  # BFS_23595725
}
