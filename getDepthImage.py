import glob
import json
import os

import cv2
import numpy as np


def load_calibration(calib_path):
    """Load the JSON written by runCalibration.py and build everything that only depends on it.

    The rectification maps and the stereo matcher are built here once, so call this once and pass
    the result to compute_depth_image for every pair.
    """
    with open(calib_path) as f:
        c = json.load(f)

    def mat(key):
        return np.array(c[key], dtype=np.float64)

    img_size = tuple(c["image_size"])  # (w, h)

    # Maps from rectified pixel -> raw pixel, which undistort and rectify in one cv2.remap
    map_l = cv2.initUndistortRectifyMap(mat("K_left"), mat("D_left"), mat("R1"), mat("P1"), img_size, cv2.CV_32FC1)
    map_r = cv2.initUndistortRectifyMap(mat("K_right"), mat("D_right"), mat("R2"), mat("P2"), img_size, cv2.CV_32FC1)

    # With f = 660 px and B = 50.6 mm, depth = f*B/d gives about 0.26 m at d = 128 px, so this
    # search range covers everything farther away than that. Raise it (in steps of 16) for closer objects
    num_disparities = 128
    block_size = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        # Smoothness penalties, the values recommended in the OpenCV docs for 3-channel input
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    return {
        "image_size": img_size,
        "map_l": map_l,
        "map_r": map_r,
        "focal_px": float(c["focal_px"]),
        "baseline_mm": float(c["baseline_mm"]),
        "matcher": matcher,
    }

def compute_rectified_image(img_left, img_right, calib):
    """Turn one raw (distorted, unrectified) BGR pair into a an undistorted, rectified BGR pair.

    Returns a tuple of (left, right).
    """
    h, w = img_left.shape[:2]
    if (w, h) != calib["image_size"] or img_right.shape[:2] != (h, w):
        raise ValueError(f"Images are {w}x{h} but the calibration was done at "
                         f"{calib['image_size'][0]}x{calib['image_size'][1]}")

    rect_l = cv2.remap(img_left, *calib["map_l"], cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_right, *calib["map_r"], cv2.INTER_LINEAR)

    return rect_l, rect_r




def compute_depth_image(img_left, img_right, calib):
    """Turn one raw (distorted, unrectified) BGR pair into an RGBD image.

    Returns an H x W x 4 float32 array: channels 0-2 are the rectified left image as RGB (0-255),
    channel 3 is depth in mm. Pixels without a valid disparity have depth 0.
    """

    rect_l, rect_r = compute_rectified_image(img_left, img_right, calib)

    # SGBM returns fixed-point disparity scaled by 16
    disparity = calib["matcher"].compute(rect_l, rect_r).astype(np.float32) / 16.0

    # Rectified, the cameras differ only by a sideways shift, so depth = f * B / disparity.
    # Disparity <= 0 means no match was found (occlusion, no texture, or the left border strip)
    depth = np.zeros_like(disparity)
    valid = disparity > 0
    depth[valid] = calib["focal_px"] * calib["baseline_mm"] / disparity[valid]

    rgb = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.dstack([rgb, depth])


def get_depth_images(dataset_folder_path, calib_path="stereo_calibration.json"):
    """Yield (name, RGBD image) for every pair in dataset_folder_path/left/*.png and dataset_folder_path/right/*.png."""
    calib = load_calibration(calib_path)

    images_left = sorted(glob.glob(f"{dataset_folder_path}/left/*.png"))
    images_right = sorted(glob.glob(f"{dataset_folder_path}/right/*.png"))
    assert images_left, f"No images found in {dataset_folder_path}/left"
    assert len(images_left) == len(images_right), "left/ and right/ have different numbers of images"

    for path_l, path_r in zip(images_left, images_right):
        yield os.path.basename(path_l), compute_depth_image(cv2.imread(path_l), cv2.imread(path_r), calib)

def get_rectified_images(dataset_folder_path, calib_path="stereo_calibration.json"):
    """Yield rectified pair for every pair in dataset_folder_path/left/*.png and dataset_folder_path/right/*.png."""
    calib = load_calibration(calib_path)

    images_left = sorted(glob.glob(f"{dataset_folder_path}/left/*.png"))
    images_right = sorted(glob.glob(f"{dataset_folder_path}/right/*.png"))
    assert images_left, f"No images found in {dataset_folder_path}/left"
    assert len(images_left) == len(images_right), "left/ and right/ have different numbers of images"

    for path_l, path_r in zip(images_left, images_right):
        yield os.path.basename(path_l), compute_rectified_image(cv2.imread(path_l), cv2.imread(path_r), calib)


def depth_to_color(depth, max_depth_mm=5000):
    """Colour-map a depth channel for viewing: near is red, far is blue, invalid is black."""
    scaled = np.clip(depth / max_depth_mm, 0, 1)
    color = cv2.applyColorMap((255 - scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color[depth <= 0] = 0
    return color


if __name__ == "__main__":
<<<<<<< HEAD
    dataset_folder_path = "data/1_no_obstacle"
    out_dir = "output/1_no_obstacle/depth"
=======
    dataset = "data/0_calib/0_calib"
    out_dir = "output/depth"
>>>>>>> dcdc89e (Added file for 3d, also changed square size in calibration to 80x80mm)
    os.makedirs(out_dir, exist_ok=True)

    for name, rgbd in get_depth_images(dataset_folder_path):
        depth = rgbd[:, :, 3]
        valid = depth > 0
        print(f"{name}: {valid.mean() * 100:.0f}% valid, median depth {np.median(depth[valid]):.0f} mm")

        # Rectified left image next to its colour-mapped depth
        bgr = cv2.cvtColor(rgbd[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{out_dir}/{name}", np.hstack([bgr, depth_to_color(depth)]))

    print(f"Saved previews to {out_dir}/")
