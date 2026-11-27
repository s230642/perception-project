import argparse
import os
from glob import glob
import numpy as np
import cv2

from detect_robot import detect_sequence , detect_sequence_WithKalman, detect_sequence3D, detect_sequence3D_roi, detect_sequence3D_roi_WithKalman
from getDepthImage import get_depth_images, get_rectified_images


def detect2D(dataset_path, imagesAreRaw = True, calib_path="stereo_calibration.json"):
    """Rectify the dataset first if needed, then run detection on the rectified set of left images."""

    if imagesAreRaw:
        lefts = (left for _, (left, _) in get_rectified_images(dataset_path, calib_path))
    else:
        lefts = (cv2.imread(p) for p in sorted(glob(f"{dataset_path}/left/*.png")))

    detect_sequence(lefts)

def detect2D_WithKalman(dataset_path, imagesAreRaw = True, calib_path="stereo_calibration.json"):
    """Same as detect2D, with a 2D (image-plane) Kalman Filter on top of the detections."""

    if imagesAreRaw:
        lefts = (left for _, (left, _) in get_rectified_images(dataset_path, calib_path))
    else:
        lefts = (cv2.imread(p) for p in sorted(glob(f"{dataset_path}/left/*.png")))

    detect_sequence_WithKalman(lefts)

def detect3D(dataset_path, imagesAreRaw = True, calib_path="stereo_calibration.json"):
    """Rectify the dataset first if needed, then run 3D detection on the rectified images."""

    if imagesAreRaw:
        RGB_D = (rgbd for _, rgbd in get_depth_images(dataset_path, calib_path))
    else:
        RGB_D = (np.load(p) for p in sorted(glob(f"{dataset_path}/RGB_D/*.npy")))

    detect_sequence3D(RGB_D, calib_path=calib_path)

def detect3D_roi(dataset_path, imagesAreRaw = True, calib_path="stereo_calibration.json"):
    """Detect the robot and compute its XYZ, running SGBM only around the detected box each
    frame instead of over the whole image first (see detect_robot.detect_sequence3D_roi)."""
    assert imagesAreRaw, "detect3D_roi needs raw images -- it rectifies internally per detection"

    lefts = sorted(glob(f"{dataset_path}/left/*.png"))
    rights = sorted(glob(f"{dataset_path}/right/*.png"))
    image_pairs = ((cv2.imread(l), cv2.imread(r)) for l, r in zip(lefts, rights))

    detect_sequence3D_roi(image_pairs, calib_path=calib_path)

def detect3D_roi_WithKalman(dataset_path, imagesAreRaw = True, calib_path="stereo_calibration.json"):
    """Detect the robot and compute its XYZ with a Kalman Filter on top, running SGBM only around the detected box each
    frame instead of over the whole image first (see detect_robot.detect_sequence3D_roi)."""
    assert imagesAreRaw, "detect3D_roi needs raw images -- it rectifies internally per detection"

    lefts = sorted(glob(f"{dataset_path}/left/*.png"))
    rights = sorted(glob(f"{dataset_path}/right/*.png"))
    image_pairs = ((cv2.imread(l), cv2.imread(r)) for l, r in zip(lefts, rights))

    detect_sequence3D_roi_WithKalman(image_pairs, calib_path=calib_path)


def produce_rectified(dataset_path, imagesAreRaw = True, calib_path="stereo_calibration.json"):
    """Rectify the raw dataset and save the pairs to output/rectified/<dataset>/left and /right."""
    assert imagesAreRaw, "produce_rectified needs raw images"

    out_dir = f"output/rectified/{os.path.basename(os.path.normpath(dataset_path))}"
    os.makedirs(f"{out_dir}/left", exist_ok=True)
    os.makedirs(f"{out_dir}/right", exist_ok=True)

    for name, (left, right) in get_rectified_images(dataset_path, calib_path):
        cv2.imwrite(f"{out_dir}/left/{name}", left)
        cv2.imwrite(f"{out_dir}/right/{name}", right)

def produce_RGBD(dataset_path, imagesAreRaw = True, calib_path="stereo_calibration.json"):
    """ WARNING, THIS MAKES MANY GIGABYTE DATA.
    Compute RGBD images from the raw dataset and save them to output/rectified/<dataset>/RGB_D as .npy."""
    assert imagesAreRaw, "produce_RGBD needs raw images"

    out_dir = f"output/rectified/{os.path.basename(os.path.normpath(dataset_path))}/RGB_D"
    os.makedirs(out_dir, exist_ok=True)

    for name, rgbd in get_depth_images(dataset_path, calib_path):
        np.save(f"{out_dir}/{os.path.splitext(name)[0]}.npy", rgbd)

TASKS = {
    "produce_rectified": produce_rectified,
    "produce_RGBD": produce_RGBD,
    "detect2D": detect2D,
    "detect2D_WithKalman": detect2D_WithKalman,
    "detect3D": detect3D,
    "detect3D_roi": detect3D_roi,
    "detect3D_roi_WithKalman": detect3D_roi_WithKalman,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one pipeline stage on one sequence.")
    parser.add_argument("--dataset", required=True, help="e.g. data/1_no_obstacle-2")
    parser.add_argument("--task", required=True, choices=TASKS.keys())
    parser.add_argument("--rectified", choices=["true", "false"], default="false",
                        help="write true if the dataset images are already rectified, false if they are raw (default: false)")
    parser.add_argument("--calib", default="stereo_calibration.json", help="calibration JSON path")

    args = parser.parse_args()

    TASKS[args.task](args.dataset, imagesAreRaw=args.rectified == "false", calib_path=args.calib)
