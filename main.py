import argparse
import os
from glob import glob
import numpy as np
import cv2

from detect_robot import detect_sequence , detect_sequence3D
from getDepthImage import get_depth_images, get_rectified_images


def detect2D(dataset_path, imagesAreRaw = True):
    """Rectify the dataset first if needed, then run detection on the rectified set of left images."""

    if imagesAreRaw:
        lefts = (left for _, (left, _) in get_rectified_images(dataset_path))
    else: 
        lefts = (cv2.imread(p) for p in sorted(glob(f"{dataset_path}/left/*.png")))

    detect_sequence(lefts)

def detect3D(dataset_path, imagesAreRaw = True):
    """Rectify the dataset first if needed, then run 3D detection on the rectified images."""
  
    if imagesAreRaw:
        RGB_D = (rgbd for _, rgbd in get_depth_images(dataset_path))
    else: 
        RGB_D = (np.load(p) for p in sorted(glob(f"{dataset_path}/RGB_D/*.npy")))

    detect_sequence3D(RGB_D)

def produce_rectified():
    pass   

def produce_RGBD():
    pass   

TASKS = {
    "produce_rectified": produce_rectified,
    "produce_RGBD": produce_RGBD,
    "detect2D": detect2D,
    "detect3D": detect3D,

}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one pipeline stage on one sequence.")
    parser.add_argument("--dataset", required=True, help="e.g. data/1_no_obstacle-2")
    parser.add_argument("--task", required=True, choices=TASKS.keys())
    args = parser.parse_args()

    TASKS[args.task](args.dataset)
