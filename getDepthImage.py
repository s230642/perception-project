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

    # With f = 660 px and B = 120.4 mm, depth = f*B/d gives about 0.62 m at d = 128 px, so this
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

    # Rectified left camera matrix: fx == fy here, cx/cy is the image centre used for back-projection
    P1 = mat("P1")

    return {
        "image_size": img_size,
        "map_l": map_l,
        "map_r": map_r,
        "focal_px": float(c["focal_px"]),
        "baseline_mm": float(c["baseline_mm"]),
        "cx": float(P1[0, 2]),
        "cy": float(P1[1, 2]),
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


def compute_depth_roi(img_left, img_right, calib, roi):
    """Run SGBM only around roi=(x, y, w, h) in rectified-image coordinates, instead of over the
    whole frame -- much faster when depth is only needed for one already-detected region.

    Returns an h x w float32 array of depth in mm, covering exactly the roi (0 where no valid
    disparity was found, same convention as compute_depth_image).
    """
    rect_l, rect_r = compute_rectified_image(img_left, img_right, calib)

    x, y, w, h = roi
    num_disp = calib["matcher"].getNumDisparities()
    block_pad = calib["matcher"].getBlockSize() // 2
    pad = 10

    # both crops must be the same size, and extend left by num_disp so the matcher has pixels
    # to search for the roi's own columns -- cropping tightly to the roi would leave it with no
    # search room and every pixel in it would come back invalid
    desired_x0 = x - num_disp - pad
    desired_x1 = x + w + pad
    x0 = max(0, desired_x0)
    x1 = min(rect_l.shape[1], desired_x1)

    # a box near the left or right edge of the frame gets clamped on that side -- reclaim the
    # lost margin on the other side, otherwise the crop can end up far narrower than num_disp
    # and StereoSGBM miscalculates its internal buffer size instead of just rejecting the input
    if x0 > desired_x0:
        x1 = min(rect_l.shape[1], x1 + (x0 - desired_x0))
    elif x1 < desired_x1:
        x0 = max(0, x0 - (desired_x1 - x1))

    y0 = max(0, y - block_pad)
    y1 = min(rect_l.shape[0], y + h + block_pad)

    crop_l = rect_l[y0:y1, x0:x1]
    crop_r = rect_r[y0:y1, x0:x1]

    disparity = calib["matcher"].compute(crop_l, crop_r).astype(np.float32) / 16.0
    depth_crop = np.zeros_like(disparity)
    valid = disparity > 0
    depth_crop[valid] = calib["focal_px"] * calib["baseline_mm"] / disparity[valid]

    # depth_crop is aligned to (x0, y0); slice out just the roi's own region
    return depth_crop[y - y0:y - y0 + h, x - x0:x - x0 + w]


def depth_to_xyz(depth_region, u, v, calib, min_valid_frac=0.2, mask=None):
    """Median of the valid pixels in depth_region (mm) back-projected to XYZ in metres.

    (u, v) is the pixel position to use for X/Y -- pass the absolute rectified-image coordinates
    of depth_region's centre, not a coordinate local to depth_region itself. Returns None if too
    little of depth_region has a valid depth reading to trust.
    """
    inside = np.ones(depth_region.shape, bool) if mask is None else mask
    valid = (depth_region > 0) & inside
    if not valid.any() or valid.sum() < min_valid_frac * inside.sum():
        return None

    z = np.median(depth_region[valid]) / 1000.0  # mm -> m
    X = (u - calib["cx"]) * z / calib["focal_px"]
    Y = (v - calib["cy"]) * z / calib["focal_px"]
    return X, Y, z


def measurement_variance(xyz, calib, sigma_u=3.0, sigma_v=3.0, sigma_d=0.15):
    """Variances (m^2) of a detected (X, Y, Z), for the Kalman filter's R. Depth noise grows with Z^2.

    sigma_u, sigma_v: box-centre jitter in px. sigma_d: disparity error in px. The defaults were
    fitted to the detections in data/3_curved_bend.
    """
    X, Y, Z = xyz
    f = calib["focal_px"]
    B = calib["baseline_mm"] / 1000.0

    sigma_Z = Z**2 / (f * B) * sigma_d
    var_X = (Z / f * sigma_u)**2 + (X / Z * sigma_Z)**2
    var_Y = (Z / f * sigma_v)**2 + (Y / Z * sigma_Z)**2
    return np.array([var_X, var_Y, sigma_Z**2])


def box_to_xyz(depth, box, calib, min_valid_frac=0.2):
    """depth is a full-size depth map (mm); box=(x, y, w, h) is cropped out of it first."""
    x, y, w, h = box
    region = depth[y:y + h, x:x + w]
    return depth_to_xyz(region, x + w / 2, y + h / 2, calib, min_valid_frac)


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


def _pick(n, max_frames):
    """Indices of all n frames, or of max_frames of them spread evenly from first to last."""
    if max_frames is None or max_frames >= n:
        return range(n)
    return np.unique(np.linspace(0, n - 1, max_frames).round().astype(int))

def iter_image_pairs(dataset_path, max_frames=None):
    """Yield (name, left BGR, right BGR) for every stereo pair in dataset_path/left and /right, in
    frame order. max_frames: only yield that many pairs, spread evenly over the whole sequence."""
    images_left = sorted(glob.glob(f"{dataset_path}/left/*.png"))
    images_right = sorted(glob.glob(f"{dataset_path}/right/*.png"))
    assert images_left, f"No images found in {dataset_path}/left"
    assert len(images_left) == len(images_right), "left/ and right/ have different numbers of images"

    for i in _pick(len(images_left), max_frames):
        yield os.path.basename(images_left[i]), cv2.imread(images_left[i]), cv2.imread(images_right[i])
        
def median_background(dataset_path, calib_path="stereo_calibration.json", imagesAreRaw=True, max_frames=50):
    """Background image = per-pixel median of max_frames rectified left frames spread over the sequence.

    The robot keeps moving, so at every pixel most of the sampled frames show the empty scene and
    the median picks that. Unlike using frame 0, this works when the robot is already visible at
    the start (3_curved_bend): frame 0 as background leaves a "ghost" detection where it started.
    Needs the whole sequence up front, so it is an offline method.
    """
    calib = load_calibration(calib_path) if imagesAreRaw else None
    frames = []
    for _, left, right in iter_image_pairs(dataset_path, max_frames=max_frames):
        frames.append(compute_rectified_image(left, right, calib)[0] if imagesAreRaw else left)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


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
    dataset_folder_path = "data/1_no_obstacle"
    out_dir = "output/1_no_obstacle/depth"
    os.makedirs(out_dir, exist_ok=True)

    for name, rgbd in get_depth_images(dataset_folder_path):
        depth = rgbd[:, :, 3]
        valid = depth > 0
        print(f"{name}: {valid.mean() * 100:.0f}% valid, median depth {np.median(depth[valid]):.0f} mm")

        # Rectified left image next to its colour-mapped depth
        bgr = cv2.cvtColor(rgbd[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{out_dir}/{name}", np.hstack([bgr, depth_to_color(depth)]))

    print(f"Saved previews to {out_dir}/")
