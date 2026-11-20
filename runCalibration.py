import cv2
import glob
import json
import numpy as np

SQUARE_SIZE_MM = 33.6
NB_VERTICAL = 7 
NB_HORIZONTAL = 5 
DATA_FOLDER = "data/0_calib"
ALPHA=0    # alpha=0 crops so the rectified images hold only valid pixels, alpha=1 keeps every pixel (black borders)
OUT_PATH = "stereo_calibration.json"

# Known corner positions on the flat board, so Z = 0. The pattern counts internal corners, not
# squares, and (NB_VERTICAL, NB_HORIZONTAL) is used as (columns, rows)
objp = np.zeros((NB_HORIZONTAL * NB_VERTICAL, 3), np.float32)
objp[:,:2] = np.mgrid[0:NB_VERTICAL, 0:NB_HORIZONTAL].T.reshape(-1,2)

objp = objp * SQUARE_SIZE_MM

objpoints = []
img_points_left, img_points_right = [], []

images_left = sorted(glob.glob(f'{DATA_FOLDER}/left/*.png'))
images_right = sorted(glob.glob(f'{DATA_FOLDER}/right/*.png'))

# Stop refining a corner after 30 iterations or once it moves less than 0.001 px
criteria = criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

assert len(images_left) == len(images_right)
processed = 0
print(f"Found {len(images_left)} image pairs")

for img_left, img_right in zip(images_left, images_right):
    img_left = cv2.imread(img_left)
    img_right = cv2.imread(img_right)

    gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    retl, cornersl = cv2.findChessboardCornersSB(gray_l, (NB_VERTICAL, NB_HORIZONTAL))
    retr, cornersr = cv2.findChessboardCornersSB(gray_r, (NB_VERTICAL, NB_HORIZONTAL))

    # Both cameras must see the board
    if retl and retr:

        # An 8x6-square board looks the same rotated 180 degrees
        # first-to-last corner direction must agree; if it doesn't, reverse the right corners
        dir_l = cornersl[-1, 0] - cornersl[0, 0]
        dir_r = cornersr[-1, 0] - cornersr[0, 0]
        if np.dot(dir_l, dir_r) < 0:
            cornersr = cornersr[::-1].copy()

        objpoints.append(objp)
        img_points_left.append(cornersl)
        img_points_right.append(cornersr)
        processed += 1

print(f"Corners found in {processed} image pairs")


# numpy is (h, w) while OpenCV wants (w, h)
img_size = gray_l.shape[::-1]

retl, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, img_points_left, img_size, None, None)
retr, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, img_points_right, img_size, None, None)

stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
# Keep the intrinsics we just solved and let the stereo step only find R and T 
# This way there are fewer free parameters and a more stable fit
stereo_flags = cv2.CALIB_FIX_INTRINSIC

ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, img_points_left, img_points_right,
    mtx_l, dist_l, mtx_r, dist_r, img_size,
    criteria=stereo_criteria, flags=stereo_flags
)

print(f"RMS reprojection error: left {retl:.3f} px, right {retr:.3f} px, stereo {ret_stereo:.3f} px")


#Output of the camera calibration
calib = {
    "image_size": img_size,
    "pattern_size": (NB_VERTICAL, NB_HORIZONTAL),
    "num_pairs": processed,
    "rms_left": retl, "rms_right": retr, "rms_stereo": ret_stereo,
    "K_left": mtx_l, "D_left": dist_l, "K_right": mtx_r, "D_right": dist_r,
    "R": R, "T": T, "E": E, "F": F,
}



# Rotates both cameras into a common view where matching points lie on the same row.
# alpha=0 crops so the rectified images hold only valid pixels, alpha=1 keeps every pixel (black borders)
R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
    calib["K_left"], calib["D_left"], calib["K_right"], calib["D_right"],
    calib["image_size"], calib["R"], calib["T"],
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=ALPHA
)

# depth = focal_px * baseline_mm / disparity_px, both read from the rectified right camera matrix
focal_px = float(P1[0, 0])
baseline_mm = float(-P2[0, 3] / P2[0, 0])

def as_list(a):
    return np.asarray(a, dtype=np.float64).tolist()

data = {
    "units": "mm",
    "image_size": list(calib["image_size"]),
    "pattern_size": list(calib["pattern_size"]),
    "square_size_mm": SQUARE_SIZE_MM,
    "num_pairs": calib["num_pairs"],
    "rms_left": float(calib["rms_left"]),
    "rms_right": float(calib["rms_right"]),
    "rms_stereo": float(calib["rms_stereo"]),
    "alpha": ALPHA,
    "K_left": as_list(calib["K_left"]),
    "D_left": as_list(calib["D_left"]),
    "K_right": as_list(calib["K_right"]),
    "D_right": as_list(calib["D_right"]),
    "R": as_list(calib["R"]),
    "T": as_list(calib["T"]),
    "E": as_list(calib["E"]),
    "F": as_list(calib["F"]),
    "R1": as_list(R1),
    "R2": as_list(R2),
    "P1": as_list(P1),
    "P2": as_list(P2),
    "Q": as_list(Q),
    # (x, y, w, h) of the valid pixels in each rectified image
    "roi_left": [int(v) for v in roi_l],
    "roi_right": [int(v) for v in roi_r],
    "focal_px": focal_px,
    "baseline_mm": baseline_mm,
}

with open(OUT_PATH, "w") as f:
    json.dump(data, f, indent=2)
print(f"Saved calibration to {OUT_PATH} (focal {focal_px:.1f} px, baseline {baseline_mm:.1f} mm)")
