import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn import linear_model
from getDepthImage import load_calibration, compute_depth_image, depth_to_color

calib = load_calibration('stereo_calibration.json')
P1 = np.array(json.load(open('stereo_calibration.json'))['P1'])
f, cx, cy = P1[0, 0], P1[0, 2], P1[1, 2]          # RECTIFIED left camera: focal length + image centre
print(f'f = {f:.1f} px, cx = {cx:.1f}, cy = {cy:.1f}')

def load_rgbd(seq, frame):
    folder = f'data/{seq}/{seq}'
    L = cv2.imread(f'{folder}/left/{frame:06d}.png')
    R = cv2.imread(f'{folder}/right/{frame:06d}.png')
    rgbd = compute_depth_image(L, R, calib)
    return rgbd[:, :, :3].astype(np.uint8), rgbd[:, :, 3]     # rgb (uint8), depth (mm)

rgb_empty, depth_empty = load_rgbd('1_no_obstacle', 0)      # no robot: used for the floor
rgb_robot, depth_robot = load_rgbd('1_no_obstacle', 128)    # robot in view

def depth_to_xyz(depth_mm):
    """H×W depth in mm -> H×W×3 points in METRES, in the left camera frame."""
    h, w = depth_mm.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))   # pixel column and row of every pixel
    Z = depth_mm / 1000.0
    X = (u - cx) * Z / f
    Y = (v - cy) * Z / f
    return np.dstack([X, Y, Z])


xyz_robot = depth_to_xyz(depth_robot)
valid = depth_robot > 0
print(f'{valid.mean()*100:.0f}% of pixels have a 3D point')

# Click-free check: the 3D position of the robot (box from earlier)
box = valid[430:540, 400:550]
print('robot median X, Y, Z [m]:', np.median(xyz_robot[430:540, 400:550][box], axis=0).round(2))

h, w = depth_robot.shape
camera = o3d.camera.PinholeCameraIntrinsic(w, h, f, f, cx, cy)   # YOUR camera instead of PrimeSenseDefault

color_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb_robot))
depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_robot.astype(np.float32)))

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d, depth_o3d,
    depth_scale=1000.0,        # our depth is in mm -> divide by 1000 = metres
    depth_trunc=15.0,          # drop points farther than 15 m (noisy, small disparities)
    convert_rgb_to_intensity=False)   # keep colours (week 5 used grayscale)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera)

# Same flip as week 5, so the cloud isn't upside down in the viewer
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
print(pcd)
o3d.visualization.draw_geometries([pcd])

xyz_empty = depth_to_xyz(depth_empty)
H = depth_empty.shape[0]
rows = np.arange(H)[:, None] * np.ones((1, depth_empty.shape[1]))

# Floor candidates: valid depth, bottom part of the image, not too far away
floor_mask = (depth_empty > 0) & (rows > 0.75 * H) & (xyz_empty[:, :, 2] < 15)
pts = xyz_empty[floor_mask][::10]          # every 10th point is plenty for a plane
print(len(pts), 'floor candidate points')

XZ = pts[:, [0, 2]]                         # inputs: X and Z   (shape N×2, like X.reshape in week 6)
Y  = pts[:, 1]                              # output: Y

ransac = linear_model.RANSACRegressor(residual_threshold=0.03)   # inlier if within 3 cm
ransac.fit(XZ, Y)

a, b = ransac.estimator_.coef_
c = ransac.estimator_.intercept_
norm = np.sqrt(a**2 + b**2 + 1)
cam_height = c / norm                                 # distance from the camera (0,0,0) to the floor
tilt = np.degrees(np.arccos(1 / norm))                # angle between the floor and the camera's "level"

print(f'floor: Y = {a:.4f}·X + {b:.4f}·Z + {c:.3f}')
print(f'inliers: {ransac.inlier_mask_.mean()*100:.0f}%')
print(f'camera height ≈ {cam_height:.2f} m, tilt ≈ {tilt:.1f}°')


