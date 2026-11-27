import os
import cv2
from glob import glob
import imutils
import numpy as np
from kalman import Kalman3D, Kalman2D
from getDepthImage import compute_rectified_image, compute_depth_roi, depth_to_xyz, box_to_xyz, load_calibration, measurement_variance


#returns tuple left, right top corners and width, height of box
def detect(frame, background, diff_threshold=100, min_area=100, return_mask=False):
    diff = cv2.absdiff(frame, background)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, diff_threshold, 255, cv2.THRESH_BINARY)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask[:200, :] = 0

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]

    if len(cnts) < 1:
        return (None, mask) if return_mask else None

    biggest = max(cnts, key=cv2.contourArea)
    box = cv2.boundingRect(biggest)
    return (box, mask) if return_mask else box


def detect_sequence(image_set, background=None):
    frames = iter(image_set)
    background = next(frames)

    for image in frames:
        box = detect(image, background, diff_threshold=25, min_area=300)

        if box is None:
            print("No movement was detected")
        else:
            x, y, width, height = box
            print(f"detected at ({x}, {y}, {width}, {height})")

        if _draw_previous_frame("detections", (image, box)) == "q-pressed":
            break
    
    cv2.destroyAllWindows()


def _anchored_centre(lo, size, ref_size, pred_centre):
    """Centre of a box that may be clipped (by an occluder or the image border) along one axis.

    lo, size: measured start and length of the box on this axis. ref_size: the robot's full size.
    A clipped box still has one TRUE edge -- the side away from the occluder -- so pick the edge
    closer to where the filter predicts it and measure the centre from it plus half the full
    size. Using the clipped box's own centre instead pulls the estimate back and slows it down.
    """
    hi = lo + size
    if abs(lo - (pred_centre - ref_size / 2)) <= abs(hi - (pred_centre + ref_size / 2)):
        return lo + ref_size / 2       # near edge visible, far side hidden
    return hi - ref_size / 2           # far edge visible, near side hidden


def _draw_previous_frame(window, last_frame, kalman_box=None, label=None):
    """Draw the detection (red), the Kalman estimate (green: centre u, v and size w, h in px) and a
    label on a copy of the frame, and show it. last_frame: (frame, box), or None when there is no frame
    yet (then nothing is drawn). Returns "q-pressed" if the user pressed q, else None."""
    if last_frame is None:
        return None
    frame, box = last_frame
    output = frame.copy()
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)
    if kalman_box is not None:
        u, v, w, h = kalman_box
        cv2.putText(output, "The red box is detection. The green is Kalman.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
        cv2.rectangle(output, (int(u - w / 2), int(v - h / 2)), (int(u + w / 2), int(v + h / 2)), (0, 255, 0), 3)
    if label is not None:
        cv2.putText(output, label, (10, 60 if kalman_box is not None else 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(window, output)
    return "q-pressed" if cv2.waitKey(30) & 0xFF == ord('q') else None


def _grow_fast_shrink_slow(ref, new):
    """Occlusions only ever make the box SMALLER, so trust a bigger box at once, a smaller one slowly."""
    return new if ref is None else (0.5 * ref + 0.5 * new if new > ref else 0.9 * ref + 0.1 * new)


def _classify_box(box, ref_w, ref_h, s, pred_u, pred_v, small_stable_streak , prev_w, partial_ratio, min_visible):
    """Shared by the 2D and 3D trackers. ref_w/ref_h: robot's full size, compared as w*s (2D: s=1,
    sizes in px; 3D: s=Z, sizes in px*m so they don't depend on distance).
    Returns (detection_kind, centre, small_stable_streak ), detection_kind one of "full", "reset needed", "partial", "small"."""
    x, y, w, h = box
    partial_w = ref_w is not None and w * s < partial_ratio * ref_w
    partial_h = ref_h is not None and h * s < partial_ratio * ref_h
    if not (partial_w or partial_h):
        return "full", None, 0
    small_stable_streak  = small_stable_streak  + 1 if prev_w is not None and abs(w - prev_w) <= 3 else 0
    if small_stable_streak  >= 7:
        # box smaller but stable for a while -> robot really got smaller, not occluded: re-seed
        print("box stayed small, resetting the reference size")
        return "reset needed", None, 0
    if w * s >= min_visible * ref_w and h * s >= min_visible * ref_h:
        # Partially hidden: measure the centre from the robot's visible edge instead of skipping
        cu = _anchored_centre(x, w, ref_w / s, pred_u) if partial_w else x + w / 2
        cv = _anchored_centre(y, h, ref_h / s, pred_v) if partial_h else y + h / 2
        return "partial", (cu, cv), small_stable_streak 
    print("detection too small to trust, skipping Kalman update")
    return "small", None, small_stable_streak 


def _project(X, Y, Z, c):
    return c["focal_px"] * X / Z + c["cx"], c["focal_px"] * Y / Z + c["cy"]


#2D tracking: same detector as detect_sequence, plus a constant-velocity Kalman filter on the
#box centre in pixels. Needs only the left images (no SGBM), so it also works where stereo has no depth
def detect_sequence_WithKalman(image_set, partial_ratio=0.9, min_visible=0.25, background=None):
    kf = Kalman2D()

    frames = iter(image_set)
    if background is None:
        background = next(frames)
    ref_w = ref_h = prev_w = None
    small_stable_streak = 0

    def draw_last_frame_frame():
        """Draw the previous pass's frame with the filter as that pass left it. True if q was pressed."""
        kalman_box = label = None
        if kf.initialized and ref_w is not None:
            u, v = kf.state[0, 0], kf.state[2, 0]
            print(f"Kalmanfilter at,  u={u:.1f} v={v:.1f} px")
            kalman_box, label = (u, v, ref_w, ref_h), f"u={u:.0f} v={v:.0f} px"
        return _draw_previous_frame("2D detections, with Kalman", last_frame, kalman_box, label)

    # The previous pass's frame is drawn at the top of the next one, so the `continue`s below can't skip it
    last_frame = None
    for image in frames:
        prev_w = last_frame[1][2] if last_frame is not None and last_frame[1] is not None else None
        if draw_last_frame_frame() == "q-pressed":
            break

        kf.predict()
        box = detect(image, background, diff_threshold=25, min_area=300)
        last_frame = (image, box)

        if box is None: continue

        x, y, w, h = box
        # No depth here, so sizes are compared in px (valid while the distance is constant)
        detection_kind, centre, small_stable_streak  = _classify_box(box, ref_w, ref_h, 1, kf.state[0, 0], kf.state[2, 0],
                                                small_stable_streak , prev_w, partial_ratio, min_visible)
        if detection_kind == "full":
            kf.update([x + w / 2, y + h / 2])
            ref_w, ref_h = _grow_fast_shrink_slow(ref_w, w), _grow_fast_shrink_slow(ref_h, h)
            print(f"Bounding box at {box}, centre u={x + w / 2:.1f} v={y + h / 2:.1f} px")
        elif detection_kind == "reset needed":
            ref_w = ref_h = None
        elif detection_kind == "partial":
            kf.update(centre)
            print(f"partial detection {box}, centre from visible edge u={centre[0]:.1f} v={centre[1]:.1f} px")
    
    else:  # loop ran to the end (no q pressed): the final frame still needs drawing
        draw_last_frame_frame()
    cv2.destroyAllWindows()


#works on whole rgbd pictures
def detect_sequence3D(rgbd_frames, calib_path="stereo_calibration.json", background=None):
    calib = load_calibration(calib_path)

    frames = iter(rgbd_frames)
    # background = cv2.cvtColor(next(frames)[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)

    if background is None:
        background = cv2.cvtColor(next(frames)[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    for rgbd in frames:
        rgb = cv2.cvtColor(rgbd[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        depth = rgbd[:, :, 3]

        box = detect(rgb, background, diff_threshold=25, min_area=300)
        label = None
        if box is not None:
            xyz = box_to_xyz(depth, box, calib)
            if xyz is None:
                print(f"detected at {box} but depth is unreliable there")
            else:
                X, Y, Z = xyz
                print(f"detected at {box}, X={X:.2f} Y={Y:.2f} Z={Z:.2f} m")
                label = f"X={X:.2f} Y={Y:.2f} Z={Z:.2f} m"

        if _draw_previous_frame("3D detections", (rgb, box), label=label) == "q-pressed":
            break

    cv2.destroyAllWindows()

#it doesn't produce whole rgbd after detecting the robot we extract it from the picture and than calculate sgbm over this region
#so it works faster and produces movei with higer fps rate
def detect_sequence3D_roi(image_pairs, calib_path="stereo_calibration.json", background=None):
    calib = load_calibration(calib_path)

    pairs = iter(image_pairs)
    # bg_l, bg_r = next(pairs)
    # background, _ = compute_rectified_image(bg_l, bg_r, calib)

    if background is None:
        bg_l, bg_r = next(pairs)
        background, _ = compute_rectified_image(bg_l, bg_r, calib)

    for img_l, img_r in pairs:
        rect_l, _ = compute_rectified_image(img_l, img_r, calib)

        box = detect(rect_l, background, diff_threshold=25, min_area=300)
        label = None
        if box is not None:
            x, y, w, h = box
            depth_roi = compute_depth_roi(img_l, img_r, calib, box)
            xyz = depth_to_xyz(depth_roi, x + w / 2, y + h / 2, calib)
            if xyz is None:
                print(f"detected at {box} but depth is unreliable there")
            else:
                X, Y, Z = xyz
                print(f"detected at {box}, X={X:.2f} Y={Y:.2f} Z={Z:.2f} m")
                label = f"X={X:.2f} Y={Y:.2f} Z={Z:.2f} m"

        if _draw_previous_frame("3D detections (ROI)", (rect_l, box), label=label) == "q-pressed":
            break

    cv2.destroyAllWindows()


def detect_sequence3D_roi_WithKalman(image_pairs, calib_path="stereo_calibration.json", partial_ratio=0.9, min_visible=0.25, background=None):
    calib = load_calibration(calib_path)
    kf = Kalman3D()

    pairs = iter(image_pairs)
    if background is None:
        bg_l, bg_r = next(pairs)
        background, _ = compute_rectified_image(bg_l, bg_r, calib)
    ref_wZ = ref_hZ = prev_w = None
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    small_stable_streak = 0

    def draw_last_frame():
        """Draw the previous pass's frame with the filter as that pass left it. True if q was pressed."""
        kX, kY, kZ = kxyz[0, 0], kxyz[2, 0], kxyz[4, 0]
        kalman_box = label = None
        if kZ > 0:  # filter has a measurement (state starts at 0)
            print(f"Kalmanfilter at, X={kX:.2f} Y={kY:.2f} Z={kZ:.2f} m")
            if ref_wZ is not None:  # reference size not just reset
                kalman_box = (*_project(kX, kY, kZ, calib), ref_wZ / kZ, ref_hZ / kZ)
                label = f"X={kX:.2f} Y={kY:.2f} Z={kZ:.2f} m"
        return _draw_previous_frame("3D detections (ROI), with Kalman ", last_frame, kalman_box, label)

    # The previous pass's frame is drawn at the top of the next one, so the `continue`s below can't skip it
    last_frame = None
    kxyz = kf.state  # all zeros, so kZ = 0: nothing is printed or drawn on the first pass
    for img_l, img_r in pairs:

        prev_w = last_frame[1][2] if last_frame is not None and last_frame[1] is not None else None
        if draw_last_frame() == "q-pressed":
            break

        rect_l, _ = compute_rectified_image(img_l, img_r, calib)
        kxyz, _ = kf.predict()
        box, mask = detect(rect_l, background, diff_threshold=25, min_area=300, return_mask=True)
        last_frame = (rect_l, box)

        if box is None: continue

        x, y, w, h = box
        depth_roi = compute_depth_roi(img_l, img_r, calib, box)
        m = cv2.erode(mask, erode_kernel)[y:y + h, x:x + w] > 0
        xyz = depth_to_xyz(depth_roi, x + w / 2, y + h / 2, calib, mask=m)

        if xyz is None:
            print(f"detected at {box} but depth is unreliable there")
            continue
            
        # _classify_box runs at the filter's PREDICTED depth once it has one
        Zs = kxyz[4, 0] if kxyz[4, 0] > 0 else xyz[2]
        detection_kind, centre, small_stable_streak  = _classify_box(box, ref_wZ, ref_hZ, Zs, *_project(kxyz[0, 0], kxyz[2, 0], Zs, calib),
                                                small_stable_streak , prev_w, partial_ratio, min_visible)
        if detection_kind == "full":
            kxyz, _ = kf.update(xyz, measurement_variance(xyz, calib))
            X, Y, Z = xyz
            ref_wZ, ref_hZ = _grow_fast_shrink_slow(ref_wZ, w * Z), _grow_fast_shrink_slow(ref_hZ, h * Z)
            print(f"Bounding box at {box}")
            print(f"Detection at,    X={X:.2f} Y={Y:.2f} Z={Z:.2f} m")
        elif detection_kind == "reset needed":
            ref_wZ = ref_hZ = None
        elif detection_kind == "partial":
            # back-project the edge-based centre at the predicted depth; Z gets a huge
            # variance, so this update corrects X and Y only
            pxyz = ((centre[0] - calib["cx"]) * Zs / calib["focal_px"],
                    (centre[1] - calib["cy"]) * Zs / calib["focal_px"], Zs)
            var = measurement_variance(pxyz, calib)
            var[2] = 1e6
            kxyz, _ = kf.update(pxyz, var)
            print(f"partial detection {box}, centre from visible edge: "
                    f"X={pxyz[0]:.2f} Y={pxyz[1]:.2f} m (Z not used, measured {xyz[2]:.2f} m)")
    
    else:  # loop ran to the end (no q pressed): the final frame still needs drawing
        draw_last_frame()

    cv2.destroyAllWindows()




#THIS IS ONLY USED FOR DEBUGGING THRESHOLD VALUE
#plays the sequence with a trackbar to tune diff_threshold live
def test_thresholds(dataset_path, start_threshold=25):
    dataset = sorted(glob(f'{dataset_path}/left/*.png'))
    background = cv2.imread(dataset[0])

    cv2.namedWindow("detections")
    cv2.createTrackbar("threshold", "detections", start_threshold, 255, lambda x: None)

    for image in dataset[1:]:
        image = cv2.imread(image)
        threshold = cv2.getTrackbarPos("threshold", "detections")

        diff = cv2.absdiff(image, background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        box = detect(image, background, diff_threshold=threshold)
        if box is not None:
            x, y, width, height = box
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 0, 255), 3)

        cv2.putText(output, f"threshold={threshold}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("detections", output)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

