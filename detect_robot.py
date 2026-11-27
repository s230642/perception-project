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

def detect_sequence(image_set):
    frames = iter(image_set)
    background = next(frames)

    for image in frames:
        output = image.copy()
        box = detect(image, background, diff_threshold=25, min_area=300)

        if box is None:
            print("No movement was detected")
        else:
            x, y, width, height = box
            print(f"detected at ({x}, {y}, {width}, {height})")
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 0, 255), 3)

        cv2.imshow("detections", output)
        if cv2.waitKey(30) & 0xFF == ord('q'):
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


#2D tracking: same detector as detect_sequence, plus a constant-velocity Kalman filter on the
#box centre in pixels. Needs only the left images (no SGBM), so it also works where stereo has no depth
def detect_sequence_WithKalman(image_set, partial_ratio=0.9, min_visible=0.25):
    kf = Kalman2D()

    frames = iter(image_set)
    background = next(frames)
    ref_w = ref_h = prev_w = None
    small_stable_streak = 0

    for image in frames:
        output = image.copy()
        ### PREDICT KALMAN ###
        kf.predict()
        ######################
        box = detect(image, background, diff_threshold=25, min_area=300)
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)

            # No depth here, so the partial-box test compares w and h directly instead of w*Z.
            # Valid for the occlusions in sequence 2, where the distance to the camera is constant
            partial = ref_w is not None and (w < partial_ratio * ref_w or h < partial_ratio * ref_h)

            if not partial:
                small_stable_streak = 0
                ### UPDATE KALMAN ###
                kf.update([x + w / 2, y + h / 2])
                #####################
                # Grow fast, shrink slowly: occlusions only ever make the box SMALLER, so a bigger
                # box is real (robot fully in view / closer) -- e.g. the first detection is often
                # clipped by the image border and the old 10% rate needed ~35 frames to catch up
                if ref_w is None:
                    ref_w, ref_h = w, h
                else:
                    ref_w = 0.5 * ref_w + 0.5 * w if w > ref_w else 0.9 * ref_w + 0.1 * w
                    ref_h = 0.5 * ref_h + 0.5 * h if h > ref_h else 0.9 * ref_h + 0.1 * h
                print(f"Bounding box at {box}, centre u={x + w / 2:.1f} v={y + h / 2:.1f} px")
            else:
                small_stable_streak = small_stable_streak + 1 if prev_w is not None and abs(w - prev_w) <= 3 else 0
                if small_stable_streak >= 7:
                    # the box has been smaller but stable for a while -> the robot really got smaller
                    # (moved away), not occluded, so drop the reference size and re-seed it
                    ref_w = ref_h = None
                    small_stable_streak = 0
                    print("box stayed small, resetting the reference size")
                elif w >= min_visible * ref_w and h >= min_visible * ref_h:
                    # Partially hidden: measure the centre from the robot's visible edge instead
                    # of skipping the frame. Keeps the speed estimate right going into an
                    # occlusion and re-locks as soon as the robot starts to reappear
                    cu = _anchored_centre(x, w, ref_w, kf.state[0, 0]) if w < partial_ratio * ref_w else x + w / 2
                    cv = _anchored_centre(y, h, ref_h, kf.state[2, 0]) if h < partial_ratio * ref_h else y + h / 2
                    ### UPDATE KALMAN ###
                    kf.update([cu, cv])
                    #####################
                    print(f"partial detection {box}, centre from visible edge u={cu:.1f} v={cv:.1f} px")
                else:
                    print("detection too small to trust, skipping Kalman update")

        prev_w = box[2] if box is not None else None

        if kf.initialized and ref_w is not None:
            u, v = kf.state[0, 0], kf.state[2, 0]
            print(f"Kalmanfilter at,  u={u:.1f} v={v:.1f} px")
            # the state is the box CENTRE, so offset by half the size
            cv2.putText(output, "The red box is detection. The green is Kalman.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
            cv2.rectangle(output, (int(u - ref_w / 2), int(v - ref_h / 2)),
                          (int(u + ref_w / 2), int(v + ref_h / 2)), (0, 255, 0), 3)
            cv2.putText(output, f"u={u:.0f} v={v:.0f} px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("2D detections, with Kalman", output)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

#works on whole rgbd pictures
def detect_sequence3D(rgbd_frames, calib_path="stereo_calibration.json"):
    calib = load_calibration(calib_path)

    frames = iter(rgbd_frames)
    background = cv2.cvtColor(next(frames)[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)

    for rgbd in frames:
        rgb = cv2.cvtColor(rgbd[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        depth = rgbd[:, :, 3]
        output = rgb.copy()

        box = detect(rgb, background, diff_threshold=25, min_area=300)
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)

            xyz = box_to_xyz(depth, box, calib)
            if xyz is None:
                print(f"detected at {box} but depth is unreliable there")
            else:
                X, Y, Z = xyz
                print(f"detected at {box}, X={X:.2f} Y={Y:.2f} Z={Z:.2f} m")
                cv2.putText(output, f"X={X:.2f} Y={Y:.2f} Z={Z:.2f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("3D detections", output)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

#it doesn't produce whole rgbd after detecting the robot we extract it from the picture and than calculate sgbm over this region
#so it works faster and produces movei with higer fps rate
def detect_sequence3D_roi(image_pairs, calib_path="stereo_calibration.json"):
    calib = load_calibration(calib_path)

    pairs = iter(image_pairs)
    bg_l, bg_r = next(pairs)
    background, _ = compute_rectified_image(bg_l, bg_r, calib)

    for img_l, img_r in pairs:
        rect_l, _ = compute_rectified_image(img_l, img_r, calib)
        output = rect_l.copy()

        box = detect(rect_l, background, diff_threshold=25, min_area=300)
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)

            depth_roi = compute_depth_roi(img_l, img_r, calib, box)
            xyz = depth_to_xyz(depth_roi, x + w / 2, y + h / 2, calib)
            if xyz is None:
                print(f"detected at {box} but depth is unreliable there")
            else:
                X, Y, Z = xyz
                print(f"detected at {box}, X={X:.2f} Y={Y:.2f} Z={Z:.2f} m")
                cv2.putText(output, f"X={X:.2f} Y={Y:.2f} Z={Z:.2f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("3D detections (ROI)", output)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def detect_sequence3D_roi_WithKalman(image_pairs, calib_path="stereo_calibration.json", partial_ratio=0.9, min_visible=0.25):
    calib = load_calibration(calib_path)
    kf = Kalman3D()

    pairs = iter(image_pairs)
    bg_l, bg_r = next(pairs)
    background, _ = compute_rectified_image(bg_l, bg_r, calib)
    ref_wZ = ref_hZ = prev_w = None
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    small_stable_streak = 0

    for img_l, img_r in pairs:
        rect_l, _ = compute_rectified_image(img_l, img_r, calib)
        output = rect_l.copy()
        ### PREDICT KALMAN ###
        kxyz, kP = kf.predict()
        #####################
        box, mask = detect(rect_l, background, diff_threshold=25, min_area=300, return_mask=True)
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)

            depth_roi = compute_depth_roi(img_l, img_r, calib, box)
            m = cv2.erode(mask, erode_kernel)[y:y + h, x:x + w] > 0
            xyz = depth_to_xyz(depth_roi, x + w / 2, y + h / 2, calib, mask=m)

            if xyz is None:
                print(f"detected at {box} but depth is unreliable there")
            else:
                # w*Z and h*Z are the robot's size independent of distance; compare with the reference.
                # Use the filter's PREDICTED depth for this test once it has one: the measured depth of
                # a half-hidden robot is unreliable (SGBM picks up background next to the occluder's
                # edge), and a too-far Z makes a clipped box look full-size
                Z_meas = xyz[2]
                kZ = kf.state[4, 0]
                Zs = kZ if kZ > 0 and ref_wZ is not None else xyz[2]
                partial_w = ref_wZ is not None and w * Zs < partial_ratio * ref_wZ
                partial_h = ref_hZ is not None and h * Zs < partial_ratio * ref_hZ

                if not (partial_w or partial_h):
                    small_stable_streak = 0
                    ### UPDATE KALMAN ###
                    kxyz, kP = kf.update(xyz, measurement_variance(xyz, calib))
                    #####################
                    # Grow fast, shrink slowly: occlusions only ever make the box smaller
                    Z = xyz[2]
                    if ref_wZ is None:
                        ref_wZ, ref_hZ = w * Z, h * Z
                    else:
                        ref_wZ = 0.5 * ref_wZ + 0.5 * w * Z if w * Z > ref_wZ else 0.9 * ref_wZ + 0.1 * w * Z
                        ref_hZ = 0.5 * ref_hZ + 0.5 * h * Z if h * Z > ref_hZ else 0.9 * ref_hZ + 0.1 * h * Z

                    # # Original reference-size update: 10% per frame, same rate up and down
                    # ref_wZ = w * xyz[2] if ref_wZ is None else 0.9 * ref_wZ + 0.1 * w * xyz[2]
                    # ref_hZ = h * xyz[2] if ref_hZ is None else 0.9 * ref_hZ + 0.1 * h * xyz[2]
                    # X, Y, Z = xyz
                    # print(f"Bounding box at {box}")
                    # print(f"Detection at,    X={X:.2f} Y={Y:.2f} Z={Z:.2f} m")
                else:
                    small_stable_streak = small_stable_streak + 1 if prev_w is not None and abs(w - prev_w) <= 3 else 0
                    if small_stable_streak >= 7:
                        ref_wZ = ref_hZ = None
                        small_stable_streak = 0
                        print("box stayed small, resetting the reference size")
                    elif w * Zs >= min_visible * ref_wZ and h * Zs >= min_visible * ref_hZ:
                        # Partially hidden: re-measure the centre from the robot's visible edge (same
                        # as in 2D) and back-project it at the PREDICTED depth. The measured depth is
                        # not trusted here, so Z gets a huge variance: this update corrects X and Y only
                        full_w, full_h = ref_wZ / Zs, ref_hZ / Zs          # full size in px at this depth
                        kX, kY = kf.state[0, 0], kf.state[2, 0]
                        u_pred = calib["focal_px"] * kX / Zs + calib["cx"]
                        v_pred = calib["focal_px"] * kY / Zs + calib["cy"]
                        cu = _anchored_centre(x, w, full_w, u_pred) if partial_w else x + w / 2
                        cv = _anchored_centre(y, h, full_h, v_pred) if partial_h else y + h / 2
                        xyz = ((cu - calib["cx"]) * Zs / calib["focal_px"],
                               (cv - calib["cy"]) * Zs / calib["focal_px"], Zs)
                        var = measurement_variance(xyz, calib)
                        var[2] = 1e6
                        ### UPDATE KALMAN ###
                        kxyz, kP = kf.update(xyz, var)
                        #####################
                        print(f"partial detection {box}, centre from visible edge: "
                              f"X={xyz[0]:.2f} Y={xyz[1]:.2f} m (Z not used, measured {Z_meas:.2f} m)")
                    else:
                        print("detection too small to trust, skipping Kalman update")

        prev_w = box[2] if box is not None else None
        kX, kY, kZ = kxyz[0, 0], kxyz[2, 0], kxyz[4, 0]
        if kZ <= 0:
            # filter has no measurement yet (state starts at 0), nothing to project
            cv2.imshow("3D detections (ROI), with Kalman ", output)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            continue
        print(f"Kalmanfilter at, X={kX:.2f} Y={kY:.2f} Z={kZ:.2f} m")

        if ref_wZ is None or ref_hZ is None:
            # reference size was just reset and hasn't been reseeded by a detection yet
            cv2.imshow("3D detections (ROI), with Kalman ", output)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            continue

        u = calib["focal_px"] * kX / kZ + calib["cx"]
        v = calib["focal_px"] * kY / kZ + calib["cy"]
        w, h = ref_wZ / kZ, ref_hZ / kZ
        # the state is the box CENTRE, so offset by half the size
        cv2.putText(output, "The red box is detection. The green is Kalman.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
        cv2.rectangle(output, (int(u - w / 2), int(v - h / 2)), (int(u + w / 2), int(v + h / 2)), (0, 255, 0), 3)
        cv2.putText(output, f"X={kX:.2f} Y={kY:.2f} Z={kZ:.2f} m", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("3D detections (ROI), with Kalman ", output)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

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

    cv2.destroyAllWindows()

