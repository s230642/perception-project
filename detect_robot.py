import os
import cv2
from glob import glob
import imutils
import numpy as np
from kalman import Kalman3D
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

#detects motion between two consecutive frames via dense optical flow, instead of diffing against
#a fixed background -- a genuinely static object never produces flow, regardless of lighting
#drift, so it can't get mistaken for the tracked target the way background subtraction can
def detect_flow(prev_frame, frame, mag_threshold=0.5, min_area=100, return_mask=False):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mask = (mag > mag_threshold).astype(np.uint8) * 255

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    # the inflatable dome's fabric visibly flexes, producing real sub-pixel motion up here
    # that's comparable in magnitude to the robot's own slow motion -- the robot never
    # appears above y=383 in this dataset, so cut well below that instead of detect()'s y=200
    mask[:350, :] = 0

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]

    if len(cnts) < 1:
        return (None, mask) if return_mask else None

    biggest = max(cnts, key=cv2.contourArea)
    box = cv2.boundingRect(biggest)
    return (box, mask) if return_mask else box

#sparse Lucas-Kanade flow restricted to a padded window around the previous detection, instead of
#dense flow over the whole frame -- much cheaper once a target is locked on. Falls back to
#searching the whole frame when prev_box is None (no lock yet, or the previous frame lost track)
def detect_flow_sparse(prev_frame, frame, prev_box, pad=60, mag_threshold=0.5, min_points=5, return_mask=False):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_h, img_w = prev_gray.shape

    if prev_box is not None:
        px, py, pw, ph = prev_box
        x0, y0 = max(0, px - pad), max(0, py - pad)
        x1, y1 = min(img_w, px + pw + pad), min(img_h, py + ph + pad)
    else:
        x0, y0, x1, y1 = 0, 0, img_w, img_h

    roi = prev_gray[y0:y1, x0:x1]
    corners = cv2.goodFeaturesToTrack(roi, maxCorners=200, qualityLevel=0.01, minDistance=7)
    if corners is None:
        return (None, None) if return_mask else None

    corners[:, 0, 0] += x0
    corners[:, 0, 1] += y0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None)
    status = status.reshape(-1).astype(bool)
    prev_pts = corners.reshape(-1, 2)[status]
    next_pts = next_pts.reshape(-1, 2)[status]

    disp = np.linalg.norm(next_pts - prev_pts, axis=1)
    moving = next_pts[disp > mag_threshold]

    if len(moving) < min_points:
        return (None, None) if return_mask else None

    xs, ys = moving[:, 0], moving[:, 1]
    box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1)

    if not return_mask:
        return box

    mask = np.zeros(prev_gray.shape, np.uint8)
    for mx, my in moving.astype(int):
        cv2.circle(mask, (mx, my), 5, 255, -1)
    return box, mask

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

#same as detect_sequence, but tracks motion frame-to-frame via sparse optical flow
#(detect_flow_sparse) instead of diffing each frame against a single fixed background. Once
#locked onto a box, each next frame only searches a padded window around it -- much cheaper
#than dense flow over the whole frame -- and re-searches the whole frame if track is lost
def detect_sequence_flow(image_set):
    frames = iter(image_set)
    prev = next(frames)
    prev_box = None

    for frame in frames:
        output = frame.copy()
        box = detect_flow_sparse(prev, frame, prev_box)

        if box is None:
            print("No movement was detected")
        else:
            x, y, width, height = box
            print(f"detected at ({x}, {y}, {width}, {height})")
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 0, 255), 3)

        cv2.imshow("flow detections", output)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        prev, prev_box = frame, box

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



def detect_sequence3D_roi_WithKalman(image_pairs, calib_path="stereo_calibration.json"):
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
            elif ref_wZ and (w * xyz[2] < 0.8 * ref_wZ or h * xyz[2] < 0.8 * ref_hZ):
                small_stable_streak = small_stable_streak + 1 if prev_w is not None and abs(w - prev_w) <= 3 else 0
                if small_stable_streak >= 7:
                    ref_wZ = ref_hZ = None
                    small_stable_streak = 0
                print("partial detection, skipping Kalman update")
            else:
                small_stable_streak = 0
                ### UPDATE KALMAN ###
                kxyz, kP = kf.update(xyz, measurement_variance(xyz, calib))
                #####################
                ref_wZ = w * xyz[2] if ref_wZ is None else 0.9 * ref_wZ + 0.1 * w * xyz[2]
                ref_hZ = h * xyz[2] if ref_hZ is None else 0.9 * ref_hZ + 0.1 * h * xyz[2]

                X, Y, Z = xyz
                print(f"Bounding box at {box}")
                print(f"Detection at,    X={X:.2f} Y={Y:.2f} Z={Z:.2f} m")

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


#same as detect_sequence3D_roi_WithKalman, but uses sparse optical flow (detect_flow_sparse)
#instead of background-diff detection -- keeps a separate function so both can still be compared
def detect_sequence3D_roi_WithKalman_flow(image_pairs, calib_path="stereo_calibration.json"):
    calib = load_calibration(calib_path)
    kf = Kalman3D()

    pairs = iter(image_pairs)
    bg_l, bg_r = next(pairs)
    prev_rect_l, _ = compute_rectified_image(bg_l, bg_r, calib)
    prev_box = None
    ref_wZ = ref_hZ = prev_w = None
    small_stable_streak = 0

    for img_l, img_r in pairs:
        rect_l, _ = compute_rectified_image(img_l, img_r, calib)
        output = rect_l.copy()
        ### PREDICT KALMAN ###
        kxyz, kP = kf.predict()
        #####################
        box = detect_flow_sparse(prev_rect_l, rect_l, prev_box)
        prev_rect_l, prev_box = rect_l, box

        if box is not None:
            x, y, w, h = box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)

            depth_roi = compute_depth_roi(img_l, img_r, calib, box)
            xyz = depth_to_xyz(depth_roi, x + w / 2, y + h / 2, calib)

            if xyz is None:
                print(f"detected at {box} but depth is unreliable there")
            elif ref_wZ and (w * xyz[2] < 0.8 * ref_wZ or h * xyz[2] < 0.8 * ref_hZ):
                small_stable_streak = small_stable_streak + 1 if prev_w is not None and abs(w - prev_w) <= 3 else 0
                if small_stable_streak >= 7:
                    ref_wZ = ref_hZ = None
                    small_stable_streak = 0
                print("partial detection, skipping Kalman update")
            else:
                small_stable_streak = 0
                ### UPDATE KALMAN ###
                kxyz, kP = kf.update(xyz, measurement_variance(xyz, calib))
                #####################
                ref_wZ = w * xyz[2] if ref_wZ is None else 0.9 * ref_wZ + 0.1 * w * xyz[2]
                ref_hZ = h * xyz[2] if ref_hZ is None else 0.9 * ref_hZ + 0.1 * h * xyz[2]

                X, Y, Z = xyz
                print(f"Bounding box at {box}")
                print(f"Detection at,    X={X:.2f} Y={Y:.2f} Z={Z:.2f} m")

        prev_w = box[2] if box is not None else None
        kX, kY, kZ = kxyz[0, 0], kxyz[2, 0], kxyz[4, 0]
        if kZ <= 0:
            # filter has no measurement yet (state starts at 0), nothing to project
            cv2.imshow("3D detections (ROI), flow + Kalman", output)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            continue
        print(f"Kalmanfilter at, X={kX:.2f} Y={kY:.2f} Z={kZ:.2f} m")

        if ref_wZ is None or ref_hZ is None:
            # reference size was just reset and hasn't been reseeded by a detection yet
            cv2.imshow("3D detections (ROI), flow + Kalman", output)
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

        cv2.imshow("3D detections (ROI), flow + Kalman", output)
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