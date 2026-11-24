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