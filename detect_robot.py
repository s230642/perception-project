import os
import cv2
from glob import glob
import imutils


#returns tuple left, right top corners and width, height of box
def detect(frame, background, diff_threshold=100, min_area=100):
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
        return None

    biggest = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    return (x, y, w, h)

def detect_sequence(image_set):
    background = image_set[0]

    for image in image_set[1:]:
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

def detect_sequence3D(dataset_path):
    print("This function is not yet implemented")


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