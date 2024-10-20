import cv2
import numpy as np

def pre_process(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def get_lines(frame):
    pp = pre_process(frame)
    edges = cv2.Canny(pp, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    # Draw lines on image
    line_image = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return line_image


def get_contour(frame):
    _, threshold = cv2.threshold(pre_process(frame), 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = frame.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            cv2.drawContours(contour_image, [contour], -1, (0, 0, 255), 3)
    return contour_image