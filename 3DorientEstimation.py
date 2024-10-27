import cv2
import numpy as np
import imageio
# Replace these with your actual camera intrinsic parameters
fx = 1000  # Focal length in pixels along x-axis
fy = 1000  # Focal length in pixels along y-axis
cx = 640   # Principal point x-coordinate (usually image width / 2)
cy = 360   # Principal point y-coordinate (usually image height / 2)

# Camera intrinsic matrix K
K = np.array(
[
    [1126.00516, 0.0, 1006.32321],
    [0.0, 278.159008, 588.130689],
    [0.0, 0.0, 1.0]
  ]
    )

# Invert K
K_inv = np.linalg.inv(K)

# Global variables to store points
points = []
lines = []

def click_event(event, x, y, flags, param):
    global points, lines, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        points.append((x, y))
        if len(points) % 2 == 0:
            # We have two points, store the line
            cv2.line(img_display, points[-2], points[-1], (255, 0, 0), 2)
            lines.append((points[-2], points[-1]))
            cv2.putText(img_display, f'Line {len(lines)}', points[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow('Image', img_display)

# Load the image

video_path = "clips/clip1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    reader = imageio.get_reader('./IPCV-Project/clips/clip1.mp4')
    frame = reader.get_data(0)
else:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video.")
        cap.release()



img = frame
if img is None:
    print("Error loading image.")
    exit()

img_display = img.copy()

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', click_event)

print("Please click two points for Line 1, then two points for Line 2.")

while True:
    cv2.imshow('Image', img_display)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or len(lines) == 2:  # Esc key to exit, or when two lines are selected
        break

cv2.destroyAllWindows()

if len(lines) < 2:
    print("Not enough lines selected.")
    exit()

# Proceed with the computation
# Function to convert image points to camera coordinates
def image_to_camera(p):
    p_image = np.array([p[0], p[1], 1])
    p_cam = np.dot(K_inv, p_image)
    return p_cam

# For Line 1
p1_line1 = lines[0][0]  # First point of Line 1
p2_line1 = lines[0][1]  # Second point of Line 1

p1_cam_line1 = image_to_camera(p1_line1)
p2_cam_line1 = image_to_camera(p2_line1)

d1_cam = p2_cam_line1 - p1_cam_line1  # Direction vector in camera coordinates

# For Line 2
p1_line2 = lines[1][0]
p2_line2 = lines[1][1]

p1_cam_line2 = image_to_camera(p1_line2)
p2_cam_line2 = image_to_camera(p2_line2)

d2_cam = p2_cam_line2 - p1_cam_line2  # Direction vector in camera coordinates

# Compute the orthogonal direction
d3_cam = np.cross(d1_cam, d2_cam)

# Check for zero vector (lines are parallel or invalid)
if np.linalg.norm(d3_cam) == 0:
    print("The computed orthogonal direction is zero. Check if the lines are valid and not parallel.")
    exit()

# Project this direction back onto the image plane to get the vanishing point
v3 = np.dot(K, d3_cam)

# Normalize to get image coordinates
v3 = v3 / v3[2]

# Now, define the line in homogeneous coordinates passing through a point and the vanishing point
# Using the first point of Line 1 for reference
p1_image = np.array([p1_line1[0], p1_line1[1], 1])
v3_image = v3

line3 = np.cross(p1_image, v3_image)

# Now, find two points along line3 within the image boundaries
height, width, _ = img.shape

def compute_line_intersections(line, width, height):
    intersections = []

    # Line equation: a*x + b*y + c = 0
    a, b, c = line

    # Avoid division by zero
    epsilon = 1e-10

    # Left border x = 0
    x = 0
    if abs(b) > epsilon:
        y = - (a * x + c) / b
        if 0 <= y <= height - 1:
            intersections.append((int(x), int(y)))

    # Right border x = width -1
    x = width - 1
    if abs(b) > epsilon:
        y = - (a * x + c) / b
        if 0 <= y <= height - 1:
            intersections.append((int(x), int(y)))

    # Top border y = 0
    y = 0
    if abs(a) > epsilon:
        x = - (b * y + c) / a
        if 0 <= x <= width - 1:
            intersections.append((int(x), int(y)))

    # Bottom border y = height -1
    y = height - 1
    if abs(a) > epsilon:
        x = - (b * y + c) / a
        if 0 <= x <= width -1:
            intersections.append((int(x), int(y)))

    return intersections

intersections = compute_line_intersections(line3, width, height)

if len(intersections) >= 2:
    pointA = intersections[0]
    pointB = intersections[1]

    # Draw the line on the image
    img_result = img.copy()
    cv2.line(img_result, pointA, pointB, (0, 255, 0), 2)
    cv2.putText(img_result, 'Orthogonal Line', pointB, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow('Result', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not find enough intersections to draw the line.")
