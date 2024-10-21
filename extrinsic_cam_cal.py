import cv2
import numpy as np
from line_detection import dis_to_line, should_merge, merge_lines, intersection  # Import your provided functions

# Input video file (make sure to adjust the path)
video_path = "clips/clip1.mp4"
output_video_path = "output_video.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Camera matrix (intrinsic parameters from calibration)
camera_matrix = np.array([
    [663.03494796, 0, 322.09280073],
    [0, 664.02355522, 259.80140194],
    [0, 0, 1]
])

# Example distortion coefficients from calibration
dist_coeffs = np.array([-0.02185454, -0.16792912, -0.00770087, 0.00714092, 0.88086782])

# Process each frame
while True:
    # Read a frame from the video
    ret, img_og = cap.read()
    if not ret:
        break  # Exit the loop if no frames are left

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img_og, cv2.COLOR_BGR2HSV)

    # Define the green color range in HSV for the soccer field
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([60, 255, 255])

    # Create a binary mask for green areas
    binary_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw large contours
    large_contour_mask = np.zeros_like(binary_mask)

    # Filter contours by area
    min_area = 500  # Adjust this value as needed
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(large_contour_mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Convert the original image to grayscale for line detection
    gray = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY)

    # Use the large contour mask to filter the grayscale image
    filtered_gray = cv2.bitwise_and(gray, gray, mask=large_contour_mask)

    # Apply Canny edge detection
    edges = cv2.Canny(filtered_gray, 50, 150)

    # Use Hough Transform to detect lines
    rho = 1
    theta = np.pi / 180
    threshold = 250
    minLineLength = 200
    maxLineGap = 50
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    detected_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angle2 = (angle1 + 180) % 360  # Add 180 degrees for the opposite angle
            detected_lines.append({
                'p1': (x1, y1),
                'p2': (x2, y2),
                'angles': (angle1, angle2)
            })

    # Merge lines based on angle and distance thresholds
    merged_lines = merge_lines(detected_lines, angle_threshold=5, dis_threshold=50)

    # Create an image to draw the detected lines and intersection points
    lines_img = np.zeros_like(img_og)

    intersection_points = []  # To store intersection points
    # Find intersections
    for i in range(len(merged_lines)):
        for j in range(i + 1, len(merged_lines)):
            point = intersection(merged_lines[i], merged_lines[j])
            if point is not None:
                intersection_points.append(point)

    # Draw merged lines and intersection points
    for line in merged_lines:
        cv2.line(lines_img, line['p1'], line['p2'], (255, 0, 0), 2)  # Draw in red
    for point in intersection_points:
        cv2.circle(lines_img, point, 5, (0, 255, 0), -1)  # Draw circles at intersection points in green

    # Combine the original image with the detected lines and intersection points
    output_with_lines = cv2.addWeighted(img_og, 0.8, lines_img, 1, 0)

    # Example real-world 3D points (assign coordinates based on detected intersections)
    object_points = np.array([
        [0, 0, 0],       # Example real-world points
        [7.32, 0, 0],    # Adjust based on detected goalposts or field lines
        # Add more points based on detected features
    ], dtype=np.float32)

    # Example 2D image points from detected intersections
    if len(intersection_points) >= 2:
        image_points = np.array(intersection_points[:len(object_points)], dtype=np.float32)

        # Solve for extrinsic parameters
        retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

        print("Rotation Vector (rvec):\n", rvec)
        print("Translation Vector (tvec):\n", tvec)

    # Write the processed frame to the output video
    out.write(output_with_lines)

    # Display the processed frame (optional)
    cv2.imshow('Processed Frame', output_with_lines)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
