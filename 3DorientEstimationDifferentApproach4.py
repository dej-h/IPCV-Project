import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio

video_path = "clips/clip1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    reader = imageio.get_reader(video_path)
    frame = reader.get_data(0)
else:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video.")
        cap.release()
    cap.release()
# Intrinsic matrix
K = np.array([
    [1126.00516, 0.0, 1006.32321],
    [0.0, 278.159008, 588.130689],
    [0.0, 0.0, 1.0]
])
image=frame
# Intrinsic matrix
K = np.array([
    [1126.00516, 0.0, 1006.32321],
    [0.0, 278.159008, 588.130689],
    [0.0, 0.0, 1.0]
])

# Extrinsic parameters: rotation vector and translation vector
rvecs = np.array([
    [-0.25534736],
    [0.64746844],
    [0.98111009]
])

tvecs = np.array([
    [10.53383823],
    [-18.79197468],
    [46.77667688]
])

# Convert rotation vector to rotation matrix
R, _ = cv2.Rodrigues(rvecs)

# Define the surface normal in 3D world coordinates
surface_normal_world = R @ np.array([0, 0, 1])  # Transform normal to world coordinates

# Define the start point (a point on the surface in 3D)
point_on_surface_world = tvecs.flatten()

# Define the end point of the line by extending the normal vector
line_length = 20  # Adjust this length to extend the line further
end_point_world = point_on_surface_world + line_length * surface_normal_world

# Project the start and end points of the line to 2D image coordinates
start_point_2d, _ = cv2.projectPoints(point_on_surface_world, rvecs, tvecs, K, None)
end_point_2d, _ = cv2.projectPoints(end_point_world, rvecs, tvecs, K, None)

# Convert points to integers for display
start_point_2d = tuple(start_point_2d[0][0].astype(int))
end_point_2d = tuple(end_point_2d[0][0].astype(int))

# Load the original image (replace with the actual path to your image)

# Draw the line on the image
cv2.line(image, start_point_2d, end_point_2d, (0, 0, 255), 2)  # Red line for the normal

# Display the image with the normal line
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.scatter([start_point_2d[0], end_point_2d[0]], [start_point_2d[1], end_point_2d[1]], color='blue')  # Mark points
plt.title("3D Orthogonal Line Projected on Image")
plt.show()