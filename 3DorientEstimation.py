import cv2
import numpy as np




# Camera intrinsic matrix K (example values, replace with actual calibration data)
K = np.array([
    [1126.00516, 0.0, 1006.32321],
    [0.0, 278.159008, 588.130689],
    [0.0, 0.0, 1.0]
])

# Inverse of intrinsic matrix
K_inv = np.linalg.inv(K)

def select_points(event, x, y, flags, param):
    global points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # Display the point selected by the user
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green circle for selected points
        cv2.imshow('Select Points', image)

# Load the image
image = cv2.imread('ImageTest/testImage.png')

# Resize the image to fit the window size
screen_res = 1280, 720
scale_width = screen_res[0] / image.shape[1]
scale_height = screen_res[1] / image.shape[0]
scale = min(scale_width, scale_height)
window_width = int(image.shape[1] * scale)
window_height = int(image.shape[0] * scale)
image = cv2.resize(image, (window_width, window_height))

cv2.imshow('Select Points', image)

# Allow the user to select two lines and one point on the surface
points = []
cv2.setMouseCallback('Select Points', select_points)

print("Please select 3 points: two for a line and one on the surface.")
while len(points) < 3:
    cv2.waitKey(1)

cv2.destroyWindow('Select Points')

# Extract selected points
(x_A, y_A), (x_B, y_B), (x_C, y_C) = points

# 2D image points
A_2D = np.array([x_A, y_A, 1])
B_2D = np.array([x_B, y_B, 1])
C_2D = np.array([x_C, y_C, 1])

# Back-project to get 3D ray directions
A_ray = K_inv @ A_2D
B_ray = K_inv @ B_2D
C_ray = K_inv @ C_2D

# Assume an arbitrary depth (e.g., d = 1)
depth = 1
A_3D = A_ray * depth
B_3D = B_ray * depth
C_3D = C_ray * depth

# Calculate vectors AB and AC
AB = B_3D - A_3D
AC = C_3D - A_3D

# Calculate the normal vector to the plane
N = np.cross(AB, AC)

# Choose a point on the plane, e.g., A_3D
lambda_factor = -10  # Length factor for visualization
N_end = A_3D + lambda_factor * N

# Projection matrix P (for simplicity, assuming identity for extrinsic, replace with actual [R|t])
P = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

# Project A_3D and N_end back to 2D
A_proj_homogeneous = P @ np.append(A_3D, 1)
N_proj_homogeneous = P @ np.append(N_end, 1)

# Convert homogeneous coordinates to Cartesian coordinates
A_proj_2D = A_proj_homogeneous[:2] / A_proj_homogeneous[2]
N_proj_2D = N_proj_homogeneous[:2] / N_proj_homogeneous[2]

# Draw the normal line in 2D
A_proj_2D = tuple(A_proj_2D.astype(int))
N_proj_2D = tuple(N_proj_2D.astype(int))
cv2.line(image, A_proj_2D, N_proj_2D, (0, 0, 255), 2)  # Red for the normal line

# Show the result
cv2.imshow('Image with Normal Line', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
