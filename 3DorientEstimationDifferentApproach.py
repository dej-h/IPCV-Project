import cv2
import numpy as np
import imageio



# Camera intrinsic matrix K (example values, replace with actual calibration data)
K = np.array(
[
    [1126.00516, 0.0, 1006.32321],
    [0.0, 278.159008, 588.130689],
    [0.0, 0.0, 1.0]
  ]
    )

rvecs=np.array([
    [-0.25534736],
    [0.64746844],
    [0.98111009]
    ])
R,_=	cv2.Rodrigues(rvecs)

tvecs = np.array([
    [10.53383823],
    [-18.79197468],
    [46.77667688]
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



image = frame
#image = cv2.imread('ImageTest/testImage.png')

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
A_2D = np.array([x_A, y_A, 0.67])
B_2D = np.array([x_B, y_B, 0.78])
C_2D = np.array([x_C, y_C, 0.51])

# Back-project to get 3D ray directions
A_ray = K_inv @ A_2D
B_ray = K_inv @ B_2D
C_ray = K_inv @ C_2D

# Assume an arbitrary depth (e.g., d = 1)
depth = 1
A_3D = A_ray * (y_B+y_C)/(y_A+y_B+y_C)
B_3D = B_ray * (y_A+y_C)/(y_A+y_B+y_C)
C_3D = C_ray * (y_A+y_B)/(y_A+y_B+y_C)

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


def calculate_depth_with_selection(image, intrinsic_matrix, rotation_matrix, translation_vector):
    """
    Display an image and allow the user to select three points, then calculate
    their depth (distance along the z-axis) relative to the camera.

    Parameters:
    - image (np.array): The image frame for point selection.
    - intrinsic_matrix (np.array): 3x3 intrinsic matrix of the camera.
    - rotation_matrix (np.array): 3x3 rotation matrix of the camera.
    - translation_vector (np.array): 3x1 translation vector of the camera.

    Returns:
    - depths (list): Depths of the selected points along the z-axis in meters.
    """
    # List to store selected points
    points = []

    # Define the mouse callback function to select points
    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green circle for selected points
            cv2.imshow("Select Points for depth", image)

    # Show the image and set the mouse callback
    cv2.imshow("Select Points", image)
    cv2.setMouseCallback("Select Points", select_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Verify the number of selected points
    if len(points) != 3:
        raise ValueError("Please select exactly three points.")

    # Convert translation vector to a column vector if it isn't already
    translation_vector = np.array(translation_vector).reshape(-1, 1)

    # Invert the intrinsic matrix for camera-space conversion
    K_inv = np.linalg.inv(intrinsic_matrix)

    depths = []
    for point in points:
        # Convert the point to homogeneous coordinates
        u, v = point
        image_point_h = np.array([u, v, 1])

        # Map to camera space (pre-multiply by K inverse)
        camera_point = K_inv @ image_point_h

        # Compute the depth using the extrinsic parameters
        scale_factor = translation_vector[2] / (rotation_matrix[2, :] @ camera_point)

        # Scale the normalized camera point to real-world coordinates
        world_point_3D = scale_factor * camera_point

        # The Z component of this point will be the depth along the z-axis
        depth = world_point_3D[2]
        depths.append(depth)

    return depths

if __name__ == "__main__":
    # Open the video file
    video_path = "clips/clip1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        # Try reading the first frame using imageio
        reader = imageio.get_reader(video_path)
        frame = reader.get_data(0)
    else:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video.")
            cap.release()
        cap.release()

    # Calculate depths by selecting points on the image
    depths = calculate_depth_with_selection(frame, K, R, tvecs)

    print("Depths (distances along z-axis from camera to each point):")
    for i, depth in enumerate(depths):
        print(f"Depth of Point {i + 1}: {depth:.2f} meters")