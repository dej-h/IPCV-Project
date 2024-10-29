import cv2
import numpy as np
import imageio

# Camera intrinsic matrix (replace with actual calibration data)
K = np.array([
    [1126.00516, 0.0, 1006.32321],
    [0.0, 278.159008, 588.130689],
    [0.0, 0.0, 1.0]
])

# Rotation vector and translation vector (extrinsic parameters)
rvecs = np.array([
    [-0.25534736],
    [0.64746844],
    [0.98111009]
])
R, _ = cv2.Rodrigues(rvecs)  # Convert rotation vector to rotation matrix

tvecs = np.array([
    [10.53383823],
    [-18.79197468],
    [46.77667688]
])

def calculate_3d_points_with_depths(image, intrinsic_matrix, rotation_matrix, translation_vector):
    """
    Display an image, allow the user to select three points, and calculate their
    3D coordinates using depth information.

    Parameters:
    - image (np.array): Image frame for point selection.
    - intrinsic_matrix (np.array): 3x3 intrinsic matrix of the camera.
    - rotation_matrix (np.array): 3x3 rotation matrix of the camera.
    - translation_vector (np.array): 3x1 translation vector of the camera.

    Returns:
    - three_d_points (list): List of 3D coordinates of the selected points.
    - points (list): List of 2D points in the image corresponding to the selected points.
    """
    # List to store selected points
    points = []

    # Mouse callback to select points
    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Points", image)

    # Display the image and set the mouse callback
    cv2.imshow("Select Points", image)
    cv2.setMouseCallback("Select Points", select_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 3:
        raise ValueError("Please select exactly three points.")

    # Convert translation vector to column vector if not already
    translation_vector = np.array(translation_vector).reshape(-1, 1)

    # Invert intrinsic matrix
    K_inv = np.linalg.inv(intrinsic_matrix)

    # Calculate 3D coordinates using depth and camera model
    three_d_points = []
    for (u, v) in points:
        image_point_h = np.array([u, v, 1])
        camera_point = K_inv @ image_point_h  # Camera-space normalized coordinates

        # Compute depth scaling factor using extrinsics
        scale_factor = translation_vector[2] / (rotation_matrix[2, :] @ camera_point)
        world_point_3D = scale_factor * camera_point  # Convert to 3D coordinates

        three_d_points.append(world_point_3D)

    return three_d_points, points

def calculate_normal_vector_and_draw(image, three_d_points, intrinsic_matrix, rotation_matrix, translation_vector, points_2d):
    """
    Calculate the normal vector of the plane defined by three 3D points and draw it on the image
    starting from the first selected point.

    Parameters:
    - image (np.array): The image to draw the normal vector on.
    - three_d_points (list): List of three 3D points.
    - intrinsic_matrix (np.array): Intrinsic matrix of the camera.
    - rotation_matrix (np.array): Rotation matrix of the camera.
    - translation_vector (np.array): Translation vector of the camera.
    - points_2d (list): List of 2D points corresponding to three_d_points.
    """
    A, B, C = np.array(three_d_points)
    A_2D = points_2d[0]  # The first selected 2D point

    # Calculate vectors AB and AC
    AB = B - A
    AC = C - A

    # Calculate the normal vector using the cross product
    normal_vector = np.cross(AB, AC)
    normal_vector /= np.linalg.norm(normal_vector)  # Normalize

    # Define the endpoint of the normal vector in 3D (scaling for visualization)
    lambda_factor = 5  # Adjust length factor for display
    N_end_3D = A + lambda_factor * normal_vector

    # Projection matrix using rotation and translation (extrinsics)
    extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
    P = intrinsic_matrix @ extrinsic_matrix  # Complete projection matrix

    # Project N_end (endpoint of normal vector) back to 2D for visualization
    N_proj_2D_h = P @ np.append(N_end_3D, 1)
    N_proj_2D = (N_proj_2D_h[:2] / N_proj_2D_h[2]).astype(int)

    # Shift the vector to start from the first selected point (A_2D) in 2D
    direction_vector = np.array(N_proj_2D) - np.array(A_2D)
    shifted_N_proj_2D = (np.array(A_2D) + direction_vector).astype(int)

    # Draw the normal vector on the image starting from A_2D
    cv2.line(image, tuple(A_2D), tuple(shifted_N_proj_2D), (0, 0, 255), 2)  # Red line for normal vector

    # Draw the image with the normal vector
    cv2.imshow("Image with Normal Vector", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the video frame
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
    K1=np.array([[3081.60369932853, 0, 2029.56144828946],
        [0, 3079.96654929184, 1532.48975330644],
        [0, 0, 1 ]])
    R1 = np.array([
    [-0.04822042, 0.99448781, -0.09310636],
    [-0.8966135, -0.00201784, 0.4428094],
    [0.44018067, 0.10483287, 0.89176849]
    ])

# Define the translation vector T
    T1 = np.array([-100.45038532, 89.36738259, 631.51313513])


    frame2=cv2.imread('ImageTest/IMG_9007.JPG')
    # Calculate 3D points and get the 2D points used
    three_d_points, points_2d = calculate_3d_points_with_depths(frame2, K1, R1, T1)

    # Calculate and draw normal vector starting from the first selected point
    calculate_normal_vector_and_draw(frame2, three_d_points, K1, R1, T1, points_2d)
