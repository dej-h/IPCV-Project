import cv2
import numpy as np

# Global points list to store selected points
points = []  # Initialize global points list

import numpy as np
import cv2

def project_3d_to_2d(point_3d, intrinsic_matrix, extrinsic_matrix):
    # Convert the 3D point to homogeneous coordinates
    point_3d_homogeneous = np.array([point_3d[0], point_3d[1], point_3d[2], 1]).reshape(4, 1)
    
    # Project the 3D point into the camera coordinate system using the extrinsic matrix
    camera_coordinates = np.dot(extrinsic_matrix, point_3d_homogeneous)  # 3x1 result
    
    # Project into the image plane using the intrinsic matrix
    image_coordinates_homogeneous = np.dot(intrinsic_matrix, camera_coordinates[:3])
    
    # Convert homogeneous coordinates to 2D by dividing by the z component
    x = image_coordinates_homogeneous[0] / image_coordinates_homogeneous[2]
    y = image_coordinates_homogeneous[1] / image_coordinates_homogeneous[2]
    
    return (int(x), int(y))  # Return as pixel coordinates

# Test the function with known 3D points and check if the output 2D points align
# Example usage (replace with your values):







# Define the mouse callback function
def select_point(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return  # Ignore all other events
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))  # Add the selected point
        # Draw a small circle at each selected point for visual feedback
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Points", image)  # Update image with selected point
        if len(points) == 4:  # Close window when 4 points are selected
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load image and check if it's loaded successfully
    image = cv2.imread("ImageTest/IMG_9020.JPG")
    if image is None:
        print("Error: Could not load image. Check the file path.")
        exit()

    # Set up the window and callback for point selection
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", select_point)

    # Keep the window open until 4 points are selected
    while len(points) < 4:
        cv2.imshow("Select Points", image)
        cv2.waitKey(1)
    print("Selected points:", points)

    # Define the camera intrinsic matrix (example values)
    K1 = np.array([[3081.60369932853, 0, 2029.56144828946],
                   [0, 3079.96654929184, 1532.48975330644],
                   [0, 0, 1]])

    # Define known 3D points in object space
    obj_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)

    # Use selected 2D image points
    img_points = np.array(points, dtype=np.float32)

    # Define distortion coefficients (if available, otherwise use zeros)
    dist_coeffs = np.zeros((4, 1))

    # Estimate the extrinsic parameters (rotation and translation vectors)
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K1, dist_coeffs)

    if success:
        # Convert rvec to rotation matrix (optional, for extrinsic matrix)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        extrinsic_matrix = np.hstack((rotation_matrix, tvec))
        print("Extrinsic Matrix:\n", extrinsic_matrix)
    else:
        print("Error: solvePnP failed to find a solution.")

    # Define a 3D point in world coordinates
    test_3d_point = (0, 0, 0)  # Replace with actual test point

    # Use the intrinsic matrix and extrinsic matrix calculated earlier
    projected_2d_point = project_3d_to_2d(test_3d_point, K1, extrinsic_matrix)
    print("Projected 2D point:", projected_2d_point)
    orthognal_3d_point=(0.5,0,-0.5)
    orthognal_2d_point= project_3d_to_2d(orthognal_3d_point, K1, extrinsic_matrix)
    print("orthognal_2d_point:", orthognal_2d_point)
    color = (0, 255, 0)  # Green color
    thickness = 10

    # Draw the line
    cv2.line(image, projected_2d_point, orthognal_2d_point, color, thickness)
    cv2.imshow('Orthognal',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()