import numpy as np
import cv2
import json

def box(image):
    # Load camera parameters from JSON file
    with open('CamCalParam.json') as f:
        cam_params = json.load(f)

    camera_matrix = np.array(cam_params['cameraMatrix'])
    dist_coeffs = np.array(cam_params['distCoeffs']).flatten()
    rvec = np.array(cam_params['rvecs']).flatten()
    tvec = np.array(cam_params['tvecs']).flatten()

    # Pixel coordinates for the center of the box
    pixel_coordinates = np.array([1266, 263, 1])  # Homogeneous coordinates

    # Calculate the inverse camera projection to find the corresponding 3D point in world coordinates
    depth = 1.0  # Adjust depth based on your needs
    camera_point = np.linalg.inv(camera_matrix) @ pixel_coordinates  # Camera coordinates
    camera_point *= depth / camera_point[2]  # Scale by the depth value

    # Transform from camera coordinates to world coordinates
    rotation_matrix, _ = cv2.Rodrigues(rvec)  # Get the rotation matrix
    world_point = rotation_matrix.T @ (camera_point - tvec)  # Transform to world coordinates

    # Define the box dimensions
    box_length = 1.0  # meters
    box_height = 0.03  # meters
    # Calculate box vertices based on the world center
    direction_vector_pixel = np.array([559, 160, 0])
    direction_vector_normalized = direction_vector_pixel / np.linalg.norm(direction_vector_pixel)
    scale_factor = 1.0  # Adjust based on your scene scale
    direction_vector_3d = direction_vector_normalized * scale_factor
    orthogonal_vector = np.cross(direction_vector_3d, np.array([0, 0, 1]))
    orthogonal_vector_scaled = orthogonal_vector / np.linalg.norm(orthogonal_vector) * box_height
    direction_vector_scaled = direction_vector_3d * box_length / 2

    box_vertices = np.array([
        world_point - direction_vector_scaled - orthogonal_vector_scaled,
        world_point + direction_vector_scaled - orthogonal_vector_scaled,
        world_point + direction_vector_scaled + orthogonal_vector_scaled,
        world_point - direction_vector_scaled + orthogonal_vector_scaled,
        # Repeat for the top
        world_point - direction_vector_scaled - orthogonal_vector_scaled + np.array([0, 0, box_height]),
        world_point + direction_vector_scaled - orthogonal_vector_scaled + np.array([0, 0, box_height]),
        world_point + direction_vector_scaled + orthogonal_vector_scaled + np.array([0, 0, box_height]),
        world_point - direction_vector_scaled + orthogonal_vector_scaled + np.array([0, 0, box_height]),
    ])



    

    # Project the box vertices to 2D
    projected_points = []
    for vertex in box_vertices:
        # Transform from world coordinates to camera coordinates
        camera_coordinate = rotation_matrix @ vertex + tvec  # Correct transformation

        # Project to 2D
        print(camera_coordinate.shape)
        print(camera_matrix.shape)
        projected_point_3d = camera_matrix @ camera_coordinate  # Append 1 for homogeneous coordinates
        projected_point = projected_point_3d[:2] / projected_point_3d[2]  # Normalize

        # Apply distortion correction
        r2 = projected_point[0]**2 + projected_point[1]**2
        distorted_x = projected_point[0] * (1 + dist_coeffs[0]*r2 + dist_coeffs[1]*r2**2 + dist_coeffs[4]*r2**3) + \
                      dist_coeffs[2]*(2*projected_point[0]*projected_point[1]) + dist_coeffs[3]*(r2 + 2*projected_point[0]**2)
        distorted_y = projected_point[1] * (1 + dist_coeffs[0]*r2 + dist_coeffs[1]*r2**2 + dist_coeffs[4]*r2**3) + \
                      dist_coeffs[3]*(2*projected_point[0]*projected_point[1]) + dist_coeffs[2]*(r2 + 2*projected_point[1]**2)

        # Convert to integer coordinates (for drawing)
        projected_points.append(projected_point)

    projected_points = np.array(projected_points)
    print(projected_points)
    # Draw the box in the image
    for i in range(4):
        # Draw bottom edges
        cv2.line(image, tuple(projected_points[i].astype(int)), tuple(projected_points[(i + 1) % 4].astype(int)), (0, 255, 0), 2)
        # Draw top edges
        cv2.line(image, tuple(projected_points[i + 4].astype(int)), tuple(projected_points[(i + 1) % 4 + 4].astype(int)), (0, 255, 0), 2)
        # Draw vertical edges
        cv2.line(image, tuple(projected_points[i].astype(int)), tuple(projected_points[i + 4].astype(int)), (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow('Image with 3D Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "./clips/clip2.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video.")
            cap.release()
    
    box(frame)
    cap.release()  # Don't forget to release the capture


