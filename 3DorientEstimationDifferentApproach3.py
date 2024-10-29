import cv2
import numpy as np
import imageio

# Camera intrinsic matrix (replace with actual calibration data)
K = np.array([
    [1126.00516, 0.0, 1006.32321],
    [0.0, 278.159008, 588.130689],
    [0.0, 0.0, 1.0]
])

# Initialize global variable for storing selected points
points = []

def select_points(event, x, y, flags, param):
    """
    Mouse callback function to select points on the image.
    Stores up to four points for two lines (two points per line).
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)  # Draw green circles for selected points
        cv2.imshow("Select Points", param)

def calculate_vanishing_point(line1, line2):
    """
    Calculate the vanishing point from two lines in homogeneous coordinates.
    
    Parameters:
    - line1, line2 (tuples): Each line is defined by two points (x1, y1) and (x2, y2).
    
    Returns:
    - vanishing_point (np.array): The vanishing point in Cartesian coordinates or None if invalid.
    """
    # Convert lines to homogeneous coordinates (ax + by + c = 0)
    line1_h = np.cross(np.array([line1[0][0], line1[0][1], 1]), np.array([line1[1][0], line1[1][1], 1]))
    line2_h = np.cross(np.array([line2[0][0], line2[0][1], 1]), np.array([line2[1][0], line2[1][1], 1]))
    
    # Vanishing point is the intersection of the two lines
    vanishing_point_h = np.cross(line1_h, line2_h)
    
    # Check if the homogeneous coordinate is zero to prevent NaN
    if abs(vanishing_point_h[2]) < 1e-6:  # Threshold to handle near-infinity
        print("Warning: Vanishing point calculation resulted in a point at infinity.")
        return None
    
    # Convert to Cartesian coordinates
    vanishing_point = vanishing_point_h / vanishing_point_h[2]
    return vanishing_point

def calculate_normal_vector_from_vanishing_points(vanishing_point1, vanishing_point2, intrinsic_matrix):
    """
    Calculate the normal vector of the plane defined by two orthogonal vanishing points.
    
    Parameters:
    - vanishing_point1, vanishing_point2 (np.array): Vanishing points in Cartesian coordinates.
    - intrinsic_matrix (np.array): The camera's intrinsic matrix.
    
    Returns:
    - normal_vector (np.array): The normal vector of the plane defined by the vanishing points or None if invalid.
    """
    # Ensure both vanishing points are valid
    if vanishing_point1 is None or vanishing_point2 is None:
        print("Error: One or both vanishing points are invalid.")
        return None

    # Convert vanishing points to normalized camera coordinates by pre-multiplying with K_inv
    K_inv = np.linalg.inv(intrinsic_matrix)
    vp1_normalized = K_inv @ np.array([vanishing_point1[0], vanishing_point1[1], 1.0])
    vp2_normalized = K_inv @ np.array([vanishing_point2[0], vanishing_point2[1], 1.0])

    # Compute the normal vector as the cross product of the two vanishing directions
    normal_vector = np.cross(vp1_normalized, vp2_normalized)
    
    # Check if normal vector is valid
    if np.linalg.norm(normal_vector) == 0:
        print("Error: Normal vector calculation failed due to parallel vanishing directions.")
        return None
    
    # Normalize the result
    normal_vector /= np.linalg.norm(normal_vector)
    return normal_vector

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

    # Show image and let the user select four points
    cv2.imshow("Select Points", frame)
    cv2.setMouseCallback("Select Points", select_points, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Check if exactly four points were selected
    if len(points) != 4:
        print("Error: Please select exactly four points (two for each line).")
    else:
        # Use selected points to define two lines
        line1 = [points[0], points[1]]  # First line
        line2 = [points[2], points[3]]  # Second line (orthogonal to the first in real-world)

        # Calculate vanishing points for the two lines
        vanishing_point1 = calculate_vanishing_point(line1, line1)
        vanishing_point2 = calculate_vanishing_point(line2, line2)

        # Ensure both vanishing points are valid before proceeding
        if vanishing_point1 is None or vanishing_point2 is None:
            print("Error: Unable to calculate vanishing points correctly. Please select different lines.")
        else:
            # Calculate normal vector using vanishing points
            normal_vector = calculate_normal_vector_from_vanishing_points(vanishing_point1, vanishing_point2, K)

            if normal_vector is not None:
                # Display the normal vector
                print("Estimated normal vector (orthogonal direction):", normal_vector)

                # For visualization: draw the normal vector from the first point on line1
                start_point = line1[0]
                scale = 50  # Scaling factor for visualization in 2D
                end_point = (
                    int(start_point[0] + scale * normal_vector[0]),
                    int(start_point[1] + scale * normal_vector[1])
                )

                # Draw the lines and normal vector on the image
                cv2.line(frame, line1[0], line1[1], (255, 0, 0), 2)  # Line 1 in blue
                cv2.line(frame, line2[0], line2[1], (0, 255, 0), 2)  # Line 2 in green
                cv2.line(frame, start_point, end_point, (0, 0, 255), 2)  # Normal vector in red

                # Display the result
                cv2.imshow("Normal Vector Visualization", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Error: Failed to compute a valid normal vector.")
