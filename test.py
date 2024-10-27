import numpy as np
import cv2


def draw_lines(image):
    # Define the original front points
    point1 = np.array([1474, 351, 0])  # Front right point
    point2 = np.array([915, 189, 0])    # Front left point

    # Calculate the offset for the new upper points
    offset = 30
    new_point1 = np.array([point1[0], point1[1] - offset, 0])  # Upper point above point1
    new_point2 = np.array([point2[0], point2[1] - offset, 0])  # Upper point above point2

    # Draw lines between the original points and new upper points
   

    # Define the selected points for direction vector calculation
    selected_point1 = np.array([1244, 374])
    selected_point2 = np.array([1475, 351])

    # Calculate the direction vector from the selected points
    direction_vector = selected_point2 - selected_point1

    # Normalize the direction vector
    direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)

    # Define the length for the extended points
    line_length = 20

    # Store the extended points
    extended_points = []

    # Extend lines from the original points and new upper points
    for point in [point1, new_point1, point2, new_point2]:
        # Create a 3D direction vector with z = 0
        direction_vector_3d = np.array([direction_vector_normalized[0], direction_vector_normalized[1], 0])
        extend_point = point + direction_vector_3d * line_length
        extended_points.append(extend_point)  # Store the new points
        #cv2.line(image, tuple(point[:2].astype(int)), tuple(extend_point[:2].astype(int)), (0, 0, 255), 2)  # Red lines

    # Store all eight points
    all_points = np.array([point1, point2, new_point1, new_point2] + extended_points, dtype=np.int32)

    # Correctly define the plane points using np.array and ensure it's shaped correctly
    plane_points = np.array([all_points[0][:2], all_points[1][:2], all_points[7][:2], all_points[5][:2]], dtype=np.int32)
    shadow_points = np.array([all_points[0][:2], all_points[1][:2], all_points[6][:2], all_points[4][:2]], dtype=np.int32)

    # Fill the plane with white color
    cv2.fillConvexPoly(image, shadow_points, (0,0,0))
    cv2.fillConvexPoly(image, plane_points, (255, 255, 255))

    # Display the resulting image
    cv2.imshow('Image with Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return all_points  # Return all points


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

    all_points = draw_lines(frame)  # Call the function and store all points
    cap.release()  # Don't forget to release the capture
