import numpy as np
import cv2
from line_detection import get_intersection_points

def draw_lines(image, ref_points):
    # Convert all points in ref_points to NumPy arrays if not already
    ref_points = [np.array(point) for point in ref_points]

    # Define the original front points
    point1 = ref_points[0][:2]  # Front right point in 2D
    point2 = ref_points[1][:2]  # Front left point in 2D

    # Calculate the offset for the new upper points
    offset = 30
    new_point1 = np.array([point1[0], point1[1] - offset])  # Upper point above point1
    new_point2 = np.array([point2[0], point2[1] - offset])  # Upper point above point2

    # Define the selected point for direction vector calculation
    selected_point1 = ref_points[2][:2]

    # Calculate the direction vector from the selected points
    direction_vector = point1 - selected_point1
    norm = np.linalg.norm(direction_vector)
    
    # Check for zero norm to avoid division by zero
    if norm == 0:
        direction_vector_normalized = np.array([0, 0])
    else:
        direction_vector_normalized = direction_vector / norm

    # Define the length for the extended points
    line_length = 20

    # Store the extended points
    extended_points = []

    # Extend lines from the original points and new upper points
    for point in [point1, new_point1, point2, new_point2]:
        extend_point = point + direction_vector_normalized * line_length
        extended_points.append(extend_point.astype(int))  # Store the new points as integers

    # Store all eight points
    all_points = np.array([point1, point2, new_point1, new_point2] + extended_points, dtype=int)

    # Define the plane points and shadow points
    plane_points = np.array([all_points[0], all_points[1], all_points[7], all_points[5]], dtype=int)
    shadow_points = np.array([all_points[0], all_points[1], all_points[6], all_points[4]], dtype=int)

    # Fill the plane and shadow regions with colors
    cv2.fillConvexPoly(image, shadow_points, (0, 0, 0))
    cv2.fillConvexPoly(image, plane_points, (255, 255, 255))

    return all_points  # Return all points as integers



def track_corners(intersection_points, ref_points, max_distance=40):
    frame_tracked_points = []

    # For each reference point, find the closest point within max_distance, or keep the previous point
    for prev_point in ref_points:
        # Find points within max_distance of the previous point
        possible_points = [
            pt for pt in intersection_points
            if np.linalg.norm(np.array(prev_point) - np.array(pt)) <= max_distance
        ]

        # Choose the closest point, or keep the previous point if none are within max_distance
        if possible_points:
            new_point = min(possible_points, key=lambda pt: np.linalg.norm(np.array(prev_point) - np.array(pt)))
        else:
            new_point = prev_point  # Retain the previous point if no match is found
            #print("no points found")
        frame_tracked_points.append(new_point)

    return frame_tracked_points


if __name__ == "__main__":
    video_path = "./clips/clip2.mp4"
    ref_points = [np.array([1195, 272]), np.array([915, 189]), np.array([512, 330])]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if there are no more frames

            # Retrieve intersection points for the current frame
            intersection_points = get_intersection_points(frame)
            print(intersection_points)
            # Track corners for this frame based on previous frame’s ref_points
            ref_points = track_corners(intersection_points, ref_points)

            # Draw lines using the updated ref_points and display the frame
            all_points = draw_lines(frame, ref_points)
            cv2.imshow("Tracked Video", frame)

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

