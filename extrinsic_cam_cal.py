import cv2
import numpy as np
import json
import time
from line_detection import dis_to_line, should_merge, merge_lines, intersection  # Import your provided functions

def merge_close_intersections(points, distance_threshold=10):
    """Merge intersection points that are close to each other by averaging them."""
    merged_points = []
    used = set()  # Keep track of points we've already processed
    
    for i in range(len(points)):
        if i in used:
            continue
        # Get the first point
        p1 = points[i]
        cluster = [p1]
        
        for j in range(i + 1, len(points)):
            if j in used:
                continue
            p2 = points[j]
            # Calculate Euclidean distance between p1 and p2
            dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            if dist < distance_threshold:
                # If the points are close, add to the cluster
                cluster.append(p2)
                used.add(j)
        
        # Average the points in the cluster
        if len(cluster) > 1:
            avg_x = int(np.mean([p[0] for p in cluster]))
            avg_y = int(np.mean([p[1] for p in cluster]))
            merged_points.append((avg_x, avg_y))
        else:
            merged_points.append(p1)
        
        used.add(i)
    
    return merged_points

# Function to load camera parameters from JSON
def load_camera_parameters(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['distortion_coefficients'])
    return camera_matrix, dist_coeffs

# Input video file (make sure to adjust the path)
video_path = "clips/combined.mp4"
output_video_path = "output_video.mp4"
json_path = "intrinsic.json"  # Path to your JSON file

# Load camera parameters from JSON
camera_matrix, dist_coeffs = load_camera_parameters(json_path)

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

# Process each frame
while True:
    start_time = time.time()
    try:
        # Read a frame from the video
        ret, img_og = cap.read()
        if not ret:
            break  # Exit the loop if no frames are left

        # Resize frame for easier visualization
        img_resized = cv2.resize(img_og, (640, 360))  # Resizing for better display

        ### 1. Convert the image to HSV color space and create a binary mask
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([60, 255, 255])
        binary_mask = cv2.inRange(hsv, lower_green, upper_green)

        ### 2. Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty mask to draw large contours
        large_contour_mask = np.zeros_like(binary_mask)

        # Filter contours by area and draw the filtered contours
        min_area = 500  # Adjust this value as needed
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(large_contour_mask, [contour], -1, (255), thickness=cv2.FILLED)

        ### 3. Convert the original image to grayscale and apply Canny edge detection
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        filtered_gray = cv2.bitwise_and(gray, gray, mask=large_contour_mask)
        edges = cv2.Canny(filtered_gray, 50, 150)

        ### 4. Use Hough Transform to detect lines
        rho = 1
        theta = np.pi / 180
        threshold = 100
        minLineLength = 200
        maxLineGap = 50
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

        ### 5. Create an image to draw the raw Hough lines (before merging)
        hough_lines_img = np.zeros_like(img_resized)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw raw Hough lines in red

        ### 6. Merge lines based on angle and distance thresholds
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
        merged_lines = merge_lines(detected_lines, angle_threshold=10, dis_threshold=20)

        ### 7. Create an image to draw the merged lines and intersection points
        lines_img = np.zeros_like(img_resized)

        intersection_points = []  # To store intersection points
        # Find intersections
        for i in range(len(merged_lines)):
            for j in range(i + 1, len(merged_lines)):
                point = intersection(merged_lines[i], merged_lines[j])
                if point is not None:
                    intersection_points.append(point)

        # Merge close intersection points
        intersection_points = merge_close_intersections(intersection_points, distance_threshold=20)
        
        # Draw merged lines and intersection points
        for line in merged_lines:
            cv2.line(lines_img, line['p1'], line['p2'], (255, 0, 255), 2)  # Draw merged lines in magenta
        for point in intersection_points:
            cv2.circle(lines_img, point, 5, (255, 0, 0), -1)  # Draw circles at intersection points in blue
        # print the amount of intersection points found
        print(f"Intersection points found: {len(intersection_points)}")
        ### 8. Combine the original image with both the raw Hough lines and merged lines
        output_with_lines = cv2.addWeighted(img_resized, 0.8, lines_img, 1, 0)
        # Overlay raw Hough lines onto the same final output
        output_with_all_lines = cv2.addWeighted(output_with_lines, 1, hough_lines_img, 1, 0)

        ### 9. Convert Grayscale and Binary Masks to BGR for display
        contours_bgr = cv2.cvtColor(large_contour_mask, cv2.COLOR_GRAY2BGR)  # Convert mask to BGR for displaying
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to BGR for concatenation

        # Concatenate the images (2 rows, 2 columns: Green Mask, Canny Edges, Hough Lines, Final Output)
        row1 = cv2.hconcat([contours_bgr, edges_bgr])  # Green Mask and Canny Edges on the top row
        row2 = cv2.hconcat([hough_lines_img, output_with_all_lines])   # Raw Hough Lines and Final Output with all lines
        combined_output = cv2.vconcat([row1, row2])  # Combine the two rows vertically

        # Display the combined image
        cv2.imshow('All Steps Combined (Raw and Merged Lines)', combined_output)

        ### 10. Example real-world 3D points (assign coordinates based on detected intersections)
        object_points = np.array([
            [0, 0, 0],       # Example real-world points
            [7.32, 0, 0],    # Adjust based on detected goalposts or field lines
            # Add more points based on detected features
        ], dtype=np.float32)

        ### 11. Example 2D image points from detected intersections
        if len(intersection_points) >= 2:
            image_points = np.array(intersection_points[:len(object_points)], dtype=np.float32)

            # Solve for extrinsic parameters
            retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            print("Rotation Vector (rvec):\n", rvec)
            print("Translation Vector (tvec):\n", tvec)

        ### 12. Write the processed frame to the output video
        out.write(img_resized)

    except Exception as e:
        # Handle any exceptions and still show the combined output
        print(f"Error: {e}")
        cv2.imshow('All Steps Combined (Raw and Merged Lines)', combined_output)

    ### 13. Wait for user input or move to the next frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed
    
    # Make sure that you show each frame for 1 second
    time_taken = time.time() - start_time
    #time.sleep(max(0, 1 - time_taken))

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
