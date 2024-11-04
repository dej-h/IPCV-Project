import cv2
import numpy as np
import json
import time
from line_detection import dis_to_line, should_merge, merge_lines, intersection  # Import your provided functions
import itertools

import numpy as np

import numpy as np

def has_lines_on_sides(points, detected_lines, tolerance=5):
    """Check if each side of the quadrilateral has a corresponding line in detected lines within tolerance.
       Also, filter out cases where all four edges are aligned with a single detected line.
    """
    # Sort points based on x and y coordinates for consistency
    points = sorted(points, key=lambda pt: (pt[0], pt[1]))

    # Define the four edges of the quadrilateral (ignore diagonals)
    edges = [
        (points[0], points[1]),  # Top side
        (points[1], points[3]),  # Right side
        (points[3], points[2]),  # Bottom side
        (points[2], points[0])   # Left side
    ]

    def distance_point_to_line(point, line_start, line_end):
        """Calculate the perpendicular distance from a point to a line segment."""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        line_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if line_len == 0:
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        # Projection formula for distance from point to line segment
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len**2))
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)
        return np.sqrt((px - projection_x) ** 2 + (py - projection_y) ** 2)

    # Check if each edge is close enough to at least one detected line
    aligned_edges_count = {}
    for edge in edges:
        edge_aligned = False
        for line_index, line in enumerate(detected_lines):
            line_p1 = line['p1']
            line_p2 = line['p2']

            # Check distance from both endpoints of the edge to the line
            dist1 = distance_point_to_line(edge[0], line_p1, line_p2)
            dist2 = distance_point_to_line(edge[1], line_p1, line_p2)

            # If both endpoints are within the tolerance, the edge is aligned with this line
            if dist1 < tolerance and dist2 < tolerance:
                edge_aligned = True

                # Track how many edges align with each detected line
                if line_index in aligned_edges_count:
                    aligned_edges_count[line_index] += 1
                else:
                    aligned_edges_count[line_index] = 1
                break

        # If any edge is not aligned with any line, return False
        if not edge_aligned:
            return False

    # Check if all four edges are aligned with a single detected line
    if any(count == 4 for count in aligned_edges_count.values()):
        # All four edges align with a single detected line, so filter out this quadrilateral
        return False

    # If all edges are aligned with some line and no single line covers all edges, return True
    return True
import numpy as np

def has_lines_on_sides(points, detected_lines, tolerance=5):
    """Check if each side of the quadrilateral has a corresponding line in detected lines within tolerance.
       Also, filter out cases where all four edges are aligned with a single detected line.
    """
    # Sort points based on x and y coordinates for consistency
    points = sorted(points, key=lambda pt: (pt[0], pt[1]))

    # Define the four edges of the quadrilateral (ignore diagonals)
    edges = [
        (points[0], points[1]),  # Top side
        (points[1], points[3]),  # Right side
        (points[3], points[2]),  # Bottom side
        (points[2], points[0])   # Left side
    ]

    def distance_point_to_line(point, line_start, line_end):
        """Calculate the perpendicular distance from a point to a line segment."""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        line_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if line_len == 0:
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        # Projection formula for distance from point to line segment
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len**2))
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)
        return np.sqrt((px - projection_x) ** 2 + (py - projection_y) ** 2)

    # Check if each edge is close enough to at least one detected line
    aligned_edges_count = {}
    for edge in edges:
        edge_aligned = False
        for line_index, line in enumerate(detected_lines):
            line_p1 = line['p1']
            line_p2 = line['p2']

            # Check distance from both endpoints of the edge to the line
            dist1 = distance_point_to_line(edge[0], line_p1, line_p2)
            dist2 = distance_point_to_line(edge[1], line_p1, line_p2)

            # If both endpoints are within the tolerance, the edge is aligned with this line
            if dist1 < tolerance and dist2 < tolerance:
                edge_aligned = True

                # Track how many edges align with each detected line
                if line_index in aligned_edges_count:
                    aligned_edges_count[line_index] += 1
                else:
                    aligned_edges_count[line_index] = 1
                break

        # If any edge is not aligned with any line, return False
        if not edge_aligned:
            return False

    # Check if all four edges are aligned with a single detected line
    if any(count == 4 for count in aligned_edges_count.values()):
        # All four edges align with a single detected line, so filter out this quadrilateral
        return False

    # If all edges are aligned with some line and no single line covers all edges, return True
    return True




def get_top_middle_points(points, contours, image, lines, num_points=4, frame_width=1920, frame_height=1080, y_margin=0):
    # Select the largest contour (assuming the main object is the largest white region)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the topmost (smallest y) point in the largest contour
    min_y = min(largest_contour[:, :, 1])[0]

    # Filter points to those within the y_margin of the minimum y value
    topmost_points = [pt[0] for pt in largest_contour if min_y <= pt[0][1] <= min_y + y_margin]

    # Select the rightmost point among these topmost points
    highest_point = max(topmost_points, key=lambda pt: pt[0])

    # Sort the points by Euclidean distance to the highest point and select the closest ones
    points.sort(key=lambda pt: np.sqrt((pt[0] - highest_point[0]) ** 2 + (pt[1] - highest_point[1]) ** 2))
    
    # Select the closest points, handling cases where we have fewer than 2 * num_points
    closest_points = points[:min(4 * num_points, len(points))]

    # Generate combinations of four points
    candidate_groups = list(itertools.combinations(closest_points, 4))

    def is_quadrilateral(points):
        """Check if four points form a quadrilateral with opposite sides approximately parallel and equal in length."""
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        # Sort points based on x and y coordinates for consistency
        points = sorted(points, key=lambda pt: (pt[0], pt[1]))

        # Calculate distances for opposite sides
        d1 = distance(points[0], points[1])
        d2 = distance(points[2], points[3])
        d3 = distance(points[0], points[2])
        d4 = distance(points[1], points[3])

        d1d2avg = (d1 + d2) / 2
        d3d4avg = (d3 + d4) / 2

        # Check if opposite sides are approximately equal in length
        if d1 < 10 or d2 < 10 or d3 < 10 or d4 < 10:
            return False
        return abs(d1 - d2) < (d1d2avg * 0.10) and abs(d3 - d4) < (d3d4avg * 0.10)

    def center_of_quadrilateral(points):
        """Calculate the center point of a quadrilateral."""
        x_coords = [pt[0] for pt in points]
        y_coords = [pt[1] for pt in points]
        return (sum(x_coords) // 4, sum(y_coords) // 4)

    def isParallel(points):
        """Check if the sides of the quadrilateral are parallel."""
        points = sorted(points, key=lambda pt: (pt[0], pt[1]))

        # Calculate the slopes of the lines connecting the points
        m1 = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0] + 1e-5)  # add small value to avoid division by zero
        m2 = (points[3][1] - points[2][1]) / (points[3][0] - points[2][0] + 1e-5)
        m3 = (points[2][1] - points[0][1]) / (points[2][0] - points[0][0] + 1e-5)
        m4 = (points[3][1] - points[1][1]) / (points[3][0] - points[1][0] + 1e-5)

        return abs(m1 - m2) < 0.1 and abs(m3 - m4) < 0.1


    # Filter valid groups based on parallel and proportion checks
    valid_angle_groups = [group for group in candidate_groups if isParallel(group)]
    valid_proportion_groups = [group for group in valid_angle_groups if is_quadrilateral(group)]
    has_lines_on_sides_groups = [group for group in valid_proportion_groups if has_lines_on_sides(group, lines, tolerance=5)]
    print(f"Number of valid groups: {len(has_lines_on_sides_groups)}")
    # Calculate distances to the highest_point for each valid proportion group
    if not has_lines_on_sides_groups:
        return None
    
    distances = [(group, np.sqrt((center_of_quadrilateral(group)[0] - highest_point[0]) ** 2 + 
                                 (center_of_quadrilateral(group)[1] - highest_point[1]) ** 2)) for group in has_lines_on_sides_groups]

    # Sort the groups by distance to find the closest and furthest
    distances.sort(key=lambda x: x[1])

    # # Draw all groups with unique colors and annotate with distances
    # temp_image = image.copy()
    # num_groups = len(distances)
    # for i, (group, distance) in enumerate(distances):
    #     # Generate a color from red (closest) to blue (furthest)
    #     color = (int(255 * (i / num_groups)), 0, int(255 * (1 - i / num_groups)))
        
    #     # Draw the quadrilateral with the calculated color
    #     cv2.polylines(temp_image, [np.array(group)], True, color, 2)

    #     # Put the distance as text in the center of the quadrilateral
    #     center = center_of_quadrilateral(group)
    #     cv2.putText(temp_image, f"{distance:.1f}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # # Draw the highest_point point as a red circle
    # cv2.circle(temp_image, highest_point, 20, (0, 0, 255), -1)

    # # Show the final image with all groups and distances
    # cv2.imshow('All Groups with Distances', temp_image)
    # cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    # cv2.destroyAllWindows()

    # Return the closest group
    return distances[0][0] if distances else None

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
video_path = "clips/clip1.mp4"
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
print(f"Frame width: {frame_width}, Frame height: {frame_height}")
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# amout of frame
total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0
combined_output_array = []

line_thickness = 3
point_radius = 5
downsampled_width = 640
downsampled_height = 360
# Process each frame
while True:
    frame_count += 1
    start_time = time.time()
    try:
        # Read a frame from the video
        ret, img_og = cap.read()
        if not ret:
            break  # Exit the loop if no frames are left

        # Resize frame for easier visualization
        img_resized = cv2.resize(img_og,(downsampled_width,downsampled_height))  # Resizing for better display

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
        edges = cv2.Canny(filtered_gray, 20, 180)

        ### 4. Use Hough Transform to detect lines
        rho = 1
        theta = np.pi / 180
        threshold = 80
        minLineLength = 140
        maxLineGap = 100
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

        ### 5. Create an image to draw the raw Hough lines (before merging)
        hough_lines_img = np.zeros_like(img_resized)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_lines_img, (x1, y1), (x2, y2), (255, 0, 0), line_thickness)  # Draw raw Hough lines in red

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
        
        
        # Get top-right 4 points
        top_right_points = get_top_middle_points(intersection_points, contours, img_resized,merged_lines, num_points=4, frame_width=downsampled_width, frame_height=downsampled_height)

        # Draw merged lines and all intersection points
        for line in merged_lines:
            cv2.line(lines_img, line['p1'], line['p2'], (255, 0, 255), line_thickness)  # Draw merged lines in magenta
        
        for point in intersection_points:
            color = (255, 0, 0)  # Default color: blue
            if ( top_right_points != None) and point in top_right_points:
                color = (0, 255, 0)  # Color top-right points: green
            cv2.circle(lines_img, point, point_radius, color, -1)
        
        # Draw a white point at a more visible top-middle reference point
        #cv2.circle(lines_img, (downsampled_width // 2, 0), 50, (0, 255, 0), -1)

        
        # print the amount of intersection points found
        #print(f"Intersection points found: {len(intersection_points)}")
        ### 8. Combine the original image with both the raw Hough lines and merged lines
        output_with_lines = cv2.addWeighted(img_resized, 0.8, lines_img, 1, 0)
        # Overlay raw Hough lines onto the same final output
        output_with_all_lines = cv2.addWeighted(output_with_lines, 1, hough_lines_img, 1, 0)
        

        

        ### 9. Convert Grayscale and Binary Masks to BGR for display
        contours_bgr = cv2.cvtColor(large_contour_mask, cv2.COLOR_GRAY2BGR)  # Convert mask to BGR for displaying
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to BGR for concatenation


        # Select the largest contour (assuming the main object is the largest white region)
        y_margin = 10
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the topmost (smallest y) point in the largest contour
        min_y = min(largest_contour[:, :, 1])[0]

        # Filter points to those within the y_margin of the minimum y value
        topmost_points = [pt[0] for pt in largest_contour if min_y <= pt[0][1] <= min_y + y_margin]

        # Select the rightmost point among these topmost points
        highest_point = max(topmost_points, key=lambda pt: pt[0])

        # Draw a circle at the highest point found
        #cv2.circle(output_with_all_lines, highest_point, 20, (0, 255, 0), -1)
        # also draw circle on contour mask
        cv2.circle(contours_bgr, highest_point, 20, (0,255,0), -1)
            
        # now resize the images to 640x360
        
        contours_bgr = cv2.resize(contours_bgr, (640, 360))
        edges_bgr = cv2.resize(edges_bgr, (640, 360))
        hough_lines_img = cv2.resize(hough_lines_img, (640, 360))
        output_with_all_lines = cv2.resize(output_with_all_lines, (640, 360))
        
        
        # Concatenate the images (2 rows, 2 columns: Green Mask, Canny Edges, Hough Lines, Final Output)
        row1 = cv2.hconcat([contours_bgr, edges_bgr])  # Green Mask and Canny Edges on the top row
        row2 = cv2.hconcat([hough_lines_img, output_with_all_lines])   # Raw Hough Lines and Final Output with all lines
        combined_output = cv2.vconcat([row1, row2])  # Combine the two rows vertically
        combined_output_array.append(combined_output)
        # Display the combined image
        #cv2.imshow('All Steps Combined (Raw and Merged Lines)', combined_output)

        # ### 10. Example real-world 3D points (assign coordinates based on detected intersections)
        # object_points = np.array([
        #     [0, 0, 0],       # Example real-world points
        #     [7.32, 0, 0],    # Adjust based on detected goalposts or field lines
        #     # Add more points based on detected features
        # ], dtype=np.float32)

        # ### 11. Example 2D image points from detected intersections
        # if len(intersection_points) >= 2:
        #     image_points = np.array(intersection_points[:len(object_points)], dtype=np.float32)

        #     # Solve for extrinsic parameters
        #     retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        #     print("Rotation Vector (rvec):\n", rvec)
        #     print("Translation Vector (tvec):\n", tvec)

        ### 12. Write the processed frame to the output video
        out.write(img_resized)

    except Exception as e:
        # Handle any exceptions and still show the combined output
        print(f"Error: {e}")
        # throw the error higher
        
        #cv2.imshow('All Steps Combined (Raw and Merged Lines)', combined_output)

    ### 13. Wait for user input or move to the next frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed
    
    # Make sure that you show each frame for 1 second
    time_taken = time.time() - start_time
    if frame_count % 100 == 0:
        print(f"Frame {frame_count}/{total_frame_count} processed in {time_taken:.2f} seconds")
    #time.sleep(max(0, 0.5 - time_taken))

# Release resources
cap.release()
out.release()

# Video player to display combined output frames with controls
frame_index = 0
is_playing = False

while True:
    if frame_index < len(combined_output_array):
        cv2.imshow("Video Player", combined_output_array[frame_index])
    
    key = cv2.waitKey(30)  # Wait for 30ms, adjust for playback speed
    if key == ord('q'):  # Quit
        break
    elif key == ord('p'):  # Pause/Play
        is_playing = not is_playing
    elif key == ord('f') and frame_index < len(combined_output_array) - 1:  # Forward
        frame_index += 1
    elif key == ord('b') and frame_index > 0:  # Backward
        frame_index -= 1

    if is_playing and frame_index < len(combined_output_array) - 1:
        frame_index += 1  # Play the next frame

# Clean up
cv2.destroyAllWindows()



