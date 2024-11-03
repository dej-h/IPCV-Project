import cv2
import numpy as np
from numpy.linalg import lstsq
def dis_to_line(line, point):
    """Calculate the distance from a point to a line segment."""
    x1, y1, x2, y2 = line
    px, py = point
    
    # Calculate the line segment length
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # If the line segment length is zero, return the distance to the point
    if line_length == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Calculate the projection of point onto the line segment
    t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_length ** 2)
    t = np.clip(t, 0, 1)  # Restrict t to the range [0, 1]
    
    # Find the closest point on the line segment
    closest_point = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    
    # Calculate and return the distance
    return np.sqrt((px - closest_point[0]) ** 2 + (py - closest_point[1]) ** 2)

def average_distance_between_lines(line1, line2):
    """Calculate the average distance between two line segments."""
    # Get endpoints of the two lines
    x1, y1 = line1['p1']
    x2, y2 = line1['p2']
    x3, y3 = line2['p1']
    x4, y4 = line2['p2']
    
    # Calculate distances from each endpoint of line1 to line2
    d1 = dis_to_line((x3, y3, x4, y4), (x1, y1))
    d2 = dis_to_line((x3, y3, x4, y4), (x2, y2))
    
    # Calculate distances from each endpoint of line2 to line1
    d3 = dis_to_line((x1, y1, x2, y2), (x3, y3))
    d4 = dis_to_line((x1, y1, x2, y2), (x4, y4))
    
    # Return the average of these distances
    return (d1 + d2 + d3 + d4) / 4

def should_merge(basis_line, merged_line, angle_threshold, dis_threshold):
    """Determine if two lines should be merged based on their angles and average distance."""
    delta_angle = min(
        abs((basis_line['angles'][0] - merged_line['angles'][0]) % 360),
        abs((basis_line['angles'][1] - merged_line['angles'][0]) % 360),
        abs((basis_line['angles'][0] - merged_line['angles'][1]) % 360),
        abs((basis_line['angles'][1] - merged_line['angles'][1]) % 360)
    )

    if delta_angle < angle_threshold:
        avg_distance = average_distance_between_lines(basis_line, merged_line)
        if avg_distance < dis_threshold:
            return True
    return False

def fit_line(points):
    """Fit a line using least squares."""
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    A = np.vstack([x, np.ones(len(x))]).T
    m, c = lstsq(A, y, rcond=None)[0]  # Line: y = mx + c
    return m, c

def merge_lines(lines, angle_threshold, dis_threshold):
    """Merge lines that are close to each other and keep linear structure."""
    merged_lines = []
    visited = set()

    for i, line1 in enumerate(lines):
        if i in visited:
            continue
        merged_line = {'p1': line1['p1'], 'p2': line1['p2'], 'angles': line1['angles']}
        visited.add(i)

        points = [line1['p1'], line1['p2']]  # Collect points for fitting

        for j in range(i + 1, len(lines)):
            line2 = lines[j]
            if should_merge(merged_line, line2, angle_threshold, dis_threshold):
                # Only merge lines if both ends are close
                points.append(line2['p1'])
                points.append(line2['p2'])
                visited.add(j)

        # Fit a line to the merged points to ensure it's linear
        m, c = fit_line(points)
        x1, y1 = min(points, key=lambda p: p[0])  # Keep the structure by refitting
        x2, y2 = max(points, key=lambda p: p[0])

        merged_line['p1'] = (x1, int(m * x1 + c))
        merged_line['p2'] = (x2, int(m * x2 + c))

        merged_lines.append(merged_line)

    return merged_lines

def intersection(line1, line2):
    """Calculate the intersection point of two line segments."""
    x1, y1 = line1['p1']
    x2, y2 = line1['p2']
    x3, y3 = line2['p1']
    x4, y4 = line2['p2']

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # Check if lines are parallel
    if denom == 0:
        return None

    # Calculate intersection
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    
    # Check if the intersection is within both line segments
    if (min(x1, x2) <= px <= max(x1, x2) and
        min(y1, y2) <= py <= max(y1, y2) and
        min(x3, x4) <= px <= max(x3, x4) and
        min(y3, y4) <= py <= max(y3, y4)):
        return (int(px), int(py))
    
    return None
# only run if main file
if __name__ == "__main__":
    # Input video file
    video_path = "clips/clip4.mp4"  # Change this to your input video file path
    output_video_path = "output_video.mp4"  # Output video file path

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

    while True:
        # Read a frame from the video
        ret, img_og = cap.read()
        if not ret:
            break  # Exit the loop if no frames are left

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(img_og, cv2.COLOR_BGR2HSV)

        # Define the green color range in HSV
        lower_green = np.array([40, 40, 40])  # Lower bound for green
        upper_green = np.array([60, 255, 255])  # Upper bound for green

        # Create a binary mask for green areas
        binary_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty mask to draw large contours
        large_contour_mask = np.zeros_like(binary_mask)

        # Filter contours by area
        min_area = 500  # Minimum area to keep (you may adjust this value)
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(large_contour_mask, [contour], -1, (255), thickness=cv2.FILLED)

        # Convert the original image to grayscale for line detection
        gray = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY)

        # Use the large contour mask to filter the grayscale image
        filtered_gray = cv2.bitwise_and(gray, gray, mask=large_contour_mask)

        # Apply Canny edge detection
        edges = cv2.Canny(filtered_gray, 50, 150)

        # Use Hough Transform to detect lines
        rho = 1  # Distance resolution in pixels
        theta = np.pi / 180  # Angle resolution in radians
        threshold = 250  # Accumulator threshold
        minLineLength = 200  # Minimum length of a line
        maxLineGap = 50  # Maximum allowed gap between points on the same line

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

        detected_lines = []  # To store lines for merging
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

        # Merge lines based on angle and distance thresholds
        merged_lines = merge_lines(detected_lines, angle_threshold=5, dis_threshold=50)

        # Create an image to draw the detected lines and intersection points
        lines_img = np.zeros_like(img_og)

        intersection_points = []  # To store intersection points
        # Find intersections
        for i in range(len(merged_lines)):
            for j in range(i + 1, len(merged_lines)):
                point = intersection(merged_lines[i], merged_lines[j])
                if point is not None:
                    intersection_points.append(point)

        # Draw merged lines and intersection points
        for line in merged_lines:
            cv2.line(lines_img, line['p1'], line['p2'], (255, 0, 0), 2)  # Draw in red
        for point in intersection_points:
            cv2.circle(lines_img, point, 5, (0, 255, 0), -1)  # Draw circles at intersection points in green

        # Combine the original image with the detected lines and intersection points
        output_with_lines = cv2.addWeighted(img_og, 0.8, lines_img, 1, 0)

        # Write the processed frame to the output video
        out.write(output_with_lines)

        # Display the processed frame (optional)
        cv2.imshow('Processed Frame', output_with_lines)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit the loop if 'q' is pressed

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
