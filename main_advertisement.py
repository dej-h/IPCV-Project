import cv2
import numpy as np
import time
from line_detection import merge_lines, intersection   # Import your provided functions
from banner_placement import draw_lines, track_corners
from point_detection import get_top_middle_points, merge_close_intersections, load_camera_parameters, video_player, order_points, smooth_points_exp, get_initial_points
from Dorient3Estimation import putBanner
# Input video file (make sure to adjust the path)
video_path = "clips/clip4.mp4"
banner_path = "banner.png"
output_video_path_points = "outputs/clip4_points.mp4"
output_video_path = "outputs/clip4_output.mp4"
json_path = "CamCalParam.json"  # Path to your JSON file

# drawing options
line_thickness = 3
point_radius = 5
# Downsample frame dimensions (for better line detection performance and results)
downsampled_width = 640
downsampled_height = 360

# ref points for the first frame
ref_points = [np.array([1195, 272]), np.array([915, 189]), np.array([950, 320]), np.array([700, 180])]
prev_points = ref_points
banner_img = cv2.imread(banner_path)
smoothed_display_points = None
# Load camera parameters from JSON
camera_matrix, dist_coeffs = load_camera_parameters(json_path)

# Get initial points by averaging over the first few frames
cap = cv2.VideoCapture(video_path)
ref_points = get_initial_points(cap, downsampled_width,downsampled_height, num_initial_frames=10)

 
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset the video to the start after initializing

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
out_points = cv2.VideoWriter(output_video_path_points, fourcc, fps, (frame_width, frame_height))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#scale the ref points by the downsampled ratio
horizontal_ratio = frame_width / downsampled_width
vertical_ratio = frame_height / downsampled_height
ref_points = [(int(point[0] * horizontal_ratio), int(point[1] * vertical_ratio)) for point in ref_points]

# amout of frames
total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0
combined_output_array = []
banner_placement_array = []



# banner placement variables
debug_mode = False
intial_distance = 50
current_distance = intial_distance
distance_step = 5
first_time = True
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
        

        ### 8. Combine the original image with both the raw Hough lines and merged lines
        output_with_lines = cv2.addWeighted(img_resized, 0.8, lines_img, 1, 0)
        # Overlay raw Hough lines onto the same final output
        output_with_all_lines = cv2.addWeighted(output_with_lines, 1, hough_lines_img, 1, 0)
        
        ### 9. point detection outputs
        # Convert Grayscale and Binary Masks to BGR for display
        contours_bgr = cv2.cvtColor(large_contour_mask, cv2.COLOR_GRAY2BGR)  # Convert mask to BGR for displaying
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to BGR for concatenation
        
        
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
        
        # Write the processed frame to the output video
        # put frame in correct dimension
        combined_output = cv2.resize(combined_output, (frame_width, frame_height))
        out_points.write(combined_output)
    
        # Virtual banner placement 
        # translate the top right points to the original image size
        horizontal_ratio = frame_width / downsampled_width
        vertical_ratio = frame_height / downsampled_height
        
        # if first_time and top_right_points != None:
        #     first_time = False
        #     ref_points = top_right_points
        
        if top_right_points == None:
            top_right_points = prev_points
        
        
        # Approach 2
        top_right_points = [(int(point[0] * horizontal_ratio), int(point[1] * vertical_ratio)) for point in top_right_points]

        top_right_points = order_points(top_right_points)
        found_corners, ref_points = track_corners(top_right_points, ref_points, max_distance=current_distance)
        if not found_corners:
            current_distance += distance_step
        else:
            current_distance = intial_distance

        # Smooth only for display purposes
        smoothed_display_points = smooth_points_exp(ref_points,smoothed_display_points)

        # Draw the lines and circles using smoothed display points
        image, all_points = draw_lines(img_og, smoothed_display_points, banner_img, debug_mode)

        # # Draw the top right points on the original image
        # for point in top_right_points:
        #     cv2.circle(image, point, point_radius, (0, 255, 0), -1)

        # Draw the smoothed display points as the reference points on the image
        # for point in smoothed_display_points:
        #     cv2.circle(image, tuple(point), point_radius, (0, 0, 255), -1)

        banner_placement_array.append(image)
        out.write(image)

        
        ## Approach 1
        # angle = 30
        # # restructure the top right points from top right -> bottom right -> bottom left -> top left
        # # Convert the points to a numpy array if they aren't already
        # top_right_points = [(int(point[0] * horizontal_ratio), int(point[1] * vertical_ratio)) for point in top_right_points]
        # top_right_points = np.array(top_right_points, dtype=np.float32)

        # # Sort points by y-coordinate to separate top and bottom points
        # top_right_points = sorted(top_right_points, key=lambda pt: pt[1])

        # # Split points into top and bottom based on sorted y-coordinates
        # top_points = sorted(top_right_points[:2], key=lambda pt: pt[0], reverse=True)  # Sort by x (right -> left)
        # bottom_points = sorted(top_right_points[2:], key=lambda pt: pt[0], reverse=True)  # Sort by x (right -> left)

        # # Arrange points in the specified order: top right, bottom right, bottom left, top left
        # ordered_points = np.array([top_points[0], bottom_points[0], bottom_points[1], top_points[1]])
        # print("Going into banner placement")
        # _,image = putBanner(img_og,ordered_points,angle,banner_img)
        # print("Coming out of banner placement")
        # banner_placement_array.append(image)
        # out.write(image)
        
    except Exception as e:
        # Handle any exceptions and still show the combined output
        print(f"Error: {e}")
        # throw the error higher
        raise e
        
        
        
        #cv2.imshow('All Steps Combined (Raw and Merged Lines)', combined_output)

    ### 13. Wait for user input or move to the next frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed
    
    # Make sure that you show each frame for 1 second
    time_taken = time.time() - start_time
    if frame_count % 25 == 0:
        print(f"Frame {frame_count}/{total_frame_count} processed in {time_taken:.2f} seconds")
    #time.sleep(max(0, 0.5 - time_taken))

# Release resources
cap.release()
out_points.release()

# video player banner placement
video_player(banner_placement_array,fps=fps)

# video player point detection
video_player(combined_output_array,fps=fps)






