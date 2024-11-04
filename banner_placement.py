import numpy as np
import cv2
from point_detection import order_points

def draw_lines(image, ref_points, ad_img, debug_mode=False):
    # Ensure all ref_points are NumPy arrays
    ref_points = [np.array(point) for point in ref_points]

    # Define the original front points
    point1 = ref_points[2][:2]
    point2 = ref_points[0][:2]
    selected_point1 = ref_points[3][:2]
    selected_point2 = ref_points[1][:2]

    # Calculate the offset for the new upper points
    offset = 30                                        
    new_point1 = np.array([point1[0], point1[1] - offset])
    new_point2 = np.array([point2[0], point2[1] - offset])

    # Calculate direction vectors using the selected points
    direction_vector1 = (point1 - selected_point1).astype(float)
    direction_vector2 = (point2 - selected_point2).astype(float)

    # Normalize the direction vectors
    norm1 = np.linalg.norm(direction_vector1)
    norm2 = np.linalg.norm(direction_vector2)
    if norm1 != 0:
        direction_vector1 /= norm1
    if norm2 != 0:
        direction_vector2 /= norm2

    # Define the length for the extended points
    line_length = 40
    #print(direction_vector1, direction_vector2)

    # Extend lines from original points and new upper points
    extended_points = [
        (point1 + direction_vector1 * line_length).astype(int),
        (new_point1 + direction_vector1 * line_length).astype(int),
        (point2 + direction_vector2 * line_length).astype(int),
        (new_point2 + direction_vector2 * line_length).astype(int)
    ]

    # Store all eight points
    all_points = np.array([point1, point2, new_point1, new_point2] + extended_points, dtype=int)

    # Define the plane points and shadow points
    plane_points = np.array([all_points[0], all_points[1], all_points[7], all_points[5]], dtype=int)
    shadow_points = np.array([all_points[0], all_points[1], all_points[6], all_points[4]], dtype=int)

    # Draw the black shadow region
    cv2.fillConvexPoly(image, shadow_points, (0, 0, 0))

    # Prepare to overlay the ad image on the plane
    ad_height, ad_width = ad_img.shape[:2]
    # flip the image if the is sideways
    if ad_height > ad_width:
        ad_img = cv2.rotate(ad_img, cv2.ROTATE_90_CLOCKWISE)
        ad_height, ad_width = ad_img.shape[:2]
        
    # flip the banner as it is upside down
    ad_img = cv2.flip(ad_img, 0)
        
    src_pts = np.array([[0, ad_height], [ad_width, ad_height], [ad_width, 0], [0, 0]], dtype=np.float32)
    dst_pts = np.array(plane_points, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_ad = cv2.warpPerspective(ad_img, matrix, (image.shape[1], image.shape[0]))

    # Create a mask for the ad region
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillConvexPoly(mask, plane_points, (255, 255, 255))

    # Blend the warped ad onto the original image in the plane region
    image_bg = cv2.bitwise_and(image, cv2.bitwise_not(mask))
    image = cv2.bitwise_or(image_bg, warped_ad)
    
    return image, all_points  # Return all points as integers

def track_corners(intersection_points, ref_points, max_distance=50):
    frame_tracked_points = []
    new_point_counter = 0
    used_points = set()  # Keep track of intersection points that have been used

    for prev_point in ref_points:
        # Find points within max_distance from the current reference point
        possible_points = [
            pt for pt in intersection_points
            if np.linalg.norm(np.array(prev_point) - np.array(pt)) <= max_distance
        ]
        
        # Select the closest point within the distance, else keep the previous point
        if possible_points:
            new_point = min(possible_points, key=lambda pt: np.linalg.norm(np.array(prev_point) - np.array(pt)))
            used_points.add(tuple(new_point))  # Mark this intersection point as used
        else:
            new_point = prev_point  # No update if no points within max_distance

        # Increment the counter if the point has changed from the previous
        if not np.array_equal(new_point, prev_point):
            new_point_counter += 1

        frame_tracked_points.append(new_point)

    # If exactly 3 points were successfully updated, set the 4th to the unused intersection point
    if new_point_counter == 3:
        # Find the remaining intersection point that hasn’t been used
        unused_points = [pt for pt in intersection_points if tuple(pt) not in used_points]
        if unused_points:
            # Update the fourth point to this unused intersection point
            for i, (point, prev_point) in enumerate(zip(frame_tracked_points, ref_points)):
                if np.array_equal(point, prev_point):  # Find the unmatched reference point
                    frame_tracked_points[i] = unused_points[0]  # Assign the unused intersection point
                    break
    
    # Return True if all points have been updated (4 points updated), else False
    if new_point_counter < 3:
        return False, frame_tracked_points
    else:
        return True, frame_tracked_points

if __name__ == "__main__":
    video_path = "./clips/clip2.mp4"
    img_path = "ad.png"
    ref_points = [np.array([1195, 272]), np.array([915, 189]), np.array([512, 330]), np.array([267, 235])]
    cap = cv2.VideoCapture(video_path)
    ad_img = cv2.imread(img_path)
    debug_mode = True
    if ad_img is None:
        print("Error: Could not load ad image.")
    elif not cap.isOpened():
        print("Error: Could not open video.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            intersection_points = get_intersection_points(frame)
            ref_points = track_corners(intersection_points, ref_points)
            image, all_points = draw_lines(frame, ref_points, ad_img, debug_mode)
            cv2.imshow("Tracked Video", image)
            if debug_mode:
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
